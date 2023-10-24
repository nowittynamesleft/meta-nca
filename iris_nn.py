import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as torchdata
import sklearn
from sklearn import datasets
from sklearn.decomposition import PCA
import wandb
import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from reparam_module import ReparamModule
#import torch.jit
#import torch.multiprocessing as mp
import threading
import queue
import random
import itertools
import kmapper


def create_weight_gif(layer_num, metaepoch, num_epochs, model_name='unnamed', gif_name='weights.gif', directory='visualizations/', save_prefix="no_prefix"):
    images = []
    for epoch in range(num_epochs):
        # Read each weight matrix as a jpg file
        images.append(imageio.imread(f'{directory}images/{save_prefix}_{model_name}_W_{layer_num}_metaepoch_{metaepoch}_epoch_{epoch}.jpg'))
    
    # Create the gif using the weight visualizations
    #imageio.mimsave(gif_name, images, fps=5)
    imageio.mimsave(directory + "/gifs/" + gif_name, images, duration=2)
    

def visualize_weights(weights, metaepoch, epoch, directory='visualizations/', model_name='unnamed', save_prefix='no_prefix'):
    curr_weights = [w.cpu().detach() for w in weights]
    for i, W in enumerate(curr_weights):
        save_string = f'{directory}images/{save_prefix}_{model_name}_W_{i}_metaepoch_{metaepoch}_epoch_{epoch}.jpg'
        print(save_string)
        plt.imshow(W, cmap='gray')
        plt.axis('off')
        plt.savefig(save_string)
        plt.close()

def compute_hidden_state_dims(weight_shape_list):
    num_layers = len(weight_shape_list)
    max_num_layer_weights = 0
    for weight_shape in weight_shape_list:
        curr_num_weights = weight_shape[0]*weight_shape[1]
        if curr_num_weights > max_num_layer_weights:
            max_num_layer_weights = curr_num_weights
    #import ipdb; ipdb.set_trace()
    layer_encoding_dim = int(torch.ceil(torch.log(torch.tensor(num_layers))/torch.log(torch.tensor(2))).item())
    #layer_encoding_dim = int(np.ceil(np.log(num_layers)/np.log(2)))

    weight_encoding_dim = int(torch.ceil(torch.log(torch.tensor(max_num_layer_weights))/torch.log(torch.tensor(2))).item())
    #weight_encoding_dim = int(np.ceil(np.log(max_num_layer_weights)/np.log(2)))
    return layer_encoding_dim, weight_encoding_dim

def get_weight_shapes(in_units, out_units, num_layers, hidden_size):
    # for now, just how many layers of hidden_size x hidden_size
    if num_layers == 1:
        return [(in_units, out_units)]
    shapes = []
    shapes.append((in_units, hidden_size))
    for i in range(0, num_layers-2):
        shapes.append((hidden_size, hidden_size))
    shapes.append((hidden_size, out_units))
    return shapes

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 16)
        self.fc3 = nn.Linear(16, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x


class Model(nn.Module):
    def __init__(self, in_units, out_units, num_layers, prop_cells_updated, hidden_size, hidden_state_layer_dim=None, hidden_state_weight_dim=None, device=None):
        super(Model, self).__init__()
        self.in_units = in_units
        self.out_units = out_units
        self.num_layers = num_layers
        self.prop_cells_updated = prop_cells_updated
        self.hidden_size = hidden_size
        self.device = device
        self.hidden_state_layer_dim = None # set it to none at first
        self.hidden_state_weight_dim = None # set it to none at first
        if hidden_state_layer_dim is not None: # if Model was not provided with this, then set them
            self.hidden_state_layer_dim = hidden_state_layer_dim
            self.hidden_state_weight_dim = hidden_state_weight_dim
        self.reset_weight()

    def forward(self, x):
        prev_layer = x
        for i in range(self.num_layers):
            #linear = torch.matmul(prev_layer, self.weights[i]) + self.bias[i]
            linear = torch.matmul(prev_layer, self.weights[i]) # no bias for now
            prev_layer = F.relu(linear)
        return F.softmax(prev_layer, dim=-1)
        return x

    def encode_weight_hidden_states(self, weights):
        # binary encoding
        if self.hidden_state_layer_dim is None:
            weight_shape_list = [weight.shape for weight in weights]
            layer_encoding_dim, weight_encoding_dim = compute_hidden_state_dims(weight_shape_list)
        else:
            layer_encoding_dim = self.hidden_state_layer_dim
            weight_encoding_dim = self.hidden_state_weight_dim
        total_encoding_dim = layer_encoding_dim + weight_encoding_dim
        hidden_states = []
        for layer_num, weight_mat in enumerate(weights):
            encodings = torch.zeros(weight_mat.shape[0], weight_mat.shape[1], total_encoding_dim, device=self.device)
            #encodings = torch.ones(weight_mat.shape[0], weight_mat.shape[1], total_encoding_dim, device=self.device)
            encodings[:, :, :layer_encoding_dim] = self.get_binary_encoding_vector(layer_num, layer_encoding_dim)
            k = 0
            for i in range(weight_mat.shape[0]):
                for j in range(weight_mat.shape[1]):
                    encodings[i, j, layer_encoding_dim:] = self.get_binary_encoding_vector(k, weight_encoding_dim)
                    k += 1
            hidden_states.append(encodings)
        #import ipdb; ipdb.set_trace()
        return hidden_states

    def get_binary_encoding_vector(self, number, encoding_dim):
        binary = bin(number)[2:]
        diff = encoding_dim - len(binary)
        if diff > 0:
            binary = diff*'0' + binary
        return torch.tensor(np.fromstring(','.join(list(binary)), sep=',', dtype=int), device=self.device)

    def reset_weight(self):
        #import ipdb; ipdb.set_trace()
        #w[:, 0] = 1.0
        #b = torch.zeros(self.out_units,)
        #b[0] = 1.0
        #w = torch.randn(self.in_units, self.out_units)
        self.weights = []
        self.hidden_states = []
        for l in range(self.num_layers):
            if self.num_layers == 1:
                w = nn.init.xavier_uniform_(torch.empty(size=(self.in_units, self.out_units))).to(self.device)
                #w = torch.zeros(self.in_units, self.out_units).to(self.device)
                #w = torch.ones(self.in_units, self.out_units).to(self.device)
            elif l == 0:
                w = nn.init.xavier_uniform_(torch.empty(size=(self.in_units, self.hidden_size))).to(self.device)
                #w = torch.zeros(self.in_units, self.hidden_size).to(self.device)
                #w = torch.ones(self.in_units, self.hidden_size).to(self.device)
            elif l == self.num_layers - 1:
                w = nn.init.xavier_uniform_(torch.empty(size=(self.hidden_size, self.out_units))).to(self.device)
                #w = torch.zeros(self.hidden_size, self.out_units).to(self.device)
                #w = torch.ones(self.hidden_size, self.out_units).to(self.device)
            else:
                w = nn.init.xavier_uniform_(torch.empty(size=(self.hidden_size, self.hidden_size))).to(self.device)
                #w = torch.zeros(self.hidden_size, self.hidden_size).to(self.device)
                #w = torch.ones(self.hidden_size, self.hidden_size).to(self.device)
            self.weights.append(nn.Parameter(w, requires_grad=False))
            #self.weights.append(nn.Parameter(w))
        '''
            if self.hidden_state_dim is not None:
                hidden_state = torch.zeros(w.shape[0], w.shape[1], self.hidden_state_dim)
                k = 0
                for i in range(w.shape[0]):
                    for j in range(w.shape[1]):
                        hidden_state[i,j,k] = 1.0 # globally distinct states for intialization
                        k += 1
            self.hidden_states.append(hidden_state)
        '''
        
        #self.weight = nn.Parameter(w, requires_grad=False).to(self.device)
        #self.bias = nn.Parameter(b, requires_grad=False).to(self.device)
        self.hidden_states = self.encode_weight_hidden_states(self.weights)


class MetaNCA(nn.Module):
    def __init__(self, in_units, out_units, num_layers=1, num_models=1, classification_nn_hidden_size=5, prop_cells_updated=1.0, sample_archs=False, device=None):
        super().__init__()
        self.device = device
        self.in_units = in_units
        self.out_units = out_units
        self.num_layers = num_layers
        self.prop_cells_updated = prop_cells_updated
        self.local_update = True
        self.classification_nn_hidden_size = classification_nn_hidden_size

        #self.local_nn = nn.Linear(in_units+out_units, 1)
        self.models = []
        if sample_archs:
            in_units_list = [in_units]
            out_units_list = [out_units]
            num_layers_list = [1, 2, 3, 4, 5]
            prop_cells_updated_list = [prop_cells_updated]
            hidden_units_list = [2, 5, 10]
            archs = self.sample_architectures(num_models, in_units_list, out_units_list, num_layers_list, prop_cells_updated_list, hidden_units_list)
            # archs is a list of tuples of arguments to instantiate Models
            max_layer_state_dim = 0
            max_weight_state_dim = 0
            for arch in archs:
                weight_shape_list = get_weight_shapes(arch[0], arch[1], arch[2], arch[4])
                layer_hidden_state_dim, weight_hidden_state_dim = compute_hidden_state_dims(weight_shape_list)
                max_layer_state_dim = max(layer_hidden_state_dim, max_layer_state_dim)
                max_weight_state_dim = max(weight_hidden_state_dim, max_weight_state_dim)
            for arch in archs:
                self.models.append(Model(*arch, hidden_state_layer_dim=max_layer_state_dim, hidden_state_weight_dim=max_weight_state_dim, device=device))
            print(archs)
            self.hidden_state_dim = max_layer_state_dim + max_weight_state_dim
        else:
            [self.models.append(Model(self.in_units, self.out_units, self.num_layers, self.prop_cells_updated, self.classification_nn_hidden_size, device=device)) for i in range(num_models)]
            self.hidden_state_dim = self.models[0].hidden_states[0].shape[-1]
        local_nn_hidden_size = 10
        #self.hidden_state_dim = self.models[0].hidden_states[0].shape[-1]
        hidden_state_shape_list = [model.hidden_states[0].shape[-1] for model in self.models]
        '''
        self.local_nn = nn.Sequential(
            #nn.Linear(in_units+out_units + 1, hidden_size), # plus 1 for bias
            nn.Linear(in_units+out_units, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        ).to(device)
        '''
        self.local_nn = nn.Sequential(
            nn.Linear(3 + 3*self.hidden_state_dim, local_nn_hidden_size), # self, sum of forward connected weights, sum of backward connected weights, self hidden state, forward hidden state sum, backward hidden state sum
            nn.ReLU(),
            nn.Linear(local_nn_hidden_size, local_nn_hidden_size),
            nn.ReLU(),
            nn.Linear(local_nn_hidden_size, 1 + self.hidden_state_dim),
            nn.Sigmoid()
        ).to(device)

    def sample_architectures(self, n, *args):
        """
        Sample n random architectures based on the provided argument values.

        Args:
        - n (int): Number of architectures to sample.
        - *args: Variable length argument list of iterables representing possible values for each parameter.

        Returns:
        - list: List of tuples, where each tuple represents a sampled architecture.
        """
        all_architectures = list(itertools.product(*args))
        return random.sample(all_architectures, n)


    def nca_local_rule(self, param, hidden_states): # does not yet take into account the data. Should it? What is a good way to do that? What about using the activations of the network as it is inputting a databatch?
        #updates = torch.zeros_like(param).to(self.device)
        torch.autograd.set_detect_anomaly(True)
        updates = torch.zeros(param.shape[0], param.shape[1], 1+hidden_states.shape[2]).to(self.device)
        #updates = torch.ones(param.shape[0], param.shape[1], 1+hidden_states.shape[2]).to(self.device)
        idx_in_units = torch.randperm(param.shape[0])[:int(param.shape[0]*self.prop_cells_updated)][:,None]
        idx_out_units = torch.randperm(param.shape[1])[:int(param.shape[1]*self.prop_cells_updated)][:,None]
        #import ipdb; ipdb.set_trace()
        for i in idx_in_units:
            for j in idx_out_units:
                #updates[i,j] = self.local_nn(torch.cat([param[i, :].flatten(), param[:, j].flatten(), self.bias[j]]))
                '''
                updates[i,j] = self.local_nn(torch.cat([param[i, :].flatten(), param[:, j].flatten()]))
                '''
                if i == 0:
                    forward = param[i+1:, j].flatten()
                    forward_states = hidden_states[i+1:,j,:]
                elif i == param.shape[0]:
                    forward = param[:i, j].flatten()
                    forward_states = hidden_states[:i,j,:]
                else:
                    forward = torch.cat((param[:i,j].flatten(), param[i+1:, j].flatten()))
                    forward_states = torch.cat((hidden_states[:i,j], hidden_states[i+1:, j]))
                if j == 0:
                    backward = param[i, j+1:].flatten()
                    backward_states = hidden_states[i,j+1:,:]
                elif j == param.shape[1]:
                    backward = param[i, :j].flatten()
                    backward_states = hidden_states[i,:j,:]
                else:
                    backward = torch.cat((param[i,:j].flatten(), param[i, j+1:].flatten()))
                    backward_states = torch.cat((hidden_states[i,:j], hidden_states[i, j+1:]), dim=1)
                concatted_weights = torch.cat((param[i,j], torch.mean(forward)[None], torch.mean(backward)[None]))
                concatted_states = torch.cat((hidden_states[i,j, :].flatten(), torch.mean(forward_states, dim=0).flatten(), torch.mean(backward_states, dim=1).flatten()))
                concatted = torch.cat((concatted_weights, concatted_states))
                #import ipdb; ipdb.set_trace()
                updates[i,j, :] = self.local_nn(concatted)
        weight_updates = updates[:,:,0] 
        hidden_state_updates = updates[:,:,1:]
        return weight_updates, hidden_state_updates
        #for j in idx_out_units:
        #    self.bias[j] += self.local_nn(torch.cat([torch.zeros((self.out_units,)), self.weight[:,j].flatten(), self.bias[j]]))

    def reset_models(self):
        for model in self.models:
            model.reset_weight()

    def reparametrize_weights(self):
        for model in self.models:
            for i in range(len(model.weights)):
                model.weights[i] = model.weights[i].detach().clone()

    def update_model(self, model):
        all_weight_updates = []
        all_hidden_state_updates = []
        for (weight, hidden_state) in zip(model.weights, model.hidden_states):
            weight_updates, hidden_state_updates = self.nca_local_rule(weight, hidden_state)

            all_weight_updates.append(weight_updates)
            all_hidden_state_updates.append(hidden_state_updates)
        # only update after all updates have been calculated
        for layer, (weight, hidden_state) in enumerate(zip(model.weights, model.hidden_states)):
            weight.copy_(weight + all_weight_updates[layer])
            #weight = weight + all_weight_updates[layer]
            hidden_state += all_hidden_state_updates[layer]

    def worker(self, model, X, results_queue):
        # for multiprocessing
        output = model(X)
        results_queue.put(output)

    def forward(self, X):
        # Random sample without replacement
        if self.local_update:
            for model in self.models:
               self.update_model(model)

            #processes = []
            #for model in self.models:
            #    p = threading.Thread(target=self.update_model, args=(model,))
            #    p.start()
            #    processes.append(p)
            #for p in processes:
            #    p.join()

            #futures_update = [torch.jit.fork(self.update_model, model) for model in self.models]
            # Wait for all updates to complete
            #[torch.jit.wait(f) for f in futures_update]


        return [model(X) for model in self.models]
        #processes = []
        #results_queue = queue.Queue()
        #for model in self.models:
        #    p = threading.Thread(target=self.worker, args=(model, X, results_queue))
        #    p.start()
        #    processes.append(p)
        #
        #results = [results_queue.get() for _ in self.models]

        #for p in processes:
        #    p.join()

        ## Now, the 'results' list contains the outputs from all models
        #return results

        #futures = [torch.jit.fork(model, X) for model in self.models]
        ## Use torch.jit.wait to collect the results
        #return [torch.jit.wait(f) for f in futures]


if __name__ == '__main__':
    #mp.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument('run_name')
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--sample_archs', action='store_true')
    parser.add_argument('--num_models', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--prop_cells_updated', type=float, default=0.8)

    args = parser.parse_args()

    if not args.no_log:
        wandb.init(project='meta_nca')
        wandb.run.name = args.run_name
    # Load the Iris dataset

    #device = 'cuda:1'
    device = 'cuda:0'
    #device = 'cpu' # gpu is slower...not sure why
    iris = datasets.load_iris()
    X = iris["data"]
    y = iris["target"]

    # Convert to PyTorch tensors and create a dataset
    X = torch.Tensor(X).to(device)
    y = torch.Tensor(y).long().to(device)
    dataset = torchdata.TensorDataset(X, y)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torchdata.random_split(dataset, [train_size, val_size])
    train_dataloader = torchdata.DataLoader(train_dataset, batch_size=len(train_dataset))
    val_dataloader = torchdata.DataLoader(val_dataset, batch_size=len(val_dataset))

    viz_dir = 'visualizations/'

    import imageio
    import os

    # Initialize the network
    regular_net = Net()
    regular_net = regular_net.to(device)
    #optimizer = optim.SGD(regular_net.parameters(), lr=0.001)
    optimizer = optim.Adam(regular_net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1000):
        loss = 0
        correct= 0
        optimizer.zero_grad()
        for (X,y) in train_dataloader:
            outputs = regular_net(X)
            loss += criterion(outputs, y)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).float().sum()
            
        loss.backward()
        accuracy = correct / train_size
        #wandb.log({'loss': loss.item(), 'accuracy': accuracy})
        optimizer.step()
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Loss {loss.item()}")
            print(f"Regular net Accuracy: {accuracy}")
            #import ipdb; ipdb.set_trace()
            val_loss = 0
            correct = 0
            for (X,y) in val_dataloader:
                outputs = regular_net(X)
                val_loss += criterion(outputs, y)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y).float().sum()
            val_accuracy = correct / val_size
            print(f"Val accuracy: {val_accuracy}")
    
    #net = MetaNCA(X.shape[1], y.max() + 1, device=device)
    num_layers = args.num_layers
    num_models = args.num_models
    #net = MetaNCA(X.shape[1], y.max() + 1, num_layers=num_layers, hidden_state_dim=5*5, prop_cells_updated=0.5, device=device)
    net = MetaNCA(X.shape[1], y.max() + 1, num_layers=num_layers, num_models=num_models, classification_nn_hidden_size=5, prop_cells_updated=args.prop_cells_updated, sample_archs=args.sample_archs, device=device)
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    # Define the loss function and optimizer
    #optimizer = optim.SGD(net.local_nn.parameters(), lr=0.01, momentum=0.9)
    #optimizer = optim.SGD(net.local_nn.parameters(), lr=0.001)
    #optimizer = optim.SGD(net.local_nn.parameters(), lr=0.01)
    optimizer = optim.Adam(net.local_nn.parameters(), lr=0.001)
    #optimizer = optim.Adam(net.local_nn.parameters(), lr=0.01)

    # Train the network

    #torch.autograd.set_detect_anomaly(True)
    import time


    start = time.time()
    log_every_n_epochs = 1
    viz_every_n_epochs = 1000
    num_metaepochs = 30000
    num_epochs = args.num_epochs

    def add_gradient_noise(model, noise_stddev=0.01):
        """Add Gaussian noise to the gradients of a PyTorch model."""
        for param in model.parameters():
            if param.grad is not None:
                param.grad += torch.randn_like(param) * noise_stddev

    prev_loss = -1
    jiggled = 0
    for metaepoch in range(num_metaepochs):
    #for metaepoch in range(1000):
        net.reset_models()
        net.zero_grad()
        optimizer.zero_grad()
        loss = 0
        #chosen_epoch = random.randint(round(0.8*num_epochs), num_epochs)
        chosen_epoch = random.randint(round(0.5*num_epochs), num_epochs)
        training = True
        epoch = 0
        #for epoch in range(num_epochs):
        training_weights = [] # list of list of lists: model, layer, timestep
        for i in range(0, num_models):
            num_layers = net.models[i].num_layers
            training_weights.append([])
            for l in range(0, num_layers):
                training_weights[i].append([])
        while training:
            # Each forward call is a "step" of the simulation
            correct = 0
            if metaepoch % viz_every_n_epochs == 0:
                #curr_weights = [w.detach() for w in net.weights]
                #visualize_weights(net.weights, metaepoch, epoch)
                #viz_start = time.time()
                if args.visualize:
                    print('visualize')
                    for i, model in enumerate(net.models):
                        visualize_weights(model.weights, metaepoch, epoch, directory=viz_dir, model_name='model_' + str(i), save_prefix=args.run_name)
                        for l, layer in enumerate(model.weights):
                            training_weights[i][l].append(layer.cpu().detach().numpy())
                    print('done visualize')
                #viz_end = time.time()
                #print('Time for visualization for current step: ' + str(viz_end - viz_start))
            #print('Reparametrize')
            #net.reparametrize_weights()
            #print('Done Reparametrize')
            for (X,y) in train_dataloader:
                #print("NCA forward")
                model_outputs = net(X)
                #print("Done forward")
                for outputs in model_outputs:
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == y).float().sum()
                    if epoch > chosen_epoch: # only apply loss after many iterations of the rule
                        loss += criterion(outputs, y)
                        training = False
            epoch += 1
        # done training
        if args.visualize:
            if metaepoch % viz_every_n_epochs == 0:
                for i, model in enumerate(net.models):
                    for l, layer in enumerate(training_weights[i]):
                        #import ipdb; ipdb.set_trace()
                        curr_model_weight_history = np.concatenate(training_weights[i][l])
                        # Initialize
                        mapper = kmapper.KeplerMapper(verbose=1)

                        # dimensionality reduction
                        #pca = PCA(curr_model_weight_history.shape[1])
                        #pca_feats = pca.fit_transform(curr_model_weight_history)
                        # Fit to and transform the data
                        #projected_data = mapper.fit_transform(pca_feats, projection=[0,1]) # X-Y axis
                        projected_data = mapper.fit_transform(curr_model_weight_history, projection=[0,1]) # X-Y axis

                        # Create a cover with 10 elements
                        cover = kmapper.Cover(n_cubes=10)

                        # Create dictionary called 'graph' with nodes, edges and meta-information
                        graph = mapper.map(projected_data, curr_model_weight_history, cover=cover, clusterer=sklearn.cluster.DBSCAN(metric='l2'))

                        # Visualize it
                        mapper.visualize(graph, path_html="./visualizations/mapper_plots/" + args.run_name + "_model_" + str(i) + "_layer_" + str(l) + "_metaepoch_" + str(metaepoch) + "_weights_mapper_output.html",
                                         title=str(net.models[i]) + '_layer_' + str(l))
            

        #loss.retain_grad()
        #print('Backward')
        print('Stopped on epoch ' + str(epoch))
        loss.backward()
        #if loss == prev_loss:
            #print('Okay, what is going on: loss is still ' + str(loss))
            #jiggled += 1
            #if jiggled > 10:
            #    print('Hmm. didnt work after ' + str(jiggled) +' times, what is going on')
            #    import ipdb; ipdb.set_trace()
            #print('Adding gradient noise to jiggle it out.')
            #add_gradient_noise(net.local_nn, noise_stddev=0.01)
        #print(net.models[0].weights)
        #print(net.models[0].hidden_states)

        #print('Done backward')
        #print('optimizer step')
        layers = list(net.local_nn.modules())[1:]

        #with torch.no_grad():
        #    for layer in layers:
        #        if 'weight' in dir(layer):
        #            layer.weight.grad = layer.weight.grad/(layer.weight.grad.norm() + 1e-8)
        #            #layer.bias.grad = layer.bias.grad/(layer.bias.grad.norm() + 1e-8)

        optimizer.step()
        prev_loss = loss
        #print('Done optimizer step')

        #loss = criterion(outputs, train_dataset.dataset.tensors[1])
        accuracy = correct / (train_size*len(model_outputs))
        if not args.no_log:
            wandb.log({'loss': loss.item(), 'accuracy': accuracy})

        #print(layers[0].weight.grad)
        if metaepoch % log_every_n_epochs == 0:
            print(f"MetaEpoch {metaepoch}: Loss {loss.item()}")
            print(f"Accuracy: {accuracy}")
            net.local_update = False
            correct = 0
            for (X, y) in val_dataloader:
                model_test_outputs = net(X)
                for test_outputs in model_test_outputs:
                    _, predicted = torch.max(test_outputs, 1)
                    correct += (predicted == y).float().sum()
            val_accuracy = correct / (val_size * len(model_test_outputs))
            print(f"Val accuracy: {val_accuracy}")
            if not args.no_log:
                wandb.log({'val_accuracy': val_accuracy})
            net.local_update = True

        if loss.item() != loss.item():
            exit()
    end = time.time()

    if args.visualize:
        for logged_metaepoch in range(int(num_metaepochs/viz_every_n_epochs)):
            metaepoch = logged_metaepoch*viz_every_n_epochs
            for i, model in enumerate(net.models):
                for layer_num in range(model.num_layers):
                    create_weight_gif(layer_num, metaepoch, num_epochs, model_name='model_' + str(i), gif_name=args.run_name + '_model_' + str(i) + '_W_' + str(layer_num) + '_metaepoch_' + str(metaepoch) + '.gif', save_prefix=args.run_name)
        
    with torch.no_grad():
        num_model_samples = 100
        acc_sum = 0
        val_acc_sum = 0
        for model_ind in range(num_model_samples):
            #new_hidden_dim = torch.randint(low=10, high=100, size=(1,))[0]
            #print('new hidden dim: ' + str(new_hidden_dim))
            #net.hidden_dim = new_hidden_dim
            net.reset_models()
            for epoch in range(num_epochs):
                # Each forward call is a "step" of the simulation
                correct = 0
                for (X,y) in train_dataloader:
                    model_outputs = net(X)
                    for outputs in model_outputs:
                        _, predicted = torch.max(outputs, 1)
                        correct += (predicted == y).float().sum()
            
            accuracy = correct / (train_size*len(model_outputs))
            acc_sum += accuracy
            print(f"Accuracy: {accuracy}")
            val_acc_sum += val_accuracy
            net.local_update = False
            correct = 0
            for (X, y) in val_dataloader:
                model_test_outputs = net(X)
                for test_outputs in model_test_outputs:
                    _, predicted = torch.max(test_outputs, 1)
                    correct += (predicted == y).float().sum()
            val_accuracy = correct / (val_size * len(model_test_outputs))
            print(f"Val accuracy: {val_accuracy}")
            net.local_update = True


    print('Avg train accuracy: ' + str(acc_sum/num_model_samples))
    print('Avg Val accuracy: ' + str(val_acc_sum/num_model_samples))
    print('Total time: ' + str(end - start))
    print('Device: ' + device)

# GPU: 21.401570320129395
# CPU: 13.126130819320679
