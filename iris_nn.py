import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from sklearn import datasets
import wandb
import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('run_name')

args = parser.parse_args()

wandb.init(project='meta_nca')
wandb.run.name = args.run_name
# Load the Iris dataset

#device = 'cuda:0'
device = 'cpu' # gpu is slower...not sure why
iris = datasets.load_iris()
X = iris["data"]
y = iris["target"]

# Convert to PyTorch tensors and create a dataset
X = torch.Tensor(X).to(device)
y = torch.Tensor(y).long().to(device)
dataset = data.TensorDataset(X, y)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size])
train_dataloader = data.DataLoader(train_dataset, batch_size=len(train_dataset))
val_dataloader = data.DataLoader(val_dataset, batch_size=len(val_dataset))

viz_dir = 'visualizations/'

import imageio
import os

def create_weight_gif(layer_num, metaepoch, num_epochs, gif_name='weights.gif', directory='visualizations/'):
    images = []
    for epoch in range(num_epochs):
        # Read each weight matrix as a jpg file
        images.append(imageio.imread(f'{directory}W_{layer_num}_metaepoch_{metaepoch}_epoch_{epoch}.jpg'))
    
    # Create the gif using the weight visualizations
    imageio.mimsave(gif_name, images, fps=5)
    

def visualize_weights(weights, metaepoch, epoch, directory='visualizations/'):
    curr_weights = [w.detach() for w in weights]
    for i, W in enumerate(curr_weights):
        plt.imshow(W, cmap='gray')
        plt.axis('off')
        plt.savefig(f'{directory}W_{i}_metaepoch_{metaepoch}_epoch_{epoch}.jpg')
        plt.close()

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


class MetaNCA(nn.Module):
    def __init__(self, in_units, out_units, num_layers=1, prop_cells_updated=1.0, device=None):
        super().__init__()
        self.device = device
        self.in_units = in_units
        self.out_units = out_units
        self.num_layers = num_layers
        self.prop_cells_updated = prop_cells_updated
        self.local_update = True

        #self.local_nn = nn.Linear(in_units+out_units, 1)
        self.classification_nn_hidden_size = 5
        local_nn_hidden_size = 10
        self.reset_weight()
        self.hidden_state_dim = self.hidden_states[0].shape[-1]
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
            nn.Linear(local_nn_hidden_size, 1 + self.hidden_state_dim)
        ).to(device)

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
                w = torch.zeros(self.in_units, self.out_units).to(self.device)
            elif l == 0:
                w = torch.zeros(self.in_units, self.classification_nn_hidden_size).to(self.device)
            elif l == self.num_layers - 1:
                w = torch.zeros(self.classification_nn_hidden_size, self.out_units).to(self.device)
            else:
                w = torch.zeros(self.classification_nn_hidden_size, self.classification_nn_hidden_size).to(self.device)
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

    def nca_local_rule(self, param, hidden_states):
        #updates = torch.zeros_like(param).to(self.device)
        torch.autograd.set_detect_anomaly(True)
        updates = torch.zeros(param.shape[0], param.shape[1], 1+hidden_states.shape[2]).to(self.device)
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

    def encode_weight_hidden_states(self, weights):
        # binary encoding
        num_layers = len(weights)
        max_num_layer_weights = 0
        for weight in weights:
            curr_num_weights = weight.shape[0]*weight.shape[1]
            if curr_num_weights > max_num_layer_weights:
                max_num_layer_weights = curr_num_weights
        #import ipdb; ipdb.set_trace()
        #layer_encoding_dim = int(torch.ceil(torch.log(torch.tensor(num_layers))/torch.log(torch.tensor(2))).item())
        layer_encoding_dim = int(np.ceil(np.log(num_layers)/np.log(2)))

        #weight_encoding_dim = int(torch.ceil(torch.log(torch.tensor(max_num_layer_weights))/torch.log(torch.tensor(2))).item())
        weight_encoding_dim = int(np.ceil(np.log(max_num_layer_weights)/np.log(2)))
        total_encoding_dim = layer_encoding_dim + weight_encoding_dim
        hidden_states = []
        for layer_num, weight_mat in enumerate(weights):
            encodings = torch.zeros(weight.shape[0], weight.shape[1], total_encoding_dim)
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
        return torch.tensor(np.fromstring(','.join(list(binary)), sep=',', dtype=int))

    def forward(self, X):
        # Random sample without replacement
        if self.local_update:
            all_weight_updates = []
            all_hidden_state_updates = []
            for (weight, hidden_state) in zip(self.weights, self.hidden_states):
                weight_updates, hidden_state_updates = self.nca_local_rule(weight, hidden_state)
                all_weight_updates.append(weight_updates)
                all_hidden_state_updates.append(hidden_state_updates)
            # only update after all updates have been calculated
            for layer, (weight, hidden_state) in enumerate(zip(self.weights, self.hidden_states)):
                weight.copy_(weight + all_weight_updates[layer])
                hidden_state += all_hidden_state_updates[layer]
                #new_weight = weight.clone()
                #new_weight += all_weight_updates[layer]
                #weight.copy_(new_weight)
                #weight.data += all_weight_updates[layer]
        prev_layer = X
        for i in range(self.num_layers):
            linear = torch.matmul(prev_layer, self.weights[i]) # no bias for now
            prev_layer = F.relu(linear)
        return F.softmax(prev_layer, dim=-1)


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
num_layers = 1
#net = MetaNCA(X.shape[1], y.max() + 1, num_layers=num_layers, hidden_state_dim=5*5, prop_cells_updated=0.5, device=device)
net = MetaNCA(X.shape[1], y.max() + 1, num_layers=num_layers, prop_cells_updated=1.0, device=device)
net = net.to(device)
criterion = nn.CrossEntropyLoss()
# Define the loss function and optimizer
#optimizer = optim.SGD(net.local_nn.parameters(), lr=0.01, momentum=0.9)
#optimizer = optim.SGD(net.local_nn.parameters(), lr=0.001)
#optimizer = optim.SGD(net.local_nn.parameters(), lr=0.01)
optimizer = optim.Adam(net.local_nn.parameters(), lr=0.001)

# Train the network

#torch.autograd.set_detect_anomaly(True)
import time

start = time.time()
log_every_n_epochs = 100
num_metaepochs = 500
num_epochs = 5

for metaepoch in range(num_metaepochs):
#for metaepoch in range(1000):
    net.reset_weight()
    net.zero_grad()
    optimizer.zero_grad()
    loss = 0
    for epoch in range(num_epochs):
        # Each forward call is a "step" of the simulation
        correct = 0
        if metaepoch % log_every_n_epochs == 0:
            #curr_weights = [w.detach() for w in net.weights]
            #visualize_weights(net.weights, metaepoch, epoch)
            visualize_weights(net.weights, metaepoch, epoch, directory=viz_dir)
        for (X,y) in train_dataloader:
            outputs = net(X)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).float().sum()
            loss += criterion(outputs, y)
    loss.backward()
    optimizer.step()

    #loss = criterion(outputs, train_dataset.dataset.tensors[1])
    accuracy = correct / train_size
    wandb.log({'loss': loss.item(), 'accuracy': accuracy})

    layers = list(net.local_nn.modules())[1:]
    with torch.no_grad():
        for layer in layers:
            if 'weight' in dir(layer):
                layer.weight.grad = layer.weight.grad/(layer.weight.grad.norm() + 1e-8)
                #layer.bias.grad = layer.bias.grad/(layer.bias.grad.norm() + 1e-8)
    #print(layers[0].weight.grad)
    if metaepoch % log_every_n_epochs == 0:
        print(f"MetaEpoch {metaepoch}: Loss {loss.item()}")
        print(f"Accuracy: {accuracy}")
        net.local_update = False
        correct = 0
        for (X, y) in val_dataloader:
            test_outputs = net(X)
            _, predicted = torch.max(test_outputs, 1)
            correct += (predicted == y).float().sum()
        val_accuracy = correct / val_size
        print(f"Val accuracy: {val_accuracy}")
        wandb.log({'val_accuracy': val_accuracy})
        net.local_update = True

    if loss.item() != loss.item():
        exit()
end = time.time()

for logged_metaepoch in range(int(num_metaepochs/log_every_n_epochs)):
    metaepoch = logged_metaepoch*log_every_n_epochs
    for layer_num in range(num_layers):
        create_weight_gif(layer_num, metaepoch, num_epochs, gif_name='W_' + str(layer_num) + '_metaepoch_' + str(metaepoch) + '.gif')
    
with torch.no_grad():
    num_model_samples = 100
    acc_sum = 0
    val_acc_sum = 0
    for model_ind in range(num_model_samples):
        new_hidden_dim = torch.randint(low=10, high=100, size=(1,))[0]
        print('new hidden dim: ' + str(new_hidden_dim))
        net.hidden_dim = new_hidden_dim
        net.reset_weight()
        for epoch in range(num_epochs):
            # Each forward call is a "step" of the simulation
            correct = 0
            for (X,y) in train_dataloader:
                outputs = net(X)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y).float().sum()
        
        accuracy = correct / train_size
        acc_sum += accuracy
        print(f"Accuracy: {accuracy}")
        val_acc_sum += val_accuracy
        net.local_update = False
        correct = 0
        for (X, y) in val_dataloader:
            test_outputs = net(X)
            _, predicted = torch.max(test_outputs, 1)
            correct += (predicted == y).float().sum()
        val_accuracy = correct / val_size
        print(f"Val accuracy: {val_accuracy}")
        net.local_update = True


print('Avg train accuracy: ' + str(acc_sum/num_model_samples))
print('Avg Val accuracy: ' + str(val_acc_sum/num_model_samples))
print('Total time: ' + str(end - start))
print('Device: ' + device)

# GPU: 21.401570320129395
# CPU: 13.126130819320679
