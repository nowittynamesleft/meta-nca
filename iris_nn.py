import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from sklearn import datasets
import wandb
import argparse

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
'''
split = int(0.8 * len(X))
train_X = X[:split, :]
train_y = y[:split]
val_X = X[split:, :]
val_y = y[split:]
'''

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
    def __init__(self, in_units, out_units, hidden_state_dim=None, prop_cells_updated=1.0, device=None):
        super().__init__()
        self.device = device
        self.in_units = in_units
        self.out_units = out_units
        self.hidden_state_dim = hidden_state_dim
        self.reset_weight()
        self.prop_cells_updated = prop_cells_updated
        self.local_update = True

        #self.local_nn = nn.Linear(in_units+out_units, 1)
        hidden_size = 10
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
            #nn.Linear(3, hidden_size), # self, sum of forward connected weights, sum of backward connected weights
            nn.Linear(3 + 3*hidden_state_dim, hidden_size), # self, sum of forward connected weights, sum of backward connected weights, self hidden state, forward hidden state sum, backward hidden state sum
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            #nn.Linear(hidden_size, 1)
            nn.Linear(hidden_size, 1 + hidden_state_dim)
        ).to(device)

    def reset_weight(self):
        #import ipdb; ipdb.set_trace()
        w = torch.zeros(self.in_units, self.out_units).to(self.device)
        #w[:, 0] = 1.0
        #b = torch.zeros(self.out_units,)
        #b[0] = 1.0
        #w = torch.randn(self.in_units, self.out_units)
        self.weight = nn.Parameter(w, requires_grad=False).to(self.device)
        #self.bias = nn.Parameter(b, requires_grad=False).to(self.device)
        if self.hidden_state_dim is not None:
            self.hidden_state = torch.zeros(self.in_units, self.out_units, self.hidden_state_dim)
            k = 0
            for i in range(self.in_units):
                for j in range(self.out_units):
                    self.hidden_state[i,j,k] = 1.0 # globally distinct states for intialization
                    k += 1

    def nca_local_rule(self, idx_in_units, idx_out_units):
        #updates = torch.zeros_like(self.weight).to(self.device)
        updates = torch.zeros(self.weight.shape[0], self.weight.shape[1], 1+self.hidden_state_dim).to(self.device)
        #import ipdb; ipdb.set_trace()
        for i in idx_in_units:
            for j in idx_out_units:
                #updates[i,j] = self.local_nn(torch.cat([self.weight[i, :].flatten(), self.weight[:, j].flatten(), self.bias[j]]))
                '''
                updates[i,j] = self.local_nn(torch.cat([self.weight[i, :].flatten(), self.weight[:, j].flatten()]))
                '''
                if i == 0:
                    forward = self.weight[i+1:, j].flatten()
                    forward_states = self.hidden_state[i+1:,j,:]
                elif i == self.weight.shape[0]:
                    forward = self.weight[:i, j].flatten()
                    forward_states = self.hidden_state[:i,j,:]
                else:
                    forward = torch.cat((self.weight[:i,j].flatten(), self.weight[i+1:, j].flatten()))
                    forward_states = torch.cat((self.hidden_state[:i,j], self.hidden_state[i+1:, j]))
                if j == 0:
                    backward = self.weight[i, j+1:].flatten()
                    backward_states = self.hidden_state[i,j+1:,:]
                elif j == self.weight.shape[1]:
                    backward = self.weight[i, :j].flatten()
                    backward_states = self.hidden_state[i,:j,:]
                else:
                    backward = torch.cat((self.weight[i,:j].flatten(), self.weight[i, j+1:].flatten()))
                    backward_states = torch.cat((self.hidden_state[i,:j], self.hidden_state[i, j+1:]), dim=1)
                concatted_weights = torch.cat((self.weight[i,j], torch.mean(forward)[None], torch.mean(backward)[None]))
                #import ipdb; ipdb.set_trace()
                concatted_states = torch.cat((self.hidden_state[i,j, :].flatten(), torch.mean(forward_states, dim=0).flatten(), torch.mean(backward_states, dim=1).flatten()))
                concatted = torch.cat((concatted_weights, concatted_states))
                updates[i,j, :] = self.local_nn(concatted)
        weight_updates = updates[:,:,0] 
        hidden_state_updates = updates[:,:,1:]
        self.weight.copy_(self.weight + weight_updates)
        self.hidden_state += hidden_state_updates
        #for j in idx_out_units:
        #    self.bias[j] += self.local_nn(torch.cat([torch.zeros((self.out_units,)), self.weight[:,j].flatten(), self.bias[j]]))

    def forward(self, X):
        # Random sample without replacement
        idx_in_units = torch.randperm(self.in_units)[:int(self.in_units*self.prop_cells_updated)][:,None]
        idx_out_units = torch.randperm(self.out_units)[:int(self.out_units*self.prop_cells_updated)][:,None]
        if self.local_update:
            self.nca_local_rule(idx_in_units, idx_out_units)
        #linear = torch.matmul(X, self.weight) + self.bias
        linear = torch.matmul(X, self.weight) # no bias for now
        return F.softmax(linear, dim=-1)


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
net = MetaNCA(X.shape[1], y.max() + 1, hidden_state_dim=3*5, prop_cells_updated=1.0, device=device)
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
print(train_dataset)
print(val_dataset)
for metaepoch in range(1000):
#for metaepoch in range(1000):
    net.reset_weight()
    loss = 0
    optimizer.zero_grad()
    for epoch in range(10):
        # Each forward call is a "step" of the simulation
        correct = 0
        for (X,y) in train_dataloader:
            outputs = net(X)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).float().sum()
            loss += criterion(outputs, y)

    #loss = criterion(outputs, train_dataset.dataset.tensors[1])
    loss.backward()
    accuracy = correct / train_size
    wandb.log({'loss': loss.item(), 'accuracy': accuracy})

    layers = list(net.local_nn.modules())[1:]
    with torch.no_grad():
        for layer in layers:
            if 'weight' in dir(layer):
                layer.weight.grad = layer.weight.grad/(layer.weight.grad.norm() + 1e-8)
                #layer.bias.grad = layer.bias.grad/(layer.bias.grad.norm() + 1e-8)
    #print(layers[0].weight.grad)
    optimizer.step()
    if metaepoch % 100 == 0:
        print(net.weight)
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

with torch.no_grad():
    num_model_samples = 100
    acc_sum = 0
    val_acc_sum = 0
    for model_ind in range(num_model_samples):
        net.reset_weight()
        for epoch in range(10):
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
        print(net.weight)

print('Avg train accuracy: ' + str(acc_sum/num_model_samples))
print('Avg Val accuracy: ' + str(val_acc_sum/num_model_samples))
print('Total time: ' + str(end - start))
print('Device: ' + device)
# GPU: 21.401570320129395
# CPU: 13.126130819320679
