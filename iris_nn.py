import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from sklearn import datasets
import wandb

wandb.init(project='meta_nca')
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
        x = self.fc3(x)
        return x


class NCA_NN(nn.Module):
    def __init__(self, in_features, out_features, hidden_size, X, Y):
        super(NCA_NN, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_size = hidden_size

        self.weights = nn.Parameter(torch.randn(in_features, out_features), requires_grad=False)
        self.update_rule_network = nn.Sequential(
            nn.Linear(in_features + out_features + X.shape[1] + Y.shape[1], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def local_update(self, X, Y):
        delta_weights = torch.zeros(self.in_features, self.out_features)
        for i in range(self.in_features):
            for j in range(self.out_features):
                neighbors = torch.cat((self.weights[i, :], self.weights[:, j]))
                delta_weights[i, j] = self.update_rule_network(neighbors)

        self.weights = self.weights + delta_weights
        return self.weights

    def forward(self, X):
        return torch.matmul(X, self.weights)


class MyLinear(nn.Module):
    def __init__(self, in_units, out_units, prop_cells_updated=1.0, device=None):
        super().__init__()
        self.device = device
        self.in_units = in_units
        self.out_units = out_units
        self.reset_weight()
        self.prop_cells_updated = prop_cells_updated

        #self.local_nn = nn.Linear(in_units+out_units, 1)
        hidden_size = 10
        self.local_nn = nn.Sequential(
            nn.Linear(in_units+out_units, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        ).to(device)

    def reset_weight(self):
        #import ipdb; ipdb.set_trace()
        self.weight = nn.Parameter(torch.zeros(self.in_units, self.out_units), requires_grad=False).to(self.device)
        #self.bias = nn.Parameter(torch.zeros(self.out_units,), requires_grad=False)

    def nca_local_rule(self, idx_in_units, idx_out_units):
        for i in idx_in_units:
            for j in idx_out_units:
                self.weight[i, j] += self.local_nn(torch.cat([self.weight[i, :].flatten(), self.weight[:, j].flatten()]))
        #self.weight.data = F.normalize(self.weight)

    def forward(self, X):
        #idx_in_units = torch.randint(0, self.in_units, (int(self.prop_cells_updated*self.in_units), 1))
        #idx_out_units = torch.randint(0, self.out_units, (int(self.prop_cells_updated*self.out_units), 1))
        # Random sample without replacement
        idx_in_units = torch.randperm(self.in_units)[:int(self.in_units*self.prop_cells_updated)][:,None]
        idx_out_units = torch.randperm(self.out_units)[:int(self.out_units*self.prop_cells_updated)][:,None]
        self.nca_local_rule(idx_in_units, idx_out_units)
        #linear = torch.matmul(X, self.weight) + self.bias
        linear = torch.matmul(X, self.weight) # no bias for now
        return F.softmax(linear, dim=-1)


# Initialize the network
#net = Net()
net = MyLinear(X.shape[1], y.max() + 1, device=device)
net = net.to(device)
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.local_nn.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.SGD(net.local_nn.parameters(), lr=0.001)

# Train the network

#prev_local_nn = torch.zeros_like(net.local_nn.weight)
# TODO: why can I not make gradient updates every epoch? Can't do it more than twice

#torch.autograd.set_detect_anomaly(True)
for metaepoch in range(10000):
    net.reset_weight()
    loss = 0
    optimizer.zero_grad()
    for epoch in range(10):
        # Each forward call is a "step" of the simulation
        outputs = net(train_dataset.dataset.tensors[0])

    loss = criterion(outputs, train_dataset.dataset.tensors[1])
    loss.backward()
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == train_dataset.dataset.tensors[1]).float().mean()
    wandb.log({'loss': loss.item(), 'accuracy': accuracy})

    layers = list(net.local_nn.modules())[1:]
    with torch.no_grad():
        for layer in layers:
            if 'weight' in dir(layer):
                layer.weight.grad = layer.weight.grad/(layer.weight.grad.norm() + 1e-8)
                layer.bias.grad = layer.bias.grad/(layer.bias.grad.norm() + 1e-8)
    #print(layers[0].weight.grad)
    optimizer.step()
    if metaepoch % 100 == 0:
        print(net.weight)
        print(f"MetaEpoch {metaepoch}: Loss {loss.item()}")
        print(f"Accuracy: {accuracy}")
    if loss.item() != loss.item():
        exit()
