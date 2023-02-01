import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from sklearn import datasets

# Load the Iris dataset
iris = datasets.load_iris()
X = iris["data"]
y = iris["target"]

# Convert to PyTorch tensors and create a dataset
X = torch.Tensor(X)
y = torch.Tensor(y).long()
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
    def __init__(self, in_units, out_units, perc=0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(in_units, out_units), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_units,), requires_grad=False)
        self.in_units = in_units
        self.out_units = out_units
        self.perc = perc

        #self.local_nn = nn.Linear(in_units+out_units, 1)
        hidden_size = 100
        self.local_nn = nn.Sequential(
            nn.Linear(in_units+out_units, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def reset_weight(self):
        #import ipdb; ipdb.set_trace()
        self.weight = nn.Parameter(torch.zeros(self.in_units, self.out_units), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(self.out_units,), requires_grad=False)

    def nca_local_rule(self, idx_in_units, idx_out_units):
        for i in idx_in_units:
            for j in idx_out_units:
                self.weight[i, j] = torch.sigmoid(self.local_nn(torch.cat([self.weight[i, :].flatten(), self.weight[:,j].flatten()]))) - 0.5

    def forward(self, X):
        idx_in_units = torch.randint(0, self.in_units, (int(self.perc*self.in_units), 1))
        idx_out_units = torch.randint(0, self.out_units, (int(self.perc*self.out_units), 1))
        self.nca_local_rule(idx_in_units, idx_out_units)
        linear = torch.matmul(X, self.weight) + self.bias
        return F.softmax(linear, dim=-1)


# Initialize the network
#net = Net()
net = MyLinear(X.shape[1], y.max() + 1)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.local_nn.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.SGD(net.local_nn.parameters(), lr=0.001)
#torch.autograd.set_detect_anomaly(True)

# Train the network
'''
prev_local_nn = torch.zeros_like(net.local_nn.weight)
for metaepoch in range(100):
    net.reset_weight()
    # Zero the gradients
    loss = 0
    for epoch in range(1000):

        # Forward pass
        #import ipdb; ipdb.set_trace()
        outputs = net(train_dataset.dataset.tensors[0])
        #outputs.requires_grad = True
        loss += criterion(outputs, train_dataset.dataset.tensors[1])

    # Print the training loss
    print(f"Epoch {epoch}: Loss {loss.item()}")
    #print(net.weight)
    #if torch.any(diff != 0):
    #   import ipdb; ipdb.set_trace()
    #import ipdb; ipdb.set_trace()
    #loss.backward(retain_graph=True)
    # Backward pass and optimization
    #loss.backward()
    loss.backward()
    optimizer.step()
    diff = prev_local_nn - net.local_nn.weight
    print(diff)
    prev_local_nn = net.local_nn.weight
'''

#prev_local_nn = torch.zeros_like(net.local_nn.weight)
# TODO: why can I not make gradient updates every epoch? Can't do it more than twice
for metaepoch in range(1000):
    net.reset_weight()
    loss = 0
    for epoch in range(10):
        # Zero the gradients

        # Forward pass
        #import ipdb; ipdb.set_trace()
        outputs = net(train_dataset.dataset.tensors[0])
        #outputs.requires_grad = True
        loss += criterion(outputs, train_dataset.dataset.tensors[1])
        '''
        loss.backward()
        optimizer.step()
        outputs.detach()
        net.weight.detach()
        net.bias.detach()
        net.local_nn.weight.detach()
        '''

        # Backward pass and optimization

        # Print the training loss
    print(f"MetaEpoch {metaepoch}: Loss {loss.item()}")
    #print(net.weight)
    if loss.item() != loss.item():
        exit()
    #import ipdb; ipdb.set_trace()
    #outputs.detach()
    #loss.backward(retain_graph=True)
    loss.backward()
    #prev_local_nn = net.local_nn.weight.clone().detach()
    optimizer.step()
    #diff = prev_local_nn - net.local_nn.weight
    #print(diff)
    #print(net.local_nn.weight)

# Evaluate the network on the validation set
with torch.no_grad():
    outputs = net(val_dataset.dataset.tensors[0])
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == val_dataset.dataset.tensors[1]).float().mean()

print(f"Validation Accuracy: {accuracy}")
