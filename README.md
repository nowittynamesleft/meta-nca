# meta-nca
A neural cellular automata (NCA) model to evolve neural networks using local rules.

![](https://github.com/nowittynamesleft/meta-nca/blob/multiarchitecture/visualizations/combined_single_layer_all_metaepochs_no_activation.gif)

<em>Evolution of the weights of a single layer network to train on the Iris dataset. No activations are used.</em>

We have the following setup:
We have a feed-forward neural network to be trained for the simple classification task on the iris dataset.
We denote this network as the “task neural network”. Each weight of the
task neural network has a hidden state vector associated with it. We have a
second neural network, termed the ”local rule network”, that is responsible
for updating the weights of the task neural network. This local rule network is trained using the loss of the task neural network.

### Motivation
Neural networks are useful models for many tasks. However, they
are expensive to train on large amounts of data, and once they are
trained, it can be difficult to continue training on new data without
losing performance on old data. In addition, these neural networks
are not robust to adding and removing new weights during training
or after training. Generally these issues do not apply to biological
neural networks – brains are able to learn continuously, are themselves
adding and removing new connections through this learning, and quite
power-efficient in comparison to large scale machine learning models
of today. If we had a learning rule that enabled a neural network to be
trained by exposure to data without needing to calculate a gradient,
and that was only locally dependent on neighboring weights, we would
be able to train neural networks with the following properties:
- No backpropagation required to train a new model, 
allowing for updates to the model to have weight-level parallelism.
- Given that the updates are local, the neural networks could have
the ability to be robust to changes in their connections, enabling new modules of the
network to be added on the fly.

Inspired by works such as https://distill.pub/2020/growing-ca/ . Analogous components and differences:
- Emoji -> task neural network, where pixels are weights and hidden states of pixels are hidden states of the weights.
  - However, in this project we don't explicitly have "dead" and "alive" weights, unlike in the growing NCA work.
  - We also have the local rule network take activations of neighboring neurons, to give a dependence on input data.
- Timesteps: in the case of activation included, timesteps are the individual forward calls of the task neural network. Without activation, the concept is the same as in the emoji case.

This is a work in progress. Currently, any task network beyond 2 layers is hard to optimize.

### Local Rule Network Inputs
In order for the local rule network to be able to output updates for any weight in any layer in the task neural network,
we remove the dependence on any particular number of neighboring weights.
To do this, for a given weight <em>w</em>, we take the sum of weights and hidden
states connected to the same neuron of the next layer relative to <em>w</em> to get
<em>w_forward</em> and <em>h_forward</em>, and the same thing for those weights connected to the
same neuron as w of the previous layer to get <em>w_backward</em> and <em>h_backward</em> . All of
these are concatenated into a single “neighborhood perception” vector, and
this is concatenated with the current weight <em>w</em> and its hidden state <em>h_w</em> .

### Including Activations
We also extend the perception vector to include activations of neurons for updating the network using individual data samples.
This gives the local rule network an actual dependence on the input data beyond optimizing the loss during training. 

### Local Rule Network Outputs: 
The output of the local rule network
is <em>∆w</em> and <em>∆h_w</em>, the change in current weight and the change of hidden state
of the current weight.

### Training the local rule neural network 
The loss function of the task neural
network is computed using a cross-entropy classification loss, and we use this
to update the weights of the local rule neural network by backpropagating
the gradient through the task neural network through to the local rule neural
network. We only update the weights of the local rule network in this way.
The only updates to the task network are given by the local rule network.

### Application of local rule
The application of the local rule neural network is stochastic so that updates are not guarenteed to happen globally
across the entire network. A fraction of task network weights is sampled at
every step to be updated by the local rule network.

### Initialization of hidden states
In order to give some amount of information of global location of weight to the local rule network, and not have
the local rule neural network keep applying the same operation for all neurons, the
hidden states are initialized to a binary encoding of the layer number <em>l</em> 
concatenated with a binary encoding of the arbitrary ordering of the neurons 
of the current layer.
