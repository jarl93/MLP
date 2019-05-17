import numpy as np
from sklearn import datasets
TRACING = False
####################################

class ReLULayer(object):
    def forward(self, input):
        # remember the input for later backpropagation
        self.input = input
        if (TRACING):
            print ("ReLU forward input shape: "+str(self.input.shape)+"\n")
        zeros = np.zeros_like(self.input)
        # ReLU = max(0, X)
        relu = np.maximum(zeros, self.input)
        # return the ReLU of the input
        return relu

    def backward(self, upstream_gradient):
        # compute the derivative of ReLU from upstream_gradient and the stored input
        # Derivative of ReLU is the step function
        # np.heaviside => step function 
        # np.multiply => element-wise multiplication
        if (TRACING):
            print ("ReLU backward input shape: "+str(self.input.shape))
            print ("ReLU upstream_gradient shape: "+ str(upstream_gradient.shape))
        downstream_gradient = np.multiply(upstream_gradient, np.heaviside(self.input, 0))
        if (TRACING):
            print ("ReLU downstream_gradient shape: "+ str(downstream_gradient.shape))
        return downstream_gradient

    def update(self, learning_rate):
        pass # ReLU is parameter-free

####################################

class OutputLayer(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def forward(self, input):
        # remember the input for later backpropagation
        self.input = input
        if (TRACING):
            print ("Output forward input shape: "+str(self.input.shape))
        # Apply the exp function to each entry
        exp = np.exp(self.input)
        # Make the sum by row
        sum_row = np.sum(exp, axis = 1)
        # Divide by each entry by the corresponding sum
        softmax = exp/sum_row[:, np.newaxis]
        if (TRACING):
            print ("Softmax forward shape: "+str(softmax.shape)+str("\n"))
        # return the softmax of the input
        return softmax

    def backward(self, predicted_posteriors, true_labels):
        # return the loss derivative with respect to the stored inputs
        # (use cross-entropy loss and the chain rule for softmax,
        #  as derived in the lecture)
        if (TRACING):
            print ("Output backward input shape: "+str(self.input.shape))
            print ("Output predicted_posteriors shape: "+str(predicted_posteriors.shape))
        downstream_gradient = predicted_posteriors
        idx = [np.arange(true_labels.size), true_labels]
        downstream_gradient[idx] -= 1
        if (TRACING):
            print ("Output downstream_gradient shape: "+ str(downstream_gradient.shape))
        return downstream_gradient

    def update(self, learning_rate):
        pass # softmax is parameter-free

####################################

class LinearLayer(object):
    def __init__(self, n_inputs, n_outputs):
        self.n_inputs  = n_inputs
        self.n_outputs = n_outputs
        # randomly initialize weights and intercepts
        mu = 0; sigma= 1; n_samples = n_inputs * n_outputs
        rand = np.random.normal(mu, sigma, n_samples)
        self.B = rand.reshape((n_outputs, n_inputs))
        self.b = np.random.normal(mu, sigma, n_outputs)
        
    def forward(self, input):
        # remember the input for later backpropagation
        self.input = input
        if (TRACING):
            print ("Linear forward input shape: "+str(self.input.shape))
        # compute the scalar product of input and weights
        # (these are the preactivations for the subsequent non-linear layer)
        #print ("self.B shape: "+str(self.B.shape))
        
        preactivations = (self.B @ self.input.T).T + self.b[np.newaxis,:]
        if (TRACING):
            print ("Preactivations shape: "+str(preactivations.shape))
        
        return preactivations

    def backward(self, upstream_gradient):
        # compute the derivative of the weights from
        # upstream_gradient and the stored input
        
        # Derivative w.r.t bias is always
        # the previous upstream gradient,
        # get the mean because is a mini-batch
        self.grad_b = np.mean(upstream_gradient, axis=0)
        # Compute autoproduct between the upstream_gradient
        # and the inputs (outputs of the previous layer)
        # to get the grad_B
        # Perhaps it's possible to avoid the loop
        if (TRACING):
            print ("Linear upstream_gradient shape: "+str(upstream_gradient.shape))
            print ("Linear backward input shape: "+str(self.input.shape))
        
        for i in range (self.input.shape[0]):
            if i==0:
                self.grad_B = np.outer(upstream_gradient[i], self.input[i])
            else:
                self.grad_B += np.outer(upstream_gradient[i], self.input[i])
        self.grad_B /= self.input.shape[0]
        
        if (TRACING):
            print ("Linear grad_B shape: "+str(self.grad_B.shape))
            print ("Linear grad_b shape: "+str(self.grad_b.shape))
        
        # compute the downstream gradient to be passed to the preceding layer
        downstream_gradient = upstream_gradient @ (self.grad_B + self.grad_b[:, np.newaxis])
        if (TRACING):
            print ("Linear downstream_gradient shape: "+str(downstream_gradient.shape))
        

        
        return downstream_gradient

    def update(self, learning_rate):
        # update the weights by batch gradient descent
        self.B = self.B - learning_rate * self.grad_B
        self.b = self.b - learning_rate * self.grad_b

####################################

class MLP(object):
    def __init__(self, n_features, layer_sizes):
        # constuct a multi-layer perceptron
        # with ReLU activation in the hidden layers and softmax output
        # (i.e. it predicts the posterior probability of a classification problem)
        #
        # n_features: number of inputs
        # len(layer_size): number of layers
        # layer_size[k]: number of neurons in layer k
        # (specifically: layer_sizes[-1] is the number of classes)
        self.n_layers = len(layer_sizes)
        self.layers   = []

        # create interior layers (linear + ReLU)
        n_in = n_features
        for n_out in layer_sizes[:-1]:
            self.layers.append(LinearLayer(n_in, n_out))
            self.layers.append(ReLULayer())
            n_in = n_out

        # create last linear layer + output layer
        n_out = layer_sizes[-1]
        self.layers.append(LinearLayer(n_in, n_out))
        self.layers.append(OutputLayer(n_out))

    def forward(self, X):
        # X is a mini-batch of instances
        batch_size = X.shape[0]
        # flatten the other dimensions of X (in case instances are images)
        X = X.reshape(batch_size, -1)

        # compute the forward pass
        # (implicitly stores internal activations for later backpropagation)
        if (TRACING):
            print ("FORWARD")
        result = X
        for layer in self.layers:
            result = layer.forward(result)
        if (TRACING):
            print ("############################")
        return result

    def backward(self, predicted_posteriors, true_classes):
        # perform backpropagation w.r.t. the prediction for the latest mini-batch X
        # Do the first backward step on the last layer
        if (TRACING):
            print ("BACKWARD")
        result = self.layers[-1].backward(predicted_posteriors, true_classes)
        if (TRACING):
            print ("Result shape: "+str(result.shape)+"\n")
        # Go backwards for the rest of the layers
        for layer in reversed(self.layers):
            if isinstance(layer, OutputLayer):
                continue
            result = layer.backward(result)
            if (TRACING):
                print ("Result shape: "+str(result.shape)+"\n")
        if (TRACING):
            print ("############################")
        return result

    def update(self, X, Y, learning_rate):
        posteriors = self.forward(X)
        self.backward(posteriors, Y)
        for layer in self.layers:
            layer.update(learning_rate)

    def train(self, x, y, n_epochs, batch_size, learning_rate):
        N = len(x)
        n_batches = N // batch_size
        for i in range(n_epochs):
            # print("Epoch", i)
            # reorder data for every epoch
            # (i.e. sample mini-batches without replacement)
            permutation = np.random.permutation(N)

            for batch in range(n_batches):
                # create mini-batch
                start = batch * batch_size
                x_batch = x[permutation[start:start+batch_size]]
                y_batch = y[permutation[start:start+batch_size]]

                # perform one forward and backward pass and update network parameters
                self.update(x_batch, y_batch, learning_rate)

##################################
if __name__=="__main__":

    # set training/test set size
    N = 2000

    # create training and test data
    X_train, Y_train = datasets.make_moons(N, noise=0.05)
    X_test,  Y_test  = datasets.make_moons(N, noise=0.05)
    n_features = 2
    n_classes  = 2

    # standardize features to be in [-1, 1]
    offset  = X_train.min(axis=0)
    scaling = X_train.max(axis=0) - offset
    X_train = ((X_train - offset) / scaling - 0.5) * 2.0
    X_test  = ((X_test  - offset) / scaling - 0.5) * 2.0

    # set hyperparameters (play with these!)
    layer_sizes = [5, 5, n_classes]
    n_epochs = 5
    batch_size = 200
    learning_rate = 0.05

    # create network
    network = MLP(n_features, layer_sizes)

    # train
    network.train(X_train, Y_train, n_epochs, batch_size, learning_rate)

    # test
    predicted_posteriors = network.forward(X_test)
    # determine class predictions from posteriors by winner-takes-all rule
    predicted_classes = predicted_posteriors.argmax(1)
    # compute and output the error rate of predicted_classes
    error_rate = (np.sum(predicted_classes != Y_test)/X_test.shape[0]) * 100
    print("error rate: {}".format(error_rate))
