import numpy as np

class Perceptron:
    """This is a simple implementation of a Perceptron but with a few tricks up it's sleeve.
    How the Perceptron works, takes input X(1000 examples, 5 features), initializes weight matrix
    according to feature size, you can choose the weight initialization method
    choose from random, zeroes or xavier.

    """
    def __init__ (self, X, Y, learning_rate=0.1, weight_init_method='random', activation_method='binary'):
        self.learning_rate = learning_rate
        self.X = X
        self.Y = Y
        # stores method choice
        self.activation_method = activation_method
        # stores method choice
        self.weight_init_method = weight_init_method
        # call the chosen weight initialization method
        self.weights = self._initialize_weights(X.shape[1])
        
    # This is the weight initialization selector
    def _initialize_weights(self, num_features):

        if self.weight_init_method == 'zeroes':
            # calls the return zero weights function
            return self._init_zeros(num_features + 1)
        
        elif self.weight_init_method == 'random':
            # calls the return random weights function
            return self._init_random(num_features + 1)
        
        elif self.weight_init_method == 'xavier':
            # calls the return xavier weights function
            return self._init_xavier(num_features + 1)

    def _init_zeros(self, num_features):
        # simple numpy function that returns an array of zeros
        return np.zeros(num_features)
    
    def _init_random(self, num_features):
        # simple numpy function that returns an array of random nums
        return np.random.randn(num_features)
    
    def _init_xavier(self, num_features):
        limit = np.sqrt(6.0 / num_features)
        return np.random.uniform(-limit, limit, num_features)
    
    def _add_bias(self, X):
        # creates a matrix of 1's of the same shape as input X
        bias_column = np.ones((X.shape[0], 1))
        # concats that matrix of 1's at the end of each X feature
        return np.concatenate([X, bias_column], axis=1)
    
    # This is the activation function selector
    def _activation_function(self, z):

        if self.activation_method == 'binary':
            # selects binary activation (1 or 0)
            return self._activation_binary(z)
        
        elif self.activation_method == 'bipolar':
            # selects bipolar activation (1 or -1)
            return self._activation_bipolar(z)
        
    def _activation_binary(self, z):
        return np.where(z >= 0, 1, 0)
    
    def _activation_bipolar(self, z):
        return np.where(z >= 0, 1, -1)
    
    def predict(self, X):
        Xb = self._add_bias(X)

        z = Xb @ self.weights
        return self._activation_function(z)
    
    def _update_weights(self, x_with_bias, y_true, y_pred):
        if y_true == y_pred:
            return
        if self.activation_method == 'bipolar':
            self.weights += self.learning_rate * y_true * x_with_bias
        else:
            error = y_true - y_pred
            self.weights += self.learning_rate * error * x_with_bias
    
    def train(self, epochs=10, shuffle=True, verbose=False, early_stop=True):
        """
        Unified training loop.
        For the SIMPLE version (no shuffle, no early stop, no prints):
            call: train(epochs=E, shuffle=False, verbose=False, early_stop=False)
        """
        Xb = self._add_bias(self.X)
        Y = self.Y
        n = len(Y)
        self.error_history = []

        for ep in range(epochs):
            if shuffle:
                idx = np.random.permutation(n)
                Xb_epoch = Xb[idx]
                Y_epoch = Y[idx]
            else:
                Xb_epoch = Xb
                Y_epoch = Y

            errors = 0
            for x_row, y_true in zip(Xb_epoch, Y_epoch):
                z = x_row @ self.weights
                y_pred = self._activation_function(z)
                if y_pred != y_true:
                    errors += 1
                self._update_weights(x_row, y_true, y_pred)

            self.error_history.append(errors)

            if verbose:
                print(f"Epoch {ep+1}/{epochs} misclassified={errors}")

            if early_stop and errors == 0:
                if verbose:
                    print("Early stop (no errors).")
                break
        return self

    def accuracy(self, X=None, Y=None):
        if X is None:
            X = self.X
        if Y is None:
            Y = self.Y
        preds = self.predict(X)
        return np.mean(preds == Y)

    def evaluate(self, X, Y):
        preds = self.predict(X)
        mis = np.sum(preds != Y)
        acc = 1.0 - mis / len(Y)
        return acc, mis