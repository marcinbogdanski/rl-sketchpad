import numpy as np
import matplotlib.pyplot as plt


class ANNFuncApprox():
    """Minimal ANN implementation
    
    Layers:
        * 1x hidden sigmoid layer
        * 1x linear output layer
    """
    def __init__(self, learn_rate, x_min, x_max, nb_in, nb_hid, nb_out):
        """
        Params:
            learn_rate - learn rate for backprop
            x_min      - minimum values for inputs - for scaling
            x_max      - maximum values for inputs - for scaling
            nb_in      - number of inputs
            nb_hid     - number of hidden neurons
            nb_outt    - number of outputs
        """
        self._lr = learn_rate
        self._x_min = x_min
        self._x_max = x_max
        
        self._Wh = np.random.normal(loc=0, scale=1/nb_in**-.5, size=[nb_in, nb_hid])   # Xavier
        self._bh = np.zeros(shape=[1, nb_hid])
        self._Wo = np.random.normal(loc=0, scale=1/nb_hid**-.5, size=[nb_hid, nb_out])
        self._bo = np.zeros(shape=[1, nb_out])
        
    def eval(self, x):
        """Forward pass on neural network
        
        Params:
            x - input, either scalar or 2d np.ndarray with dims: [batch_size, num_inputs]
        """
        if np.isscalar(x):   x = np.array([[x]])             # needed so @ operator works ok
        assert x.ndim == 2                                   # tested for 2d numpy arrays only
        
        x = (x - self._x_min) / (self._x_max - self._x_min)  # scale input to range [0..1]
        
        z_hid = x @ self._Wh + self._bh                      # input to hidden layer
        h_hid = self.sigmoid(z_hid)                          # output from hidden layer
        z_out = h_hid @ self._Wo + self._bo                  # input to output layer
        y_hat = z_out                                        # output (linear activation)
        
        return y_hat

        
    def train(self, x, y):
        """Perform batch update using backprop
        
        Params:
            x - input, either scalar or 2d np.ndarray with dims: [batch_size, num_inputs]
            y - target, either scalar or 2d np.ndarray with dims: [batch_size, num_outputs]
        """
        if np.isscalar(x):   x = np.array([[x]])             # needed so @ operator works ok
        if np.isscalar(y):   y = np.array([[y]])
        
        x = (x - self._x_min) / (self._x_max - self._x_min)  # scale to range [0..1]
        
        # Forward pass
        z_hid = x @ self._Wh + self._bh                      # input to hidden layer
        h_hid = self.sigmoid(z_hid)                          # output from hidden layer
        z_out = h_hid @ self._Wo + self._bo                  # input to output layer
        y_hat = z_out                                        # output (linear activation)
        
        # Backward pass
        ro_o = -1 * (y - y_hat)                              # error term output layer (linear)
        dWo = h_hid.T @ ro_o                      / len(x)   # delta weights output
        dbo = np.sum(ro_o, axis=0, keepdims=True) / len(x)   # delta biases output
        ro_h = ro_o @ self._Wo.T * self.sigmoid_deriv(z_hid) # error term hidden layer
        dWh = x.T @ ro_h                          / len(x)   # delta weights hidden
        dbh = np.sum(ro_h, axis=0, keepdims=True) / len(x)   # delta biases hidden
        
        # Update weights
        self._Wh += -self._lr * dWh
        self._bh += -self._lr * dbh
        self._Wo += -self._lr * dWo
        self._bo += -self._lr * dbo
        
        return dWh, dbh, dWo, dbo       # so we can do numerical gradient check
            
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def sigmoid_deriv(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))


