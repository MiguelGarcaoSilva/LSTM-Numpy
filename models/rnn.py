import numpy as np

#input shape (input_dim, batch_size, max_seq_len)

# at each time step, we use slice xt shape (input_dim, batch_size)

# a_next is passed to next time step shape (hidden_dim, batch_size)

# prediction y_pred shape (output_dim, batch_size, max_seq_len), yt_pred shape (output_dim, batch_size)

class RNN:
    def __init__(self, input_dim, hidden_dim, max_seq_len, output_dim):
        """
        RNN class
        """
        # Initialize the weightsx
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.output_dim = output_dim

        self.Wax = np.random.randn(hidden_dim, input_dim) * 0.01  # input to hidden
        self.Waa = np.random.randn(hidden_dim, hidden_dim) * 0.01  # hidden to hidden
        self.Wha = np.random.randn(input_dim, hidden_dim) * 0.01  # hidden to output

        self.ba = np.zeros((hidden_dim, 1))  # hidden bias
        self.by = np.zeros((input_dim, 1))   # output bias

    @staticmethod
    def softmax(self, x):
        """
        Softmax function
        """
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0, keepdims=True)
    

    def rnn_cell_foward(self, xt, a_prev):
        """
        RNN cell
        - Takes input and a_t-1 and computes a_t, given to the next time step cell and to predict yt_pred

        Arguments:
        xt -- Input data for every time-step, of shape (n_x, m)
        a_prev -- Previous hidden state, of shape (n_a, m)

        Returns:
        a_next -- Next hidden state, of shape (n_a, m)
        yt_pred -- Prediction at time t, of shape (n_y, m)
        caches -- tuple of values needed for the backward pass, contains (a_next, a_prev, xt)
        """
        # xt shape (input_dim, batch_size)
        # a_prev shape (hidden_dim, batch_size)
        # Wax shape (hidden_dim, input_dim)
        # Waa shape (hidden_dim, hidden_dim)
        # ba shape (hidden_dim, 1)

        # Compute the next hidden state (rnn cell)
        a_next = np.tanh(np.dot(self.Waa, a_prev) + np.dot(self.Wax, xt)  + self.ba)

        # Compute the output
        yt_pred = self.softmax(np.dot(self.Wya, a_next) + self.by)

        return a_next, yt_pred, (a_next, a_prev, xt)

    def rnn_step_forward(self, x, a0):
        """
         Performs the forward propagation through the RNN and computes the cross-entropy loss.
            It returns the loss' value as well as a "cache" storing values to be used in backpropagation.

        Arguments:
        x -- Input data for every time-step, of shape (n_x, m, T_x).
        a0 -- Initial hidden state, of shape (n_a, m)

        Returns:
        a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
        y_pred -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
        caches -- tuple of values needed for the backward pass, contains (list of caches, x)
        
        """

        # a shape (hidden_dim, batch_size, max_seq_len) to store the hidden states
        # y_pred shape (output_dim, batch_size, max_seq_len) to store the predictions
        # a_next shape (hidden_dim, batch_size) to store the next hidden state

        caches = []
        m = x.shape[1]
        a = np.zeros((self.hidden_dim, m, self.max_seq_len))
        y_pred = np.zeros((self.output_dim, m, self.max_seq_len))

        for t in range(self.max_seq_len):
            # xt shape (input_dim, batch_size)
            # a_prev shape (hidden_dim, batch_size) 
            # a_next shape (hidden_dim, batch_size)
            # yt_pred shape (output_dim, batch_size)
    
            xt = x[:, :, t]
            a_prev = a[:, :, t-1] if t > 0 else a_prev # starts with the initial hidden state

            a_next, yt_pred, cache = self.rnn_cell_forward(xt, a_prev)
            caches.append(cache)

            # store the hidden state and prediction
            a[:, :, t] = a_next
            y_pred[:, :, t] = yt_pred

        # store the caches
        caches = (caches, x)
        return a, y_pred, caches
    
    def rnn_cell_backward(self, da_next, da_prev, dc_next, dc_prev, xt, a_prev, caches):
        """
        Performs the backward propagation through time to compute the gradients of the loss with respect
        to the parameters. It returns also all the hidden states.
        Arguments:
        da_next -- Gradient of the next hidden state, of shape (n_a, m)
        da_prev -- Gradient of the previous hidden state, of shape (n_a, m)
        dc_next -- Gradient of the next cell state, of shape (n_a, m)
        dc_prev -- Gradient of the previous cell state, of shape (n_a, m)
        xt -- Input data for every time-step, of shape (n_x, m)
        a_prev -- Previous hidden state, of shape (n_a, m)
        caches -- tuple of values needed for the backward pass, contains (a_next, a_prev, xt)
        """
        a_next, a_prev, xt = caches

        # da_next inlcuding the loss gradient with dense and softmax



        # derivative of tanh
        # a_next = np.tanh(np.dot(self.Waa, a_prev) + np.dot(self.Wax, xt)  + self.ba)
        # dtanh = 1 - np.tanh(a_next) ** 2  
        dtanh = da_next * (1 - a_next ** 2)

        # parameters gradients
        dWaa = np.dot(dtanh, a_prev.T)
        dWax = np.dot(dtanh, xt.T)
        dba = np.sum( dtanh, axis=1, keepdims=True) #sum across the batch

        dxt = np.dot(self.Wax.T,  dtanh)
        da_prev = np.dot(self.Waa.T,  dtanh)


        pass

    def rnn_backward(self):
        pass

    def update_parameters(self):
        pass
    

    class BidirectionalRNN:

        def __init__(self):
            pass


