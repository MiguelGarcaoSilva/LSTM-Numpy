import numpy as np

class LSTM:

    def __init__(self, input_dim, hidden_dim, max_seq_len, output_dim):
        """
        LSTM class
        """
        # Initialize the weights
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.output_dim = output_dim

        # Forget gate (ft) parameters
        self.Wf = np.random.randn(hidden_dim, input_dim) * 0.01
        self.bf = np.zeros((hidden_dim, 1))

    @staticmethod
    def sigmoid(self, x):
        """
        Sigmoid activation function
        """
        return 1 / (1 + np.exp(-x))

    def lstm_cell_forward(self, xt, a_prev, c_prev):
        """
        LSTM cell
        - Takes input and a_t-1 and computes a_t, given to the next time step cell and to predict yt_pred

        Arguments:
        xt -- Input data for every time-step, of shape (n_x, m)
        a_prev -- Previous hidden state, of shape (n_a, m)
        c_prev -- Previous cell state, of shape (n_a, m)

        Returns:
        a_next -- Next hidden state, of shape (n_a, m)
        c_next -- Next cell state, of shape (n_a, m)
        yt_pred -- Prediction at time t, of shape (n_y, m)
        caches -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt)
        """


        input_concat = np.vstack((a_prev, xt)) #input for the gates combined
        # Forget gate (ft)
        ft = self.sigmoid(np.dot(self.Wf, input_concat) + self.bf)
        # Input gate (it)
        it = self.sigmoid(np.dot(self.Wi, input_concat) + self.bi)
        # cell state candidate (cct)
        cct = np.tanh(np.dot(self.Wc, input_concat) + self.bc)
        # Cell state update (ct)
        c_next = ft * c_prev + it * cct
        # Output gate (ot)
        ot = self.sigmoid(np.dot(self.Wo, input_concat) + self.bo)
        # Hidden state update (at)
        a_next = ot * np.tanh(c_next)

        # Prediction (yt_pred)
        yt_pred = np.dot(self.Wy, a_next) + self.by
        # Store values needed for backward pass
        caches = (a_next, c_next, a_prev, c_prev, xt)
        return a_next, c_next, yt_pred, caches
        

    def lstm_forward(self, x, a0):
        """
        LSTM step forward:
        - Takes input and a_t-1 and computes a_t, given to the next time step cell and to predict yt_pred

        Arguments:
        x -- Input data for every time-step, of shape (n_x, m)
        a0 -- Initial hidden state, of shape (n_a, m)

        Returns:
        a_next -- Next hidden state, of shape (n_a, m)
        c_next -- Next cell state, of shape (n_a, m)
        yt_pred -- Prediction at time t, of shape (n_y, m)
        caches -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt)
        """

        caches = []
        m = x.shape[1]

        a = np.zeros((self.hidden_dim, m))
        c = np.zeros((self.hidden_dim, m))
        y = np.zeros((self.output_dim, m))

        a_next = a0
        c_next = 0 #can start with 0
    
        for t in range(self.max_seq_len):
            xt = x[:, :, t] # xt shape (input_dim, batch_size) input for the time step t
            a_next, c_next, yt_pred, cache = self.lstm_cell_forward(xt, a_next, c_next)
            a[:, :, t] = a_next
            c[:, :, t] = c_next
            y[:, :, t] = yt_pred
            caches.append(cache)

        caches = (caches, x)
        return a, c, y, caches
    
    def lstm_cell_backward():
        pass
    def lstm_backward():
        pass
    def update_parameters():
        pass
    


            