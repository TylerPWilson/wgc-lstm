import tensorflow as tf

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops import init_ops

from tensorflow.python.util import nest
import layers
import numpy as np

def nz_args(a):
    return np.array([np.nonzero(a)[0], np.nonzero(a)[1]]).T

class GraphConvLSTM(object):
    
    def __init__(self, adj, n_stations, n_features, num_layers=1, n_steps=14 * 4,
               n_hidden=16, adj_trainable=False, use_sparse=False, mask_len=0,
                 learning_rate=1e-2, rank=None):
        '''
        Code to create Graph convolutional LSTM model. For ease of implementation we assume
        that each input sequence has the same number of elements so that the input dimension is
        batch_size * n_steps * n_stations * n_features, with n_steps, n_stations, and n_features
        being constant throughout training and batch_size able to vary from batch to batch.
        
        Inputs
            adj - Weighted similarity matrix. If use_sparse=True and adj_trainable=True then
                  only the non-zero elements of adj will be learned while the zero elements
                  will be fixed at zero. If use_sparse=False and adj_trainable=True then
                  all elements of adj, whether zero or non-zero will be learned. For smaller
                  graph use_sparse=False will be faster.
            n_stations - number of vertices/locations in the graph
            n_features - number of input features
            num_layers - number of layers of Graph Convolutional Cells
            n_steps - number of time steps in each input sequence
            n_hidden - hidden dimension size
            adj_trainable - whether or not the adjacency matrix can be trained
            use_sparse - boolean indicating whether or not a sparse representation of the
                         adjacency matrix should be used. For small graphs set use_sparse=False
            mask_len - Determines the number of time steps to be fed to the LSTM before making
                       making predictions. If n_steps=8 and mask_len=4 then sequences of length
                       8 will be fed to the LSTM but there will be no predictions made for the
                       first 4 inputs. This allows the LSTM to use the first 4 inputs as
                       context when making predictions for the last 4 inputs.
            learning_rate - learning rate
            rank - should be either None or a positive int. If not none then the learned
                   adjacency matrix will have a low rank factorization with an inner dimension=rank.
            
        '''
        self.n_stations = n_stations
        self.n_features = n_features
        self.num_layers = num_layers
        self.n_steps = n_steps
        self.n_hidden = n_hidden
        self.adj_trainable = adj_trainable
        self.use_sparse = use_sparse
        self.mask_len = mask_len
        self.learning_rate = learning_rate
        self.adj = adj
        self.rank = rank
        
        self._build()

        
    def _build(self):
        '''
        builds the model
        '''
        self.inputs_placeholder = tf.placeholder(tf.float32, [None, self.n_steps, self.n_stations, self.n_features], name='x')
        self.outputs_placeholder = tf.placeholder(tf.float32, [None, self.n_steps - self.mask_len, self.n_stations], name='y')
        
        self.learning_rate_variable = tf.Variable(self.learning_rate, trainable=False, name='learning_rate')
        self.learning_rate_summary = tf.summary.scalar('learning_rate', self.learning_rate_variable)
        
        batch_size = tf.squeeze(tf.gather(tf.shape(self.inputs_placeholder), [0], name='batch_size'))
        n_locations = tf.squeeze(tf.gather(tf.shape(self.inputs_placeholder), [2], name='n_locations'))
        n_features = tf.squeeze(tf.gather(tf.shape(self.inputs_placeholder), [3], name='n_features'))
        
        inputs_list = tf.split(self.inputs_placeholder, self.n_steps, axis=1)
        x_list = [tf.squeeze(x, [1]) for x in inputs_list]
                  
        # Create adjacency matrix
        with tf.name_scope('adjacency'):
            # if self.rank isn't None then use a low rank factorization of the adjacency matrix
            if self.rank:
                self.factor1 = tf.get_variable("factor1", [self.n_stations, self.rank])
                self.factor2 = tf.get_variable("factor2", [self.rank, self.n_stations])
                
                self.adj_var = [self.factor1, self.factor2]
                self.adj_tensor = self.adj_var
            # Check to see if we use a sparse or dense representation of the adjacency matrix.
            elif self.use_sparse < 0.:
                vals = self.adj[np.nonzero(self.adj)]
                indices = nz_args(self.adj)
                self.adj_var = tf.Variable(vals, dtype=tf.float32, trainable=self.adj_trainable, name='adjacency_variable')
                self.adj_tensor = tf.SparseTensor(indices, self.adj_var,  self.adj.shape)
            else: # use a dense representation of adj
                self.adj_var = tf.Variable(self.adj, dtype=tf.float32, trainable=self.adj_trainable, name='adjacency_variable_tensor')
                self.adj_tensor = self.adj_var
        
        self.w_pred = tf.Variable(tf.random_normal([self.n_hidden, 1]), name='weight')
        self.b_pred = tf.Variable(tf.random_normal([1]), name='bias')
            
        cell = GraphConvLSTMCell(self.n_hidden, self.adj_tensor, self.n_stations)
        state = None
        
        predictions = list()
        inputs = {(0, i): x for i, x in enumerate(x_list)}
        # Iterate through layers of LSTM and create each layer one at a time
        for l in range(self.num_layers):
            state = None
            with tf.variable_scope("lstm_layer-" + str(l)) as scope: # as BasicLSTMCell
                # Iterate throught time steps
                for i in range(len(x_list)):
                    with tf.name_scope('cell-' + str(i)):
                        nput = inputs[l, i]
                        if i > 0:
                            scope.reuse_variables()
                        pred, state = cell(nput, state)
                        
                        # The output of this layer will either be fed to the next layer or
                        # used as prediction
                        if (l == (self.num_layers - 1)) and (i >= self.mask_len):
                            with tf.name_scope('predictions'):
                                pred = tf.reshape(pred, [batch_size * n_locations, self.n_hidden], name='pre_reshape')
                                pred = tf.add(tf.matmul(pred, self.w_pred), self.b_pred)
                                pred = tf.reshape(pred, [batch_size, n_locations], name='post_reshape')
                                pred = tf.expand_dims(pred, axis=1)
                                predictions.append(pred)
                        else:
                            inputs[l + 1, i]  = pred
        with tf.name_scope('concat_predictions'):
            self.preds = tf.concat(predictions, 1)
        
        with tf.name_scope('loss'):
            self.mse = tf.reduce_mean(tf.squared_difference(self.outputs_placeholder, self.preds), name='mse')
            self.mse_summary = tf.summary.scalar('mse', self.mse)
            self.loss = self.mse

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        
        with tf.name_scope('optimization'):
            self.opt_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate_variable).minimize(self.loss, global_step=self.global_step, name='optimizer')

    
    def loss(self):
        return self.loss
        
    

class GraphConvLSTMCell(object):
    """ Graph Convolutional LSTM network cell (GraphConvLSTMCell).
    The implementation is based on Oliver Hennigh's implementation of a gridded convolutional
    LSTM which can be found here (https://github.com/loliverhennigh/Convolutional-LSTM-in-Tensorflow).
    """
    def __init__(self, hidden_num, adj, n_vertices,
               forget_bias=1.0, activation=tanh, name="GraphConvLSTMCell"):
        self.hidden_num = hidden_num
        self.adj = adj
        self.n_vertices = n_vertices
        self.forget_bias = forget_bias
        self.activation = activation
        self.name = name

    def zero_state(self, nput, n_vertices):
        # Clears the LSTM's cell memory and output
        zeros_dims = tf.stack([tf.shape(nput)[0], self.n_vertices, self.hidden_num*2])
        return tf.fill(zeros_dims, 0.0, name='zero_state')

    def __call__(self, inputs, state, scope=None):
        if state is None:
            state = self.zero_state(inputs, self.n_vertices)
        c, h = array_ops.split(state, 2, axis=2)

        concat = _gconv([inputs, h], 4 * self.hidden_num, self.adj, bias=True)

        i, j, f, o = array_ops.split(concat, 4, axis=2)

        new_c = tf.add(c * sigmoid(f + self.forget_bias), sigmoid(i) *
                 self.activation(j), name='c')
        new_h = tf.multiply(self.activation(new_c), sigmoid(o), name='h')
        new_state = array_ops.concat([new_c, new_h], 2, 'state')

        return new_h, new_state
      
def _gconv(args, output_size, adj, stddev=0.001, bias=True, bias_start=0.0):
    '''
    computes graph convolution
    inputs
        output_size - output dimension
        adj - weighted similarity matrix
        stddev - when variables are initialized they will be drawn from truncated normal w/ this stddev
        bias - whether or not to include bias term in output
        bias_start - initial value of bias
    '''
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 3.
    # (batch_size x height x width x arg_size)
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    n_vertices = shapes[0][1]
    for shape in shapes:
        if len(shape) != 3:
            raise ValueError("GConv is expecting 3D arguments: %s" % str(shapes))
        if shape[1] == n_vertices:
            total_arg_size += shape[2]
        else :
            raise ValueError("Inconsistent number of vertices in arguments: %s" % str(shapes))
  
    kernel = vs.get_variable("kernel", [total_arg_size, output_size], initializer=init_ops.truncated_normal_initializer(stddev=stddev))
    
    if len(args) == 1:
        res = layers.gconv(adj, args[0], kernel)
    else:
        res = layers.gconv(adj, array_ops.concat(args, 2), kernel)

    if not bias:return res
    bias_term = vs.get_variable( "bias", [output_size], initializer=init_ops.constant_initializer(bias_start))
    return res + bias_term

  
  
