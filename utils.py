import numpy as np
import pickle
import time

def construct_feed_dict(x, y, model):
    """
    Construct feed dictionary.
    Used to provide input to the tensorflow model
        x - a batch of predictors
        y - a batch of target
        model - an instance of GraphConvLSTM
    """
    feed_dict = dict()
    feed_dict.update({model.inputs_placeholder: x})
    feed_dict.update({model.outputs_placeholder: y})
    return feed_dict

def get_random_batch(x, y, n_steps, mask_len, batch_size):
    """
    Given a dataset returns a batch of random samples from that dataset.
    Parameters:
        x - Tensor of predictors. Must have shape T * S * F where T denotes number of time steps, S is the number of stations (i.e. vertices in your graph), and F is the number of input features
        y - Tensor of targets. Must have shape T * S.
        n_steps - length of each sample sequence in the batch
        mask_len - the number of elements at the beginning of each sequence to be used as context but not predicted. mask_len < n_steps
        batch_size - batch size
        
    Returns:
        x_batch - of shape batch_size * n_steps * S * F
        y_batch - of shape batch_size * (n_steps - mask_len) * S
    """
    x_batches = list()
    y_batches = list()

    for i in range(batch_size):
        start_ind = np.random.randint(0, x.shape[0] - n_steps - 1)
        x_batches.append(np.expand_dims(x[start_ind:start_ind + n_steps, :, :], 0))
        y_batches.append(np.expand_dims(y [start_ind:start_ind + n_steps, :], 0))
    x_batch = np.concatenate(x_batches, axis=0)
    y_batch = np.concatenate(y_batches, axis=0)[:,mask_len:, :]

    return x_batch, y_batch


def get_sequential_sample(x, y, batch_num, n_steps, mask_len, include_masked=False):
    """
    Given a dataset returns the batch_num .
    Parameters:
        x - Tensor of predictors. Must have shape T * S * F where T denotes number of time steps, S is the number of stations (i.e. vertices in your graph), and F is the number of input features
        y - Tensor of targets. Must have shape T * S.
        n_steps - length of each sample sequence in the batch
        mask_len - the number of elements at the beginning of each sequence to be used as context but not predicted. mask_len < n_steps
        batch_size - batch size
        
    Returns:
        x_batch - of shape batch_size * n_steps * S * F
        y_batch - of shape batch_size * (n_steps - mask_len) * S
    """
    if include_masked:
        mask_offset = mask_len
    else:
        mask_offset = 0
    # return shape [0, 0, 0, 0] array if there is no data left
    if (batch_num + 1) * (n_steps - mask_offset) + mask_offset >= x.shape[0]:
        return np.random.randn(0, 0, 0, 0), np.random.randn(0, 0, 0, 0)
    x_b = np.expand_dims(x[batch_num * (n_steps - mask_offset):(batch_num + 1) * (n_steps - mask_offset) + mask_offset, :, :], axis=0)
    y_b = np.expand_dims(y[batch_num * (n_steps - mask_offset) + mask_len:(batch_num + 1) * (n_steps - mask_offset) + mask_offset, :], axis=0)
    return x_b, y_b


def get_sequential_batch(x, y, n_steps, mask_len, include_masked=False):
    xs, ys = list(), list()
    b_num = 0
    new_x, new_y = get_sequential_sample(x, y, b_num, n_steps, mask_len, include_masked)

    while new_x.shape[0] > 0:
        xs.append(new_x)
        ys.append(new_y)
        b_num += 1
        new_x, new_y = get_sequential_sample(x, y, b_num, n_steps, mask_len, include_masked)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

def evaluate(x, y, model, sess, n_steps, mask_len):
    """
    Computes the loss on a data set. Can be used to compute the MSE on the validation and test sets
    Parameters:
        x - Tensor of predictors. Must have shape T * S * F where T denotes number of time steps, S is the number of stations (i.e. vertices in your graph), and F is the number of input features
        y - Tensor of targets. Must have shape T * S.
        model - an instance of GraphConvLSTM
        sess - a tensorflow session
        n_steps - length of each sample sequence in the batch
        mask_len - the number of elements at the beginning of each sequence to be used as context but not predicted. mask_len < n_steps
    Returns:
        batch_loss - overall batch loss including regularization loss. Will be 
    """
    t_test = time.time()

    x_b, y_b = get_sequential_batch(x, y, n_steps, mask_len)

    feed_dict_val = construct_feed_dict(x_b, y_b, model)
    
    fetches = [model.mse]
    
    batch_mse = sess.run(fetches, feed_dict=feed_dict_val)
    
    return batch_mse[0], time.time() - t_test

def load_data():
    """
    Loads the example data used for the demo. This data is for the IGRA temperature task reported in the paper.
    Returns:
        adj - an adjacency matrix with pairwise similarities
        x_train - tensor of training predictors with shape T * S * F where T is the number of time steps, S is the number of stations, and F is the number of input features
        y_train - tensor of training targets with shape T * S * F
        x_val
        y_val
        x_test
        y_test
    """
    with open('demo_data_final.pickle', 'rb') as f:
        data_dic = pickle.load(f)
    return data_dic['adj'], data_dic['x_train'], data_dic['y_train'], data_dic['x_val'], data_dic['y_val'], data_dic['x_test'], data_dic['y_test']
