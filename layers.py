import tensorflow as tf
'''
Implementation of several matrix operations that are useful for computing graph convolution
The purpose of many of these functions is to easily support both sparse and dense tensors.
'''


def is_sparse(tensor):
    '''
    check if a given tensor is sparse
    '''
    return isinstance(tensor, tf.SparseTensor)


def is_dense(tensor):
    '''
    check if a given tensor is dense
    '''
    return isinstance(tensor, tf.Tensor)


def matmul(a, b, name=None, transpose_a=False, transpose_b=False):
    '''
    implementation of matrix multiplication that supports dense-dense and dense-sparse multiplication
    '''
    if is_sparse(a) and is_sparse(b):
        raise ValueError('Only a single sparse argument to matmul is supported')
    if is_sparse(a):
        out = tf.sparse_tensor_dense_matmul(a, b, name=name, adjoint_a=transpose_a, adjoint_b=transpose_b)
    elif is_sparse(b):
        out = transpose(tf.sparse_tensor_dense_matmul(b, a, name=name, adjoint_a=(not transpose_b), adjoint_b=(not transpose_a)))
    else:
        out = tf.matmul(a, b, name=name, transpose_a=transpose_a, transpose_b=transpose_b)
    return out


def reshape(tensor, shape, name=None):
    '''
    reshapes a tensor. can handle sparse and dense tensors
    '''
    if is_sparse(tensor):
        if name is None:
            name = 'sparse_reshape'
        out = tf.sparse_reshape(tensor, shape, name=name)
    elif is_dense(tensor):
        if name is None:
            name = 'dense_reshape'
        out = tf.reshape(tensor, shape, name=name)
    else:
        raise ValueError('Passed object with invalid type %s'%str(type(tensor)))
    return out


def transpose(tensor, perm=None, name='transpose'):
    '''
    can support sparse and dense transposition
    '''
    if is_sparse(tensor):
        if name is None:
            name = 'sparse_transpose'
        out = tf.sparse_transpose(tensor, perm=perm, name=name)
    elif is_dense(tensor):
        if name is None:
            name = 'dense_transpose'
        out = tf.transpose(tensor, perm=perm, name=name)
    else:
        raise ValueError('Passed object with invalid type %s'%str(type(tensor)))
    return out


def batch_matmul(x, y, batchr=False, batchl=False, shape_dim=3):
    '''
    implementation of batched matrix multiplication that supports dense-dense and dense-sparse multiplication
    '''
    batch_size = tf.get_default_graph().get_tensor_by_name('batch_size:0')
    n_locations = tf.get_default_graph().get_tensor_by_name('n_locations:0')
    
    if batchr:
        with tf.name_scope('batch_right_matmul'):
            with tf.name_scope('pre_matmul'):
                n_features = tf.gather(tf.shape(y), [2])
                perm = list(range(shape_dim)[1:]) + [0]
                y = transpose(y, perm=perm)
                new_shape = tf.concat([n_locations, n_features * batch_size], axis=0)
                
                y = reshape(y, new_shape)
            with tf.name_scope('matmul'):
                if isinstance(x, list):
                    out = matmul(x[0], matmul(x[1], y))
                else:
                    out = matmul(x, y)
                
            with tf.name_scope('post_matmul'):
                new_shape = tf.concat([n_locations, n_features, batch_size], axis=0)
                out = reshape(out, new_shape)
            
                perm = [shape_dim -1] + list(range(shape_dim)[:-1])
                
                out = tf.transpose(out, perm=perm)
    elif batchl:
        with tf.name_scope('batch_left_matmul'):
            with tf.name_scope('pre_matmul'):
                n_features = tf.gather(tf.shape(x), [2])
                new_shape = tf.concat([batch_size * n_locations, n_features], 0)
                x = reshape(x, new_shape)
            with tf.name_scope('matmul'):
                out = matmul(x, y)
            with tf.name_scope('post_matmul'):
                new_n_features = tf.gather(tf.shape(y), [1])
                out_shape = tf.concat([batch_size, n_locations, new_n_features], axis=0)
                out = tf.reshape(out, out_shape)
    return out 


def gconv(adj, x, w):
    '''
    computes graph convolution
    '''
    out = batch_matmul(adj, x, batchr=True)
    out = batch_matmul(out, w, batchl=True)
    return out

