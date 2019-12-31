import time
import tensorflow as tf

import utils
import numpy as np
import gConvLSTM as gconv
import random
    
def train(sess, args, adjTrainable, nlayers, use_sparse, bsize,
          nbatches, learningRate, nhidden, nsteps, maskLen, rank):
    
    seed = 123
    np.random.seed(seed)
    tf.set_random_seed(seed) 
    
    # Load data
    adj, x_train, y_train, x_val, y_val, x_test, y_test = utils.load_data()
    train_var, val_var, test_var = np.var(y_train), np.var(y_val), np.var(y_test)

    # Build model
    model = gconv.GraphConvLSTM(adj, x_train.shape[1], x_train.shape[2],
                                num_layers=nlayers, n_steps=nsteps,
                                n_hidden=nhidden,
                                adj_trainable=adjTrainable,
                                use_sparse=use_sparse, mask_len=maskLen,
                                learning_rate=learningRate, rank=rank)
    
    init = tf.global_variables_initializer()
    
    # Initialize tensorflow variables
    sess.run(init)

    best_val = 99999999.
    best_test = 99999999.
    best_batch = 0
    last_lr_update = 0
    
    # Display parameters
    display_step = 1000
    between_lr_updates = 500
    lr_factor = 0.9
    
    learningRate = sess.run(model.learning_rate_variable)

    cost_val = []
    
    train_mse = 0
    denom = 0.
    
    batches_complete = sess.run(model.global_step)
    
    saved_test_mse = 9999
    
    # Train model
    while batches_complete < nbatches:
        x_train_b, y_train_b = utils.get_random_batch(x_train, y_train, nsteps, maskLen, bsize)
    
        t = time.time()
        # Construct feed dictionary
        feed_dict = utils.construct_feed_dict(x_train_b, y_train_b, model)
        feed_dict[model.learning_rate_variable] = learningRate
    
        # Training step
        _, batch_mse, batches_complete = sess.run([model.opt_op, model.mse, model.global_step], feed_dict=feed_dict)
        train_mse += batch_mse
        
        batch_time = time.time() - t
        denom += 1
    
        # Periodically compute validation and test loss
        if batches_complete % display_step == 0 or batches_complete == nbatches:
            # Validation
            val_mse, duration = utils.evaluate(x_val, y_val, model, sess, nsteps, maskLen)
            cost_val.append(val_mse)
            
            test_mse, duration = utils.evaluate(x_test, y_test, model, sess, nsteps, maskLen)
    
            # Print results
            print(
                    "Batch Number:%04d" % (batches_complete),
                    "train_mse={:.5f}".format(train_mse / denom),
                    "val_mse={:.5f}".format(val_mse),
                    "test_mse={:.5f}".format(test_mse),
                    "test_rsq={:.5f}".format(1 - (test_mse/test_var)),
                    "time={:.5f}".format(batch_time),
                    "lr={:.8f}".format(learningRate))
            train_mse = 0
            denom = 0.
    
            # Check if val loss is the best encountered so far
            if val_mse < best_val:
                best_val = val_mse
                saved_test_mse = test_mse
                best_batch = batches_complete - 1
            
            if (batches_complete - best_batch > between_lr_updates) and (batches_complete - last_lr_update > between_lr_updates):
                learningRate = learningRate * lr_factor
                last_lr_update = batches_complete
            
    print('best val mse: {0}, test mse: {1}, test rsq: {2}'.format(best_val, saved_test_mse, 1 - (saved_test_mse/test_var)))

def default_exp(nbatches):
    dic = {}
    dic['nlayers'] = 2
    dic['use_sparse'] = False
    dic['bsize'] = 32
    dic['nbatches'] = nbatches
    dic['learningRate'] = 0.01
    dic['nhidden'] = 8
    dic['nsteps'] = 8
    dic['maskLen'] = 4
    dic['adjTrainable'] = True
    dic['rank'] = None
    
    return dic    

def create_session():
    job_ppn = 4
    config = tf.ConfigProto(intra_op_parallelism_threads=job_ppn, inter_op_parallelism_threads=job_ppn - 2,
                            allow_soft_placement=True, device_count={'CPU': 1})
    sess = tf.Session(config=config)
    return sess

if __name__ == "__main__":
    print('******Running Demo******')

    nbatches = 20000
    parm_dic = default_exp(nbatches)
    train(create_session(), parm_dic, **parm_dic)

