'''
RNN_train.py

-> Given the "paths" and the sorted dataset, trains a RNN
'''

import os
import time
import pickle
import tensorflow as tf
import numpy as np


def check_test_set_accuracy(predictor, X_test, y_test, n_batches_test):
    pass


def sample_paths(paths, features, predictions, time_steps, mode = 0):
    '''
    Given the paths, the sorted data, the time steps, and the mode (see below),
        returns:
        1 - X, the rnn input data (sequences with time_steps length)
        2 - y, the predictions (one prediction per sequence)
        3 - the number of batches

    Modes:
    0 = uses ONLY the sequence of predictions to return a refined prediction
    1 = uses ONLY the sequence of features to return a refined prediction
    2 = uses both sequences (concatenated) to return a refined prediction
    '''
    X = 1
    y = 1
    n_batches = 1
    return X, y, n_batches


if __name__ == "__main__":
    start = time.time()

    #Loads the RNN definition (which also loads the simulation parameters)
    print("\nInitializing the RNN graph...")
    exec(open("RNN_definition.py").read(), globals())

    #Loads the sorted dataset and the paths
    print("\nLoading data...")
    with open(train_set_file, 'rb') as f:
        features_train, predictions_train = pickle.load(f)
    with open(test_set_file, 'rb') as f:
        features_test, predictions_test = pickle.load(f)
    with open(path_file, 'rb') as f:
        paths = pickle.load(f)

    #Now that the data is loaded and the RNN is defined, trains the RNN
    print("\nStarting the TF session...")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:

        #Obtains the test set [and deletes stuff that will not be needed again]
        print("Obtaining the test set...")
        X_test, y_test, n_batches_test = sample_paths(paths, features_test,
            predictions_test)
        del features_test, predictions_test

        #Runs the training phase
        current_batch = 0
        epochs_completed = 0
        sess.run(tf.global_variables_initializer())
        while(epochs_completed < epochs):

            #Gets a new the train set instance
            print("\nEpoch {0}: Sampling paths; ".format(epochs_completed),
                end='', flush = True)
            X_train, y_train, n_batches_train = sample_paths(paths,
                features_train, predictions_train)

            print("Training", end='', flush = True)
            progress = 0.0
            for batch_idx in range(n_batches_train):

                #prints a "." every 10% of an epoch
                if (batch_idx / n_batches_train) >= progress + 0.1:
                    print(".", end = '', flush = True)
                    progress += 0.1

                start_idx = batch_idx * time_steps
                end_idx = start_idx + time_steps
                # batchX = x[:,start_idx:end_idx]
                # batchY = y[:,start_idx:end_idx]

            #When the epoch is finished:
            epochs_completed += 1

            #Assesses the test performance (I know this should be a validation
            #   set, but a validation set would be generated the same way as
            #   this test set, so it's the same :D)
            check_test_set_accuracy(distance, X_test, y_test, n_batches_test)

    #After training the RNN, prints the execution time
    end = time.time()
    exec_time = (end-start)
    print("Execution time = {0:.4}s".format(exec_time))