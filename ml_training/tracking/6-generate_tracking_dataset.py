'''
generate_tracking_dataset.py

-> Obtains the [noisy_fetures; baseline_predictions;true_labels] data triplet,
    required to train the RNN.
'''

import os
import time
import pickle
import numpy as np


def sort_data(data_folder, sets, dimentions, n_classes, predicted_feature_size):
    '''
    Organizes the data from the step "4" in the non-tracking pipeline. At the
    end of that stage, we have:
    - up to N files with k sets of noisy_features / label pairs [N = number of
        classes at the classification stage; k = number of train+test sets,
        meaning we have k samples per position, properly separated in train
        and test sets];
    - a structured file with the regression prediction for EVERY point in the k
        datasets.

    This function reorgaines everything into:
    - train set / test set;
    - within each set, we'll have a variable with "features" and another with
        "predictions" [with the non-tracking system]. All but the innermost
        dimention of these 2 variables must match, for each set;
    - each file will be a list with length P, where each entry will have k
        elements [P = number of positions in the 2D grid, p = x + (y * n_X)].

    With this new organization, we can easily sample data, given a set of
    positions (i.e. a path)
    '''

    #initializes the data
    print("Initializing data and opening predictions' file")
    n_positions = round(dimentions[0]+1) * round(dimentions[1]+1)
    features_train = [None] * n_positions
    features_test = [None] * n_positions
    predictions_train = [None] * n_positions
    predictions_test = [None] * n_positions

    with open(data_folder + 'class_predictions', 'rb') as f:
        all_predictions = pickle.load(f)

    #looping over the data
    for i in range(n_classes):
        class_data = data_folder + 'class_' + str(i)

        if os.path.isfile(class_data):
            print("Loading and sorting class ", i, end = '', flush = True)

            #processes the features
            features_train_tmp, features_test_tmp, indeces = \
                process_features(class_data, sets, n_positions, dimentions,
                                 predicted_feature_size)

            #processes the predictions
            predictions_train_tmp, predictions_test_tmp = \
                process_predictions(all_predictions, i, indeces,
                                    sets, dimentions, n_positions, n_classes)

            #updates the main data
            features_train = expand_data(features_train,
                features_train_tmp, n_positions, sets[0])
            features_test = expand_data(features_test,
                features_test_tmp, n_positions, sets[1])
            predictions_train = expand_data(predictions_train,
                predictions_train_tmp, n_positions, sets[0])
            predictions_test = expand_data(predictions_test,
                predictions_test_tmp, n_positions, sets[1])

            print("\n", end = '', flush = True)

    #Dimention checks
    sum_sorted_samples = 0
    positions_with_data = n_positions
    for i in range(n_positions):
        #train
        if features_train[i] is not None:
            assert len(features_train[i]) == len(predictions_train[i])
            sum_sorted_samples += len(features_train[i])
        else:
            assert predictions_train[i] is None
        #test
        if features_test[i] is not None:
            assert len(features_test[i]) == len(predictions_test[i])
            sum_sorted_samples += len(features_test[i])
        else:
            assert predictions_test[i] is None

        if (features_train[i] is None) and (features_test[i] is None):
            positions_with_data -= 1

    print("\nSamples sorted: ", sum_sorted_samples)
    print("Positions with data: {0} out of {1}".format(positions_with_data,
        n_positions))

    return(features_train, features_test, predictions_train, predictions_test)


def expand_data(permanent_data, tmp_data, n_positions, limit):
    '''
    Expands the permanent_data with the content of tmp_data (wrapper to avoid
    duplicate code).

    Also double checks the dimentions :D
    '''

    sum_length_before = 0
    sum_length_after = 0
    sum_length_tmp = 0

    for i in range(n_positions):

        if permanent_data[i] is not None:
            sum_length_before += len(permanent_data[i])

        #if tmp_data[i] is None, do nothing (there is nothing to update)
        if tmp_data[i] is not None:
            sum_length_tmp += len(tmp_data[i])
            if permanent_data[i] is None:
                permanent_data[i] = tmp_data[i]
            else:
                permanent_data[i].extend(tmp_data[i])

        if permanent_data[i] is not None:
            sum_length_after += len(permanent_data[i])
            assert len(permanent_data[i]) <= limit

    assert sum_length_after == (sum_length_before + sum_length_tmp)

    return permanent_data


def process_predictions(all_predictions, class_index, indeces, sets,
        dimentions, n_positions, n_classes):
    '''
    Given the indeces extracted in the process_features state, sorts the
    predictions
    '''

    #Prepares temporary variables
    predictions_train = [None] * n_positions
    predictions_test = [None] * n_positions
    n_predictions = sets[0] + sets[1]

    if str(class_index) in all_predictions:
        predictions = all_predictions[str(class_index)]
        assert len(predictions) == n_predictions
    else:
        predictions = None
        #if there are no explicit predictions, it means our predictor falls
        # back to a safe guess - the center of that class
        lateral_partition = round(n_classes**0.5)
        # class index -> x_index + (y_index * n_X)
        x_index = class_index % lateral_partition               # \in [0; lat_part -1]
        y_index = (class_index - x_index) / lateral_partition   # \in [0; lat_part -1]
        x_pos = ( (x_index + 0.5) /lateral_partition)           # \in [0; 1]
        y_pos = ( (y_index + 0.5) /lateral_partition)           # \in [0; 1]
        class_center = np.asarray([x_pos, y_pos])


    #Sorts the predictions
    sum_len_predictions = 0
    for i in range(n_predictions):
        if predictions is not None:
            this_prediction = predictions[i]
        else:
            this_prediction = None

        if i < sets[0]:
            train = True
        else:
            train = False

        this_indeces = indeces[i]
        if this_indeces is not None:
            #if the indeces are not None, their length must match the
            #   predictions' -- if there are any prediction. If there are no
            #   predictions, copies the class center ad eternum :D
            if predictions is not None:
                assert len(this_indeces) == len(this_prediction)
                sum_len_predictions += len(this_prediction)
            else:
                sum_len_predictions += len(this_indeces)

            for j, index in enumerate(this_indeces):
                if predictions is not None:
                    to_store = this_prediction[j]
                else:
                    to_store = class_center
                assert len(to_store) == 2

                if train:
                    if predictions_train[index] is None:
                        predictions_train[index] = [to_store]
                    else:
                        predictions_train[index].append(to_store)
                else:
                    if predictions_test[index] is None:
                        predictions_test[index] = [to_store]
                    else:
                        predictions_test[index].append(to_store)
        else:
            #Otherwise, if this_indeces are None, there can be no predictions
            #   as well
            assert this_prediction is None

    #double checking lengths
    sum_double_check = 0
    for i in range(n_positions):
        if predictions_train[i] is not None:
            sum_double_check += len(predictions_train[i])
        if predictions_test[i] is not None:
            sum_double_check += len(predictions_test[i])
    assert sum_double_check == sum_len_predictions

    return(predictions_train, predictions_test)


def process_features(class_data, sets, n_positions, dimentions,
        predicted_feature_size):
    '''
    Processes a single class_data file, returning sorted features and the list
    of indeces
    '''

    #Prepares temporary variables
    features_train = [None] * n_positions
    features_test = [None] * n_positions
    all_indeces = []
    n_predictions = sets[0] + sets[1]

    #Opens and processes the contents of the file
    sum_len_features = 0
    with open(class_data, 'rb') as f:
        for i in range(n_predictions):
            if i < sets[0]:
                train = True
            else:
                train = False
            print(".", end = '', flush = True)

            this_features, this_labels, _ = pickle.load(f)
            assert this_features.shape[0] == this_labels.shape[0]
            sum_len_features += this_features.shape[0]

            indeces = labels_to_indeces(this_labels, dimentions)
            if indeces is not None:
                all_indeces.append(indeces)
                assert np.min(indeces) >= 0
                assert np.max(indeces) <= n_positions - 1

                #If we got no errors so far, starts the actual sorting here
                for j, index in enumerate(indeces):
                    to_store = this_features[j]
                    assert len(to_store) == predicted_feature_size

                    if train:
                        if features_train[index] is None:
                            features_train[index] = [to_store]
                        else:
                            features_train[index].append(to_store)
                    else:
                        if features_test[index] is None:
                            features_test[index] = [to_store]
                        else:
                            features_test[index].append(to_store)
            else:
                all_indeces.append(None)

    #double checking lengths
    sum_double_check = 0
    for i in range(n_positions):
        if features_train[i] is not None:
            sum_double_check += len(features_train[i])
        if features_test[i] is not None:
            sum_double_check += len(features_test[i])
    assert sum_double_check == sum_len_features
    assert len(all_indeces) == n_predictions

    return(features_train, features_test, all_indeces)


def labels_to_indeces(labels, dimentions):
    '''
    Given a 2D numpy array with (x,y)\in[0,1] labels and the real size along
    each dimention, converts the labels into the unidimentional indeces (to
    access the lists)

    Once again: index = x + (y * n_X)
    '''

    if len(labels.shape) < 2:
        return None

    n_labels = labels.shape[0]
    n_X = round(dimentions[0] + 1)

    labels[:,0] = labels[:,0] * dimentions[0]
    labels[:,1] = labels[:,1] * dimentions[1]

    indeces = np.around(labels[:,0]) + (np.around(labels[:,1])* n_X)
    indeces = np.asarray(indeces, dtype=int)
    assert indeces.shape[0] == labels.shape[0]
    return indeces


def check_error(predictions, dimentions):
    '''
    Given the re-sorted test set prediction, checks the average error.

    Expected values for the "default" parameters:
        - ~868k positions evaluated
        - ~4.7 m avg error
        - ~13.3 m 95th percentile error
    '''

    def distance(vector_1, vector_2, dimentions):
        '''
        Assuming vector_1 and vector_2 are 2D position vectors, returns the
        distance between the 2 positions
        '''
        d_x2 = ((vector_1[0]-vector_2[0]) * dimentions[0])**2
        d_y2 = ((vector_1[1]-vector_2[1]) * dimentions[1])**2
        return (d_x2 + d_y2)**0.5

    n_positions = round(dimentions[0]+1) * round(dimentions[1]+1)
    vector_of_distances = []
    for i in range(n_positions):
        this_predictions = predictions[i]
        if this_predictions is not None:
            #gets the true position for all the predictions in this slot
            #index = x + (y * n_X)
            x_index = i % round(dimentions[0]+1)
            y_index = (i - x_index) / round(dimentions[0]+1)
            x_pos = float(x_index) / dimentions[0]
            y_pos = float(y_index) / dimentions[1]
            assert x_pos >= 0.0
            assert x_pos <= 1.0
            assert y_pos >= 0.0
            assert y_pos <= 1.0
            true_position = [x_pos, y_pos]

            #computes the distance for each prediction
            for _, single_prediction in enumerate(this_predictions):
                vector_of_distances.append(distance(single_prediction,
                    true_position, dimentions))

    avg_error = (sum(vector_of_distances) / len(vector_of_distances))
    print("Double check: Test set average error = {} m".format(avg_error))
    print("{} predictions were assessed".format(len(vector_of_distances)))


if __name__ == "__main__":
    start = time.time()

    #Loads the tracking simulation parameters
    exec(open("tracking_parameters.py").read(), globals())

    #Sorts the existing datasets
    print("\nSorting data from previous simulations...")
    data_folder = '../class_data/'
    sets = [train_sets, test_sets]
    features_train, features_test, predictions_train, predictions_test = \
        sort_data(data_folder, sets, grid, n_classes, predicted_input_size)

    #Double checks the prediction error on the test set [if it is roughly the
    #   expected, it means the previous sorting step had the correct logic]
    check_error(predictions_test, grid)

    #Stores the reshaped dataset
    print("\nStoring the reshaped dataset ...")
    with open(train_set_file, 'wb') as f:
        pickle.dump([features_train, predictions_train], f)
    with open(test_set_file, 'wb') as f:
        pickle.dump([features_test, predictions_test], f)

    #After dataset generation, prints the execution time
    end = time.time()
    exec_time = (end-start)
    print("Execution time = {0:.4}s".format(exec_time))