'''
generate_paths.py

-> Computes possible paths, given the simulated data and the movement type
'''

import os
import time
import pickle
import numpy as np


def load_preprocessed_data(preprocessed_file, removed_invalid_slots,
    predicted_input_size):
    '''
    Loads the preprocessed data and performs basic shape checks. The returned
    array contains ALL possible labels (i.e. it also returns labels that are
    only possible with unlikely noise functuations - but those are a very
    limited set, don't worry :) )
    '''

    print("Loading the dataset...", end='', flush=True)
    with open(preprocessed_file, 'rb') as f:
        features, labels, invalid_slots = pickle.load(f)
    print(" done! Valid positions:", labels.shape)

    #These assertions can catch differences between the set parameters and
    #   the stored data, which avoids weird bugs
    input_size = features.shape[1]
    assert features.shape[0] == labels.shape[0]
    if (removed_invalid_slots == False):
        assert predicted_input_size == input_size

    print("[label] x range:", labels[:,0].min(), labels[:,0].max())
    print("[label] y range:", labels[:,1].min(), labels[:,1].max())
    return labels


def get_possible_positions(labels, grid, scale, shift):
    '''
    Returns a boolean list with all possible possitions.

    EXPECTED INPUTS:
        - labels: 2D array whose 1st dimention is the index of the entry, and
            the 2nd is the physical dimention (0 = x, 1 = y)
        - grid: 1D list with the physical sizes for each dimention
        - scale: 1D list with the scale for each dimention (therefore, a given
            dimention will have "grid/scale" elements along its axis)
        - shift: the offset for the starting point in each dimention
    '''

    #Creates an "empty" boolean 2D array
    x_elements = int((grid[0] / scale[0]) + 1)
    y_elements = int((grid[1] / scale[1]) + 1)
    valid_locations = np.full((x_elements, y_elements), False)
    print("valid_locations shape: ", valid_locations.shape)

    #Fills in the boolean array with the data from "labels"
    for i in range(labels.shape[0]):
        this_x = labels[i,0] * grid[0]
        this_y = labels[i,1] * grid[1]

        x_index = int(round((this_x - shift[0]) / scale[0]))
        y_index = int(round((this_y - shift[1]) / scale[1]))

        #there can be no repeated entries
        assert valid_locations[x_index][y_index] == False
        valid_locations[x_index][y_index] = True

    #the number of valid locations must match the size of "labels"
    assert np.sum(valid_locations) == labels.shape[0]
    return valid_locations


def generate_paths(path_options):
    '''
    Generates a dictionary of paths, given the options (also given as a
    dictionary)
    '''

    #Initializes the paths as a dictionary of empty lists, with a field per
    #   path type ('s' for static, 'p' for pedestrian, and 'c' for car)
    paths = {'s': [], 'p': [], 'c': []}

    #processes the static paths (this is a simpler case, just stores how many
    #   paths are needed, for further processing)
    if path_options['s_paths']:
        paths['s'] = [path_options['s_len_train'], path_options['s_len_test']]

    #processes the pedestrian paths

    #processes the car paths

    return paths


if __name__ == "__main__":
    start = time.time()

    #Loads the tracking simulation parameters
    exec(open("tracking_parameters.py").read(), globals())

    #Loads the preprocessed data
    labels = load_preprocessed_data(preprocessed_file, removed_invalid_slots,
        predicted_input_size)

    #Gets the list of possible positions, given the preprocessed data
    print("\nGenerating list with valid locations...")
    valid_locations = get_possible_positions(labels, grid, [1.0, 1.0], shift)
    del labels

    #Gets a dictionary with the desired paths
    print("\nGenerating the paths dictionary...")
    paths = generate_paths(path_options)

    #If the os path for the desired data doesn't exist, creates it
    if not os.path.exists(tracking_folder):
        os.makedirs(tracking_folder)

    #Stores the dictionary of paths
    print("\nStoring the paths ...")
    with open(path_file, 'wb') as f:
        pickle.dump(paths, f)

    #After path generation, prints the execution time
    end = time.time()
    exec_time = (end-start)
    print("Execution time = {0:.4}s".format(exec_time))
