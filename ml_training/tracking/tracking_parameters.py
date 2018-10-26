'''
tracking_parameters.py

-> Analogous to simulation_parameters.py for the non-tracking version
-> Loads "simulation_parameters.py" (the tracking part is an extension of the
    previous results, so everything should remain consistent)
-> gets the simulation parameters for the tracking part
'''

#Runs "simulation_parameters.py" and keeps its variables
exec(open("../simulation_parameters.py").read(), globals())


#Overwrites some old parameters
target_gpu = 0
preprocessed_file = '../processed_data/tf_dataset'


#And stores new ones
tracking_folder = 'tracking_data/'
train_set_file = tracking_folder + 'train_set'
test_set_file = tracking_folder + 'test_set'

# -> path options: 's' for static, 'p' for pedestrian, 'c' for car
path_file = tracking_folder + 'paths'
path_options = {'s_paths': True,      # enables static paths [will have 1 per position]
                's_len_train': 20,    # number of static paths per position (train)
                's_len_test': 10,     # number of static paths per position (test)
                }


# RNN parameters
rnn_parameters = {  'batch_size': 32,
                    'epochs': 10,
                    'learning_rate': 1e-4,
                    'time_steps': 3,
                    'rnn_neurons': 256
                    }
#RNN modes:
#   0 = uses ONLY the sequence of predictions to return a refined prediction
#   1 = uses ONLY the sequence of features to return a refined prediction
#   2 = uses both sequences (concatenated)
rnn_mode = 0
