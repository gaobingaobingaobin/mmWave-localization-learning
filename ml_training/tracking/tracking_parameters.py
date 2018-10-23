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
tracking_folder = 'tracking_data'

# -> path options: 's' for static, 'p' for pedestrian, 'c' for car
path_file = tracking_folder + '/paths'
path_options = {'s_paths': True,      # enables static paths [will have 1 per position]
                's_len_train': 20,    # number of static paths per position (train)
                's_len_test': 10,     # number of static paths per position (test)
                }