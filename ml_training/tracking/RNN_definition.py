'''
RNN_definition.py

-> Defines the RNN Architecture
'''


#Loads the tracking simulation parameters
exec(open("tracking_parameters.py").read(), globals())

#Sets the target GPU
os.environ["CUDA_VISIBLE_DEVICES"]=str(target_gpu)
print("[Using GPU #{0}]".format(target_gpu))

#Loads hyperparameters from the tracking parameters
dictionary = rnn_parameters
batch_size = dictionary['batch_size']
epochs = dictionary['epochs']
lr = dictionary['learning_rate']
time_steps = dictionary['time_steps']
rnn_neurons = dictionary['rnn_neurons']


##########################################################
#   NN Architecture

#Control variables
learning_rate_var = tf.placeholder(tf.float32, shape=[])  # The current learning rate

#Input/Output data placeholders
#   (shape=[batch_sice/None, time_steps, single_element_shape])
input_size = predicted_input_size
input_sequence = tf.placeholder(tf.float32, [None, time_steps, input_size])
real_location = tf.placeholder(tf.float32, [None, 2])

#RNN layer definition
cell = tf.nn.rnn_cell.BasicRNNCell(rnn_neurons)
initial_state = cell.zero_state(batch_size, dtype=tf.float32)
rnn_output, _ = tf.nn.dynamic_rnn(cell, input_sequence,
    initial_state = initial_state)
#rnn_output shape: [batch_size, time_steps, rnn_neurons]
#   the "_", usually called state, is the same as rnn_output[-1] for a single
#   RNN layer.


#Regression (output layer)
W2 = tf.Variable(np.random.rand(rnn_neurons, 2),dtype=tf.float32)
b2 = tf.Variable([0.5, 0.5], dtype=tf.float32)

#rnn_output[-1] fetches the final state of the RNN, i.e. its output AFTER going
#   through the sequence
prediction = tf.matmul(rnn_output[:,-1,:], W2) + b2
prediction_clip = tf.clip_by_value(prediction, 0.0, 1.0)

#   NN Architecture
##########################################################


##########################################################
#   TensorFlow functions

#DISTANCE -------------------------------------------------------
# contains series with delta_x AND delta_y
delta_position = real_location - prediction_clip

# contains delta_x^2 AND delta_y^2, SCALING UP BACK TO THE ORIGINAL RANGE [0;data_downscale]
delta_squared = tf.square(delta_position * data_downscale)

# reduces dimension 1 through a sum (i.e. = delta_x^2  + delta_y^2)
distance_squared = tf.reduce_sum(delta_squared,1)

# and then computes its square root
distance = tf.sqrt(distance_squared, name='distance')


#LOSS AND TRAIN -------------------------------------------------------
# loss as MMSE
loss = tf.reduce_mean(tf.square(real_location - prediction))

# defines the optimizer (ADAM)
train_step = tf.train.AdamOptimizer(learning_rate = learning_rate_var).minimize(loss)

#   TensorFlow functions
##########################################################
