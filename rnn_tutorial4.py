import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

num_epochs = 100
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
num_layers = 3
echo_step = 3
batch_size = 5
num_batches = total_series_length//batch_size//truncated_backprop_length

def generateData():
    x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0

    x = x.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
    y = y.reshape((batch_size, -1))

    return (x, y)

batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

init_state = tf.placeholder(tf.float32, [num_layers, 2, batch_size, state_size])

state_per_layer_list = tf.unstack(init_state, axis=0)

rnn_tuple_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1]) for idx in range(num_layers)])

W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)


inputs_series = tf.split(batchX_placeholder, truncated_backprop_length, 1)
labels_series = tf.unstack(batchY_placeholder, axis=1)

stacked_rnn = []
for i in range(num_layers):
    stacked_rnn.append(tf.nn.rnn_cell.BasicLSTMCell(state_size, state_is_tuple=True))
cell = tf.nn.rnn_cell.MultiRNNCell(stacked_rnn, state_is_tuple=True)
states_series, current_state = tf.contrib.rnn.static_rnn(cell, inputs_series, initial_state = rnn_tuple_state)

logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels) for logits, labels in zip(logits_series,labels_series)]
total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
loss_list = []

for epoch_idx in range(num_epochs):
    x, y = generateData()
    _current_state = np.zeros((num_layers, 2, batch_size, state_size))

    for batch_idx in range(num_batches):
        start_idx = batch_idx * truncated_backprop_length
        end_idx = start_idx + truncated_backprop_length

        batchX = x[:, start_idx:end_idx]
        batchY = y[:, start_idx:end_idx]

        _total_loss, _train_step, _current_state, _predictions_series = sess.run(
            [total_loss, train_step, current_state, predictions_series]
            , 
            feed_dict = {
                batchX_placeholder: batchX,
                batchY_placeholder: batchY,
                init_state: _current_state
            }
        )






