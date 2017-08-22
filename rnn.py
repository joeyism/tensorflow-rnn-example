import numpy
import tensorflow as tf

filename = "wonderland.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()
chars = sorted(list(set(raw_text))) # unique characters
chars_to_int = dict((c,i) for i,c in enumerate(chars)) # puts them into tuples

n_chars = len(raw_text)
n_vocab = len(chars) # no of unique characters

seq_length = 150
no_of_chars = len(chars)
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]

    dataX.append([chars_to_int[char] for char in seq_in])
    this_dataY = numpy.zeros(no_of_chars)
    this_dataY[chars_to_int[seq_out]] = 1
    dataY.append(this_dataY)

n_patterns = len(dataX)

X = numpy.reshape(dataX, (n_patterns, seq_length))
X = X/float(n_vocab) #normalize so all values are 0 to 1
Y = numpy.reshape(dataY, (n_patterns, 61))


# TensorFlow
x = tf.placeholder(tf.float32, [None, seq_length])

W1 = tf.Variable(tf.zeros([seq_length, 61]))
b1 = tf.Variable(tf.zeros([61]))
y = tf.nn.softmax(tf.matmul(x, W1) + b1)

y_ = tf.placeholder(tf.float32, [None, 61])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

sess.run(train_step, feed_dict= { x: X, y_: Y })
_, output = sess.run([train_step, y_], feed_dict= { x: X, y_: Y })
