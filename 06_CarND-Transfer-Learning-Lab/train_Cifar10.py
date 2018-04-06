from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import numpy as np
from sklearn.utils import shuffle

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# y_train.shape is 2d, (50000, 1). While Keras is smart enough to handle this
# it's a good idea to flatten the array.
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=42, stratify=y_train)

# get data size
n_train = X_train.shape[0]
n_valid = X_valid.shape[0]

# number of classes
n_classes = len(set(y_train))


def normalize(imgs):
    return (imgs / 256) - 0.5


# normalization
X_train = normalize(X_train)
X_valid = normalize(X_valid)

# parameters
learning_rate = 0.0008
epochs = 10
batch_size = 128
dropout = 0.5


def LeNet(x):
    mu = 0
    sigma = 0.1

    #     layer 1: input:32x32x3 output:28x28x6
    conv1_w = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.bias_add(tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding='VALID'), conv1_b)

    #     layer1 activation
    conv1 = tf.nn.relu(conv1)

    #     layer1 max pooling: input: 28x28x6 output: 14x14x6
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    #     layer2: input 14x14x6 output: 10x10x16
    conv2_w = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.bias_add(tf.nn.conv2d(conv1, conv2_w, strides=[1, 1, 1, 1], padding='VALID'), conv2_b)

    #     layer2 activation
    conv2 = tf.nn.relu(conv2)

    #     layer2 max pooling: input:10x10x6 output:5x5x16
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    #     flatten
    fc0 = flatten(conv2)

    #     layer3 fully-connected:input: 400 output: 120
    fc1_w = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.add(tf.matmul(fc0, fc1_w), fc1_b)

    #     layer3 activation
    fc1 = tf.nn.relu(fc1)
    #     layer3 dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    #     layer4 fully-connected:input: 120 output: 84
    fc2_w = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.add(tf.matmul(fc1, fc2_w), fc2_b)

    #     layer4 activation
    fc2 = tf.nn.relu(fc2)
    #     layer4 dropout
    fc2 = tf.nn.dropout(fc2, dropout)

    #     layer5 fully-connected:input: 84 output: n_classes
    logits_w = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean=mu, stddev=sigma))
    logits_b = tf.Variable(tf.zeros(n_classes))
    logits = tf.add(tf.matmul(fc2, logits_w), logits_b)

    return logits


x = tf.placeholder(tf.float32, (None, 32, 32, 3), name='input_x')
y = tf.placeholder(tf.int32, (None), name='output_y')
one_hot_y = tf.one_hot(y, n_classes)

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_operation = optimizer.minimize(loss)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset: offset + batch_size], y_data[offset: offset + batch_size]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x.astype(np.float32), y: batch_y.astype(np.int32)})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


# train and test
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print('Training...')
    for i in range(epochs):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, batch_size):
            end = offset + batch_size
            batch_x, batch_y = X_train[offset: end], y_train[offset: end]
            sess.run(training_operation, feed_dict={x: batch_x.astype(np.float32), y: batch_y.astype(np.int32)})

        validation_accuracy = evaluate(X_valid, y_valid)
        print('Epoch {}...'.format(i + 1))
        print('validation accuracy = {:.3f}'.format(validation_accuracy))
        print()

    saver.save(sess, '/lenet_cifar10_model/model')
    print('Model saved!')