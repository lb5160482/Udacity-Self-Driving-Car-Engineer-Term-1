import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from alexnet import AlexNet
import numpy as np

n_classes = 43
learning_rate = 0.0008
batch_size = 128
epochs = 10

# TODO: Load traffic signs data.
training_file = '/input/train.p'
with open(training_file, mode='rb') as f:
    data = pickle.load(f)

# TODO: Split data into training and validation sets.
X_train, X_val, y_train, y_val = train_test_split(data['features'], data['labels'], test_size=0.33, random_state=0)

# training_num = X_train.shape[0]
# valid_num = X_val.shape[0]
# X_train = X_train[: int(training_num / 10)]
# y_train = y_train[: int(training_num / 10)]
# X_val = X_val[: int(valid_num / 10)]
# y_val = y_val[: int(valid_num / 10)]

# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(x, (227, 227))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], n_classes)
fc8_w = tf.Variable(tf.truncated_normal(shape, stddev=0.01))
fc8_b = tf.Variable(tf.zeros(n_classes))
logits = tf.add(tf.matmul(fc7, fc8_w), fc8_b)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_operation = optimizer.minimize(loss)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

# TODO: Train and evaluate the feature extraction model.
def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset : offset + batch_size], y_data[offset : offset + batch_size]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x.astype(np.float32), y: batch_y.astype(np.int32)})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# train
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print('training...')
    for i in range(epochs):
        X_train, y_train = shuffle(X_train , y_train)
        for offset in range(0, num_examples, batch_size):
            end = offset + batch_size
            batch_x, batch_y = X_train[offset : end], y_train[offset : end]
            sess.run(training_operation, feed_dict={x: batch_x.astype(np.float32), y: batch_y.astype(np.int32)})
            # print('one batch...')
        validation_accuracy = evaluate(X_val, y_val)
        print('Epoch...{}...'.format(i + 1))
        print('Validation accuracy = {:.3f}'.format(validation_accuracy))
        print()

        saver.save(sess, './model')
        print('Model saved!')