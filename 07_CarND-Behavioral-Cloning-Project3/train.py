import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers import Convolution2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


batch_size = 128

data_path = '/input/'

lines = []
with open(data_path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
lines = lines[1:]
for line in lines:
    for i in range(3):
        source_path = line[i]
        file_name = source_path.split('/')[-1]
        file_path = data_path + '/IMG/' + file_name
        image = cv2.imread(file_path)
        # convert to RGB since in drive.py we are processing RGB image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        measurement = float(line[3])
        if i == 1:  # left camera image with steering offset towards right
            measurement += 0.1
        elif i == 2:  # right camera image with steering offset towards left
            measurement -= 0.1

        images.append(image)
        measurements.append(measurement)
        # add flip data
        images.append(cv2.flip(image, 1))
        measurements.append(measurement * -1.0)

X_train, X_valid, y_train, y_valid = train_test_split(images, measurements, test_size=0.2)


def generator(x, y, batch_size=128):
    num_samples = len(x)
    while 1:  # Loop forever so the generator never terminates
        shuffle(x, y)
        for offset in range(0, num_samples, batch_size):
            x_batch = np.array(x[offset:offset + batch_size])
            y_batch = np.array(y[offset:offset + batch_size])

            yield x_batch, y_batch


train_generator = generator(X_train, y_train, batch_size)
valid_generator = generator(X_valid, y_valid, batch_size)

# define model
model = Sequential()
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))
model.add(Convolution2D(32, (3, 3), strides=(2, 2), activation='relu'))
model.add(Convolution2D(64, (3, 3), strides=(2, 2), activation='relu'))
model.add(Convolution2D(128, (3, 3), strides=(2, 2), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
print(model.summary())
# train using generator to save memory
history_object = model.fit_generator(train_generator, steps_per_epoch=len(X_train)/batch_size,
                                     validation_data=valid_generator,
                                     validation_steps=len(X_valid)/batch_size, epochs=3)

model.save('/output/model.h5')


# plot
# print the keys contained in the history object
print(history_object.history.keys())

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('/output/loss.png')
