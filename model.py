import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.layers import Cropping2D, Flatten, Dense, Conv2D, Lambda, MaxPooling2D, Dropout
from keras.models import Sequential, Model
import matplotlib.pyplot as plt
from keras.utils import plot_model as keras_plot
from keras.models import load_model

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# define path to data image & csv file
path = './data/'

lines = []

with open(path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# to remove the title in csv
del lines[0]

#Train small data to test - dirty implementation
#lines = lines[0:4000]

# split 20% for validation
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# define generator for model
def generator(samples, batch_size=32):
    num_samples = len(samples) # steering correction for left & right camera image
    correction = 0.2
    # index of each camera images
    center_image_idx = 0
    left_image_idx = 1
    right_image_idx = 2
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            steering = []
            for batch_sample in batch_samples:
                file_name = path + batch_sample[center_image_idx].strip()
                right_file_name = path + batch_sample[right_image_idx].strip()
                left_file_name = path + batch_sample[left_image_idx].strip()
                center_image = cv2.imread(file_name)
                right_image = cv2.imread(right_file_name)
                flipped_right = cv2.flip(right_image,1)
                left_image = cv2.imread(left_file_name)
                flipped_left = cv2.flip(left_image,1)
                center_angle = float(batch_sample[3])
                right_angle = center_angle - correction
                flipped_right_angle = right_angle * (-1.0)
                left_angle = center_angle + correction
                flipped_left_angle = left_angle*(-1.0)
                images.append(cv2.cvtColor(center_image,cv2.COLOR_BGR2RGB))
                images.append(cv2.cvtColor(right_image,cv2.COLOR_BGR2RGB))
                images.append(cv2.cvtColor(left_image,cv2.COLOR_BGR2RGB))
                images.append(cv2.cvtColor(flipped_left,cv2.COLOR_BGR2RGB))
                images.append(cv2.cvtColor(flipped_right,cv2.COLOR_BGR2RGB))
                steering.append(center_angle)
                steering.append(right_angle)
                steering.append(left_angle)
                steering.append(flipped_left_angle)
                steering.append(flipped_right_angle)
            #  DONE: trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(steering)
            yield shuffle(X_train, y_train)

# plot the loss to visualize and indicate overfitting
def plot_history(history_object,name):
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig(name + '.png')
    plt.gcf().clear()

# dirty model for quick ramp up
def build_dirty_model():
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    # Develop model
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    # Crop the image
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    model.compile(loss='mse',
                  optimizer='adam')
                  #metrics=['accuracy'])

    history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples),
        validation_data=validation_generator,
        nb_val_samples=len(validation_samples),
        nb_epoch=5)

    model.save('model.h5')

    return history_object

# define function to plot model architecture
def plot_model_architecture(model,model_name):
    keras_plot(model,to_file=model_name+'.png')

# define VGG-like model - turned out to be final model
def build_VCG_model(epoch, h5_name):
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=10)
    validation_generator = generator(validation_samples, batch_size=10)

    # Develop model
    model = Sequential()
    # Data pre-processing: Normalize the data
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    # Crop the image
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
    model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(1))
    model.compile(loss='mse',
                  optimizer='adam')
                  #metrics=['accuracy'])

    history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples),
        validation_data=validation_generator,
        nb_val_samples=len(validation_samples),
        nb_epoch=epoch)

    model.save(h5_name)
    return history_object

#history_object = build_VCG_model(5,'VGG_model.h5')
#plot_history(history_object,'VGG_w_flipped_w_dropout_model')

plot_model_architecture(load_model('VGG4_model.h5'),'latest')
