import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

# driving_log.csv file consists of list of sample images and driving data collected in simulator.
# Each line in the file is tuple of center,left,right,steering,throttle,brake,speed
# center,left and right are the file paths of images taken three different cameras mounted on front of the car.
with open('data/driving_log.csv') as csvfile:
        scene_data = pd.read_csv(csvfile)
center_images = scene_data.center
left_images = scene_data.left
right_images = scene_data.right
steering_data = scene_data.steering
print("Number of center images: ", len(scene_data.center)) #8036

image_center = mpimg.imread('data/IMG/' + center_images[0].split('/')[-1]) # Image from center camera
image_left = mpimg.imread('data/IMG/' + left_images[0].split('/')[-1]) # Image from left camera
image_right = mpimg.imread('data/IMG/' + right_images[0].split('/')[-1]) # Image from right camera

print("Image shape: ", image_center.shape) # (160, 320, 3)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, Cropping2D, ELU, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam

act_func = 'relu'
CROP_SHAPE = ((75,25),(0,0))
INPUT_SHAPE = (160, 320, 3)

def nn():
    model = Sequential()

    # Crop the top and bottom part of the image because part of the images with sky and rear end of car 
    # is not really useful for training.
    model.add(Cropping2D(CROP_SHAPE, input_shape=INPUT_SHAPE)) # (60, 320, 3) output shape

    # Resize
    model.add(AveragePooling2D(pool_size=(1,4), trainable=False)) # (60, 80, 3) output shape

    # Normalize
    model.add(Lambda(lambda x: x/127.5 - 1.))

    model.add(Convolution2D(24, 3, 3, subsample=(2,2), activation=act_func))
    model.add(MaxPooling2D())
    model.add(Convolution2D(36, 3, 3, subsample=(1,1), activation=act_func))
    model.add(MaxPooling2D())
    model.add(Convolution2D(48, 3, 3, subsample=(1,1), activation=act_func))
    model.add(MaxPooling2D())

    # Dropout layer
    model.add(Dropout(0.1))

    model.add(Flatten())
    # Fully connected layers layers
    model.add(Dense(100, activation=act_func))
    model.add(Dense(50, activation=act_func))
    model.add(Dense(10, activation=act_func))

    # Node for outputting steering angle
    model.add(Dense(1, trainable=False))
    return model

def flip(image):
    return cv2.flip(image, 1)

BATCH_SIZE = 64
CORRECTION = 0.25

def random_samples(data):
    indices = np.random.randint(0, len(data), BATCH_SIZE)
    image_steering_tuples = []

    for index in indices:
        camera_option = np.random.randint(3)
        if camera_option == 0:
            steering_angle = data.iloc[index]['steering']
            image_steering_tuples.append((data.iloc[index]['center'], steering_angle))
        elif camera_option == 1:
            steering_angle = data.iloc[index]['steering'] + CORRECTION
            image_steering_tuples.append((data.iloc[index]['left'], steering_angle))
        else:
            steering_angle = data.iloc[index]['steering'] - CORRECTION
            image_steering_tuples.append((data.iloc[index]['right'], steering_angle))
    return image_steering_tuples

def transform(image, steering_angle):
    if np.random.rand() <= 0.5:
        return flip(image), -1*steering_angle
    return image, steering_angle

def generator(data):
    while True:
        X_batch = []
        y_batch = []
        image_steering_tuples = random_samples(data)
        for path, steering_angle in image_steering_tuples:
            image = mpimg.imread('data/IMG/' + path.split('/')[-1])
            transformed_image, transformed_angle = transform(image, steering_angle)
            X_batch.append(transformed_image)
            y_batch.append(transformed_angle)

        yield np.array(X_batch), np.array(y_batch)

mask = np.random.rand(len(scene_data)) <= 0.9

train_samples = scene_data[mask]
validation_samples = scene_data[~mask]

train_data_generator = generator(train_samples)
validation_data_generator = generator(validation_samples)

model = nn()
number_of_samples_per_epoch = 20032
number_of_validation_samples = 6400
LEARNING_RATE = 0.0001

model.compile(optimizer=Adam(LEARNING_RATE), loss="mse")
model.fit_generator(train_data_generator, validation_data=validation_data_generator,
                    samples_per_epoch=number_of_samples_per_epoch, nb_epoch=5, nb_val_samples=len(validation_samples))

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
    
model.save_weights("model.h5")

