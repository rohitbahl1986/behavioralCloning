# Import all the relevant modules

import csv
import cv2
import os
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Cropping2D, Convolution2D, Dropout, Activation
from keras.utils.visualize_util import plot
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2
import seaborn as sns
from random import randint


# Load the training data

def load_data(train_file):
    """
    This function creats a list of the training data 
    and the steering outputs from the csv file.
    This function add the center, left and right images to the list.
    A correction (tunable parameter) is applied to the steering output 
    of center camera to obtain output for left and right images.
    
    Input param : Training file absolute path
    Output param : List containing training data and labled steering outputs.
    """
    
    samples = []
    steering_measurements = []
    correction = 0.2
            
    with open(train_file) as file:
        reader = csv.reader(file)
        for row in reader:
            # Randomly drop samples with angle close to 0.
            if abs(float(row[3])) < 0.1:
                random = randint(0,1)
                if random == 1:
                    samples.append(row[0])
                    samples.append(row[1])
                    samples.append(row[2])

                    steering_measurements.append(float(row[3]))
                    steering_measurements.append(float(row[3]) + correction)
                    steering_measurements.append(float(row[3]) - correction)
            else:
                samples.append(row[0])
                samples.append(row[1])
                samples.append(row[2])

                steering_measurements.append(float(row[3]))
                steering_measurements.append(float(row[3]) + correction)
                steering_measurements.append(float(row[3]) - correction)
    
    print("Length of input data ", len(samples), len(steering_measurements))
    return samples, steering_measurements


# Define the generator function to prepare data and augument it

def generator(samples, steering, batch_size = 64):
    """
    This function loads set of images and steering output from 
    the disk to memory and returns them as batches.
    
    Input param : samples :  List contating path of images
    steering : List of steering outputs
    batch_size : Number of images to load
    
    Output param : Images and lables
    """
    num_samples = len(samples)
    
    assert len(samples)==len(steering), "Sample size mismatch"
    
    while True:
        
        for offset in range(0, num_samples, batch_size):
            image_samples = samples[offset:offset+batch_size]
            steering_samples = steering[offset:offset+batch_size]
            
            #Prepare the data in batches for saving memory.
            images = []
            steering_measurements = []
            for row, steer in zip(image_samples, steering_samples):

                image_name = row.split('/')[-1]
                current_path = '../data/IMG/' + image_name
                image = cv2.imread(current_path)
                
                #Augument the original data.
                image, steer = random_flip(image, steer)
                image = augment_brightness(image)
                image, steer = trans_image(image, steer, 100)  
                    
                images.append(image)
                steering_measurements.append(steer)

            #Convert list to numpy array for input to Keras.
            x_train = np.array(images)
            y_train = np.array(steering_measurements)
            
            yield sklearn.utils.shuffle(x_train, y_train)


def random_flip(image, steer):
    """
    This function randomly flips the input image with a 50% probabality.
    Input param : image and steeting label 
    Output : Flipped image and label or original image and label (if image not flipped)
    """
    random = randint(0,1)
    if random == 1:
        image = np.fliplr(image)
        steer = -steer
    return image, steer

# The augment_brightness and trans_image functions are referred from http://bit.ly/2wTy9Yt
def augment_brightness(image):
    """
    This function adds random brighteness to the input images
    Input param : image
    Output param : transformed image
    """
    image1 = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def trans_image(image,steer,trans_range):
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 40*np.random.uniform()-40/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(image.shape[1], image.shape[0]))
    
    return image_tr,steer_ang

# Define the deep neural network

def dnn_model():
    """
    This function defines the training model.
    For this project, nVidia model for self driving cars is choosen.
    """
    keep_prob = 1
    model = Sequential()
    model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,20),(0,0))))
    model.add(Convolution2D(24,5,5, subsample=(2, 2), W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Dropout(keep_prob))
    model.add(Convolution2D(36,5,5, subsample=(2, 2), W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Dropout(keep_prob))
    model.add(Convolution2D(48,5,5, subsample=(2, 2), W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Dropout(keep_prob))
    model.add(Convolution2D(64,3,3, W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Dropout(keep_prob))
    model.add(Convolution2D(64,3,3, W_regularizer=l2(0.001)))
    model.add(ELU())
    model.add(Dropout(keep_prob))
    model.add(Flatten())
    
    #First fully connected layer
    model.add(Dense(100))
    model.add(ELU())
    model.add(Dropout(keep_prob))
    #Second fully connected layer
    model.add(Dense(50))
    model.add(ELU())
    model.add(Dropout(keep_prob))
    #Third fully connected layer
    model.add(Dense(10))
    model.add(ELU())
    model.add(Dropout(keep_prob))
    #Final control output
    model.add(Dense(1))
    
    return model

def train_model(model, train_generator, train_samples, 
              validation_generator, validation_samples):
    """
    This function defines the deep neural network model
    used for the simulation. After training, the model saves 
    the trained network to disk. Currently, the deep learning 
    model by Nvidia for self driving car is implemented.
    
    Input param : train_generator : Training generator 
                  train_samples : Number of samples to train over
                  val_generator : Validation generator
                  validation_samples : Number of samples to validate over
    Output param : history_object from Keras 
    """

    model.compile(loss='mse', optimizer='adam')

    history_object = model.fit_generator(train_generator, samples_per_epoch= \
                     train_samples, validation_data=validation_generator, \
                     nb_val_samples=validation_samples, nb_epoch=6, verbose=1)

    model.save('model.h5')

    return history_object

def loss_visualization(history_object):
    """
    This function visualizes the training and validation
    loss outputs from Keras history object.
    
    Input param : Keras history object
    Output param : None
    """

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

def main():
    """
    Main driver script to run the network pipeline end to end.
    """
    train_file = '../data/driving_log.csv'
    samples, steering_measurements = load_data(train_file)

    #Plot distribution of raw data
    #%matplotlib inline
    sns.distplot(steering_measurements, kde=False,bins=30)
    plt.title('Histogram of the steering angles')
    plt.xlabel('Steering angle bins')
    plt.ylabel('Frequency of occurance')
    plt.show()
    
    sns.distplot(steering_measurements,
             hist_kws=dict(cumulative=True),
             kde_kws=dict(cumulative=True))
    plt.show()
    plt.title('CDF of steering angles')

    try:
        train_samples, validation_samples, train_steering, validation_steering = \
                    train_test_split(samples, steering_measurements, test_size=0.25, shuffle=True)
    except:
        print("using older version of train test split")
        train_samples, validation_samples, train_steering, validation_steering = \
                train_test_split(samples, steering_measurements, test_size=0.25)
    
    train_generator = generator(train_samples, train_steering)
    validation_generator = generator(validation_samples, validation_steering)
    model = dnn_model()
    history_object = train_model(model, train_generator, len(train_samples),
                                 validation_generator, len(validation_samples))
    
    model.summary()
    loss_visualization(history_object)
    # visualize model layout with pydot_ng
    plot(model, to_file='model.png', show_shapes=True)

if __name__=="__main__":
    main()
