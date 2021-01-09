
#Example program that can classify images

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#loading the data
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

'''
#look at the data types of the variables
print(type(x_train))
print(type(y_train))
print(type(x_test))
print(type(y_test))
'''

'''
#get the shape of the arrays
print('x_train shape: ', x_train.shape)
print('y_train shape: ', y_train.shape)
print('x_test shape: ', x_test.shape)
print('y_test shape: ', y_test.shape)
'''
#OUTPUT
'''
x_train shape:  (50000, 32, 32, 3)
-- this is telling us that we have 50k rows of data, 32x32 images, with depth 3
for RBG
y_train shape:  (50000, 1)
-- contians 50k rows of data and 1 column, it is 2D
x_test shape:  (10000, 32, 32, 3)
y_test shape:  (10000, 1)
'''

#take a look at the first image as an array
index = 0
x_train[index]
#print((x_train[index]))
'''
the commented print line just gets it to print the index values for the image,
seeing the image as an array
'''

'''
#show the image as a picture
img = plt.imshow(x_train[index])
#this isn't displaying the image, i think becasue of pycharm
'''

#get the image label -- this one should be a frog
print('The image label is: ', y_train[index])
#OUTPUT:   [6]
'''
in this dataset, every number corresponds with a classification, so 6
correpsonds to a frog
'''

#get image classification
classification = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
                  'frog', 'horse', 'ship', 'truck']

#print the image class
print('The image class is: ', classification[y_train[index][0]])
#OUTPUT
'''
The image label is:  [6]
The image class is:  frog
'''

'''
If you cahnge the index = to a number other than 0, you will get a different image
class; for example, if you do index = 10, you get class = deer
'''

#convert the labels into a set of 10 numbers to input into neural network
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

#print the labels
print(y_train_one_hot)
#OUTPUT
'''
[[0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 1.]
 [0. 0. 0. ... 0. 0. 1.]
 ...
 [0. 0. 0. ... 0. 0. 1.]
 [0. 1. 0. ... 0. 0. 0.]
 [0. 1. 0. ... 0. 0. 0.]]
 
you will see a set of converted labels; a set of 10 numbers, all of the columns
will contain the value 0 except for one column that has the value 1, this will
be corresponding to that label
 
'''

#print the new label of the image/picture above
print('The one hot label is: ', y_train_one_hot[index])
#OUTPUT
'''
The one hot label is:  [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]

this set of 10 numbers cocrresponds to the label 6

'''

#normalize the pixels ot be values between 0 and 1
x_train = x_train / 255
x_test = x_test / 255
#now all of the pixel values should be between 0 and 1
'''
print(x_train[index])

when you run this print statement, you will see that the values printed are
between 0 and 1
'''



#creating the models architecture
model = Sequential()

#add the first layer, this will be a convolution layer to extract features form
#the input image, it will then create 32 5x5 relu convoluted features or feature
#maps
model.add(Conv2D(32, (5,5), activation='relu', input_shape=(32,32,3)))
#have to give it an input shape becasue it is the first layer

#creating a new layer that is a pooling layer
model.add(MaxPooling2D(pool_size = (2,2)))
#this is creating a pooling layer with a 2x2 pixel filter to get the max
#element from the feature maps

#add a second convolution layer
model.add(Conv2D(32, (5,5), activation='relu'))

#add another pooling layer
model.add(MaxPooling2D(pool_size = (2,2)))

#add a flattening layer
model.add(Flatten())
#reducing the dimansionality to a linear array


#creating a layer with 1000 neurons
model.add(Dense(1000, activation='relu'))
#relu is the activation function

#add a dropout layer with a 50% dropout rate
model.add(Dropout(0.5))

#creating a layer with 500 neurons
model.add(Dense(500, activation='relu'))

#add a dropout layer with a 50% dropout rate
model.add(Dropout(0.5))

#creating a layer with 250 neurons
model.add(Dense(250, activation='relu'))

#creating a layer with 10 neurons -- becasue we have 10 different classifications
model.add(Dense(10, activation='softmax'))


#compile the model - just made hte neural network architecture
model.compile(loss='categorical_crossentropy', optimizer = 'adam',
              metrics = ['accuracy'])

#train the model
hist = model.fit(x_train, y_train_one_hot,
                 batch_size = 256,
                 epochs = 10,
                 validation_split = 0.2)  #20%



#evaluate the model using the test dataset
model.evaluate(x_test, y_test_one_hot)[1]



#visulaize the models accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()


#visualize the models loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()




from PIL import Image

#test the model with an example
image = Image.open('cattest.jpg')

#show the image
new_image = plt.imread('cattest.jpg')
img = plt.imshow(image)

#resize the image to 32x32 pixels with a depth of 3

from skimage.transform import resize
resized_image = resize(new_image, (32,32,3))
img = plt.imshow(resized_image)


#using the image to see if the model can predict that it is an image of a cat
#get the models predictions
predictions = model.predict(np.array([resized_image]))
#show the predictions
print(predictions)


#sort the predictions from least to greatest
list_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
x = predictions

for i in range (10):
    for j in range (10):
        if x[0][list_index[i]] > x[0][list_index[j]]:
            temp = list_index[i]
            list_index[i] = list_index[j]
            list_index[j] = temp

#show the sorted labels in order (highest to lowest prediction)
print(list_index)

#print first 5 predictions - top prediction is cat
for i in range(5):
    print(classification[list_index[i]], ':', round(predictions[0][list_index[i]]*100, 2), '%')








