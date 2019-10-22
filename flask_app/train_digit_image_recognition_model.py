# from keras.datasets import mnist
# import matplotlib.pyplot as plt

# # Load dataset (download if needed)
# (X_train, y_train), (X_test, y_test) = mnist.load_data()

# # visualizing input images
# plt.subplot(221)
# plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
# plt.subplot(222)
# plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
# plt.subplot(223)
# plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
# plt.subplot(224)
# plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# plt.show()

import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import backend as K

# fix the seed
seed = 7
np.random.seed(seed)

batch_size = 128
num_classes = 10
epochs = 12

# input dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# image - a matrix of densities for each channel red, green, and blue
# pxqxr tensor
if K.image_data_format() == 'channels_first': # theano
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# normalize pixel density
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# one hot encoding
# output -- [ 0 0 0 0 0 0 0 1 0 0 ] --> 0 to 9
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

def prepare_model():
    # static model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), 
                     input_shape=input_shape, 
                     activation='relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))
    # regularize
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten()) # tensor --> single vector
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, 
                  optimizer=keras.optimizers.Adam(), 
                  metrics=['accuracy'])

    return model

# build a model
model = prepare_model()

# fit
model.fit(X_train, y_train, 
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test, y_test), 
          verbose=2)

model.save('./model/model.h5') # standard compression output for deep learning keras model

# final eval
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# # Generating test images
# from PIL import Image
# from keras.datasets import mnist
# import numpy as np

# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# for i in np.random.randint(0, 10000+1, 10):
#     arr2im = Image.fromarray(X_train[i])
#     arr2im.save('./test_images/{}.png'.format(i), "PNG")


# # Test predicting test images offline
# import keras
# from PIL import Image
# import numpy as np

# model = keras.models.load_model('./model/model.h5')

# # im2arr = np.array(im).reshape((1, 1, 28, 28)) # for theano backend
# im2arr = np.array(im).reshape((1, 28, 28, 1)) # for tensorflow backend
# pred_digit = np.argmax(model.predict(im2arr))
# print(pred_digit)