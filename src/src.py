import numpy as np
import keras as keras
import matplotlib.pyplot as plt
import argparse
#from imutils import paths
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import to_categorical
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import glob
import cv2
# data_path = '../fruits-360_dataset_2018_02_08/fruits-360/'
# index = -1
# classes_dict = {}
#
# class_path = 'E:\\Python Workspace\\fruit-classification\\src'+'\\'
# # class_path = dir+'\\'+folder_name+'\\'
# index +=1
# # print(index, folder_name)
# for image_name in os.listdir(class_path):
#     if class_path.endswith('.png'):
#         image_path = class_path+image_name
#         # print(image_path)
#         img = Image.open(image_path)
#         img = img.resize((32,32))
#         img_raw = img.tobytes()

fruit_images = []
labels = []
for fruit_dir_path in glob.glob("../fruits-360_dataset_2018_02_08/fruits-360/Training/*"):
    fruit_label = fruit_dir_path.split("/")[-1]
    for image_path in glob.glob(os.path.join(fruit_dir_path, "*.jpg")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (45, 45))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        fruit_images.append(image)
        labels.append(fruit_label)

fruit_images = np.array(fruit_images)
labels = np.array(labels)






# initialize the data matrix and labels list


batch_size = 64
epochs = 20
num_classes = 25

(train_X,train_Y), (test_X,test_Y) = fashion_mnist.load_data()


print('Training data shape : ', train_X.shape, train_Y.shape)

print('Testing data shape : ', test_X.shape, test_Y.shape)

classes = np.unique(train_Y)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)

plt.figure(figsize=[5,5])

# Display the first image in training data
plt.subplot(121)
plt.imshow(train_X[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(train_Y[0]))

# Display the first image in testing data
plt.subplot(122)
plt.imshow(test_X[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(test_Y[0]))

train_X = train_X.reshape(-1, 32,32, 3)
test_X = test_X.reshape(-1, 32,32, 3)
train_X.shape, test_X.shape


train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X / 255.
test_X = test_X / 255.

# Change the labels from categorical to one-hot encoding
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)

# Display the change for category label using one-hot encoding
print('Original label:', train_Y[0])
print('After conversion to one-hot:', train_Y_one_hot[0])

train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)
train_X.shape,valid_X.shape,train_label.shape,valid_label.shape

fashion_model = Sequential()
fashion_model.add(Conv2D(128, kernel_size=(3, 3),activation='linear',input_shape=(5,5,3),padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2, 2),padding='same'))
fashion_model.add(Conv2D(64, (3, 3), activation='linear',input_shape=(5,5,64),padding='same'))
#fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Conv2D(32, (3, 3), activation='linear',input_shape=(5,5,32),padding='same'))
# fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
# fashion_model.add(Flatten())
fashion_model.add(Dense(64,input_shape=(13,13,16), activation='linear'))
# fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(Dense(num_classes, input_shape=(32,1,1), activation='softmax'))



fashion_model.summary()
fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
fashion_train = fashion_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))

test_eval = fashion_model.evaluate(test_X, test_Y_one_hot, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])