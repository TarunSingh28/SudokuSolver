
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils.np_utils import to_categorical

path = './archive/assets'
images_to_train=[]
digit_label = []
digit_folders = os.listdir(path)

for x in range(0,10):
    digit_pics = os.listdir(path+"/"+str(x))
    for y in digit_pics:
        img = cv2.imread(path+"/"+str(x)+"/"+y, 0)
        img = cv2.resize(img, (28,28))
        images_to_train.append(img)
        digit_label.append(x)

images_to_train = np.array(images_to_train)
digit_label = np.array(digit_label)

def normalize_img(img):
    img = img/255
    return img

images_to_train = np.array(list(map(normalize_img, images_to_train)))
print(images_to_train.shape)

images_to_train = images_to_train.reshape(images_to_train.shape[0], images_to_train.shape[1], images_to_train.shape[2], 1)

digit_label = to_categorical(digit_label, 10)

print(images_to_train.shape)
print(digit_label.shape)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

print(model.summary())
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(images_to_train, digit_label, epochs=50)
model.save('digits3.model')

