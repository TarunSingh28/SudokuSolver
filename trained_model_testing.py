
import tensorflow as tf
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

new_model = tf.keras.models.load_model('digits3.model')
new_model.summary()

for x in range(1,2):
    img = cv.imread(f'{x}.png')[:,:,0]
    img = cv.resize(img, (28,28))
    img = np.array([img])
    print(img.shape)
    prediction = new_model.predict(img)
    print(f'The result is probably: {np.argmax(prediction)}')
    print(prediction)
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
