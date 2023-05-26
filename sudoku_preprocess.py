
import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sudoku_solve_code import solve_try

new_model = tf.keras.models.load_model('digits3.model')

img_path = 'sudoku_try'
img = cv2.imread('sudoku_try.png', 0)
img = cv2.resize(img, (458,458))
img = cv2.bitwise_not(img)
thresh = cv2.adaptiveThreshold(img, 255, 1, 1, 11, 2)
plt.imshow(img)
plt.show()

def big_contour(contours):
    biggest = np.array([])
    max_a = 0
    for i in contours:
        a = cv2.contourArea(i)
        if a>50:
            p = cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i, 0.02*p, True)
            if a>max_a and len(approx)==4:
                biggest=approx
                max_a = a
    return biggest, max_a


img_contour = img.copy()
img_big_contour = img.copy()
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img_contour, contours, -1, (0,0,255), 25)
plt.imshow(img_contour)
plt.show()
biggest, max_a = big_contour(contours)
print(biggest)

# ARRANGE CLOCKWISE
biggest = biggest.tolist()
for x in biggest:
    x.sort(key=lambda x:(x[0],x[1]))
# GET CORNERS FOR USE LATER
xl = biggest[0][0][0]
xr = biggest[2][0][0]
yl = biggest[0][0][1]
yr = biggest[2][0][1]
biggest = np.array(biggest)
print(biggest)

cv2.drawContours(img_big_contour, biggest, -1, (0,0,0), 10)
warped = img[yl:yr, xl:xr]  # BASICALLY CROP THE IMAGE
print(warped.shape)
squares=[]
X = np.array_split(warped,9,axis=0)
for x in X:
    y = np.array_split(x, 9, axis=1)
    for z in y:
        squares.append(z)

digit_predicted = []
for i in range(81):
    tmp_img = squares[i]
    tmp_img = cv2.resize(tmp_img, (28,28))
    if i==0:
        plt.imshow(tmp_img)
        plt.show()
    tmp_img = np.array([tmp_img])
    prediction = new_model.predict(tmp_img)
    prob = np.amax(prediction)
    if prob>0.8:
        digit_predicted.append(np.argmax(prediction))
    else:
        digit_predicted.append(0)
print(digit_predicted)

plt.imshow(squares[0])
plt.show()

digit_predicted2 = []
for i in range(0,81,9):
    digit_predicted2.append(digit_predicted[i:i+9])
for x in digit_predicted2:
    print(' | '.join(map(str, x)))
    print('-'*34)

'''
arr_in = digit_predicted
arr=[]
for i in range(9):
    arr.append(arr_in[9*i:9*(i+1)])

if solve_try(arr):
    print("SOLVED")
    for x in arr:
        print(x)
else:
    print("UNSOLVED")     
'''
