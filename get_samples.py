
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
'''

path = 'get_samples'
samples = os.listdir(path)
squares=[]
for pic in samples:
    img = cv2.imread(path+"/"+pic)
    img = cv2.resize(img, (28*9,28))
    X = np.array_split(img,9,axis=1)
    w=[]
    for x in X:
        w.append(x)
    squares.append(w)

picsn=0
for pics in squares:
    picn=1
    for pic in pics:
        cv2.imwrite("D:/sudoku_solver/digit_images/"+str(picn)+"/"+str(picn)+"_"+str(picsn)+".png", pic)
        cv2.imwrite
        picn+=1
    picsn+=1
'''

for i in range(0, 10):
    sample_path = "D:/sudoku_solver/digit_images/"+str(i)
    samples = os.listdir(sample_path)
    for sample in samples:
        img = cv2.imread(sample_path+"/"+sample)
        print(img.shape)
        w,h=img.shape[1],img.shape[0]
        dw, dh = w//3, h//3
        count=0
        for i in range(0,dw,1):
            for j in range(0,dh,1):
                tmp_img = img[i:min(w,i+w), j:min(h,j+h)]
                cv2.imwrite(sample_path+"/"+str(count)+".png", tmp_img)
                count+=1
