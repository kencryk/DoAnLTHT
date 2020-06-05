import numpy as np
import os
from pathlib import Path
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import svm
from keras.preprocessing import image
from sklearn.metrics import accuracy_score
import itertools
import joblib
import cv2
cap = cv2.VideoCapture(0)
#from SVMClassification import drawImg

result_test = [[1], [1], [1], [0], [0], [1], [1], [0], [0], [0]]
svm_classifier = joblib.load('model_name.npy')
class_name = ""
def detect():
    p2 = Path("TestSet/")
    test_data = []

    for test_path in p2.glob("*.jpg"):
        test = image.load_img(test_path, target_size=(32, 32))
        test_array = image.img_to_array(test)
        test_data.append(test_array)

    #print('Test data: ')
    #print(len(test_data))

    test_data = np.array(test_data, dtype='float32')/255.0

    #for i in range(10):
        #drawImg(test_data[i])

    N = test_data.shape[0]
    test_data = test_data.reshape(N,-1)

    #print("TestSet conversion for One vs One classification: ")
    #print(test_data.shape)

    ypred_sklearn = svm_classifier.predict(test_data)
    #print(ypred_sklearn)
    #print("Do chinh xac cua thuat toan: ")
    acc = accuracy_score(result_test, ypred_sklearn)
    #print(str(acc * 100) + '%')

    if ypred_sklearn[0] == 1:
        class_name = "But"
    if ypred_sklearn[0] == 0:
        class_name = "Chia khoa"

    #print(class_name)
    return class_name


if __name__ == '__main__':
    print(detect())

    #while True:
        #ret,frame = cap.read()
        #cv2.imshow('abc', frame)
        #if cv2.waitKey(1) & 0xFF == ord('y'):
            #cv2.imwrite('TestSet/1.jpg',frame)
            #detect()
            #cv2.destroyAllWindows()
            #break
    #cap.release()

