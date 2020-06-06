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

result_test = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
detect_result = []
svm_classifier = joblib.load('model_name.npy')
class_name = ""
def detect():
    p2 = Path("TestSet2/")
    test_data = []

    for test_path in p2.glob("*.jpg"):
        test = image.load_img(test_path, target_size=(32, 32))
        test_array = image.img_to_array(test)
        test_data.append(test_array)

    #print('Test data: ')
    #print(len(test_data))

    test_data = np.array(test_data, dtype='float32')/255.0

    def drawImg(img):
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        return

    #for i in range(10):
    drawImg(test_data[0])

    N = test_data.shape[0]
    test_data = test_data.reshape(N,-1)

    #print("TestSet conversion for One vs One classification: ")
    #print(test_data.shape)

    ypred_sklearn = svm_classifier.predict_proba(test_data)
    ypred_sklearn1 = svm_classifier.predict(test_data)
    print(ypred_sklearn1)

    #print(ypred_sklearn)
    #print("Do chinh xac cua thuat toan: ")
    #acc = accuracy_score(result_test, ypred_sklearn)
    #print(str(acc * 100) + '%')

    #for i in range(8):
    # if ypred_sklearn[0] == 1:
    #     class_name = "Card"
    # if ypred_sklearn[0] == 0:
    #     class_name = "Chia khoa"
    # if ypred_sklearn[0] == 2:
    #     class_name = "Moc khoa"
    # if ypred_sklearn[0] == 3:
    #     class_name = "LED"

    #print(class_name)
    return ypred_sklearn


if __name__ == '__main__':
    #print(detect())
    #n = 0

    while True:
        ret,frame = cap.read()

        cv2.imshow('abc', frame)
        if cv2.waitKey(1) & 0xFF == ord('y'):
            #n+=1
            dim = (1280, 960)
            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            #gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #(thresh, binary) = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
            #cv2.imwrite('TestSet2/{:02}.jpg'.format(n), binary)
            cv2.imwrite('TestSet2/1.jpg', frame)
            print(detect())
            # detect_result.append(detect())
            # acc = accuracy_score(result_test, detect_result)
            # print(str(acc * 100) + '%')
            #cv2.destroyAllWindows()
            #break
    cap.release()

