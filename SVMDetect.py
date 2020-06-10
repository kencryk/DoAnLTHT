import numpy as np

from pathlib import Path
import random
import matplotlib.pyplot as plt

import joblib
from keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import cv2
from collections import Counter
from database_firebase import goToFireBase
from ConveyorX import ConveyorX
import serial
import time
import testServo
import RPi.GPIO as GPIO


cap = cv2.VideoCapture(0)
model = VGG16(weights = 'imagenet', include_top = False)


svm_classifier = joblib.load('model_name.npy')

port = '/dev/ttyUSB0'  # note I'll use Mac OS-X if i had money

ard = serial.Serial(port, 115200, timeout=1)
time.sleep(2)  # wait for Arduino

data_string_firebase = {
    '0': '-M8tXnpZZlLKBVOieq',
    '2': '-M9F0yadBS62xuVlgstD',
    '3': '-M9F1yTfADEgrUq3HnSy',
    '-1': '-M9F2OIXnlZYkzNLajs9'
}

def predict(test_data):
    average_proba = []
    labels_index = []
    ypred_sklearn = svm_classifier.predict_proba(test_data)
    
    i=0
    for row in ypred_sklearn:
        print("Row " + str(i+1))
        max_proba = np.amax(row)
        average_proba.append(max_proba)
        if max_proba > 0.75:
            index = np.argmax(row)
            labels_index.append(index)
        else:
            labels_index.append(-1)
        print(np.amax(row))
        i+=1
        #print(labels_index)
    label = dict(Counter(labels_index))
    # Find Max Value of key of Dictionary:
    max_value = max(label.values())  # maximum value
    max_keys = [k for k, v in label.items() if v == max_value][0]  # getting all keys containing the `maximum
    # SHOW AVERAGE ACCURACY OF ALGORITHM:
    if max_keys == -1:
        return -1 # UNDEFINED
    else:
        average_proba_num = sum(average_proba)/len(average_proba)
        accuracy = max_value / 5
        print("Do chinh xac cua thuat toan: ")
        print(accuracy)
        if accuracy > 0.7:
            print("Do chinh xac trung binh: ")
            print(average_proba_num)
            print("Class: ")
            return max_keys

def detect():
    p2 = Path("output/")
    test_data = []
    # Prepare Test(Valid) Data
    for test_path in p2.glob("*.jpg"):
        test = image.load_img(test_path, target_size=(224, 224))
        test_array = image.img_to_array(test)
        # Add another Dimension for array
        test_array = np.expand_dims(test_array, axis=0)
        test_array = preprocess_input(test_array)
        # Add feature to img
        vgg16_feature_test = model.predict(test_array)
        test_data.append(vgg16_feature_test)

    test_data = np.array(test_data)
    print("Shape: ")
    print(test_data.shape)

    test_data = np.array(test_data, dtype='float32')/255.0

    N = test_data.shape[0]
    test_data = test_data.reshape(N,-1)
    print("abc:")

    # print(predict(test_data))
    return predict(test_data)

# def ImageCapture(frame):
#        dim = (1280,960)
       
if __name__ == '__main__':
    #print(detect())
    #n = 0
    ConveyorX('M310 1', ard)
    ConveyorX('M313 100', ard)
    ConveyorX('M312 -200', ard)
    ConveyorX('M310 1', ard)
    dim = (1280, 960)
    while True:
        for i in range(5):
            ret,frame = cap.read()
            #cv2.imshow('abc', frame)
            if ret == False:
                break

            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            cv2.imwrite('output/{:02}.jpg'.format(i), frame)
        
            # detect()
        result = detect()
        if result == 0:
            if goToFireBase(data_string_firebase.get(list(data_string_firebase)[0])):
                testServo.servoVat0()
                ConveyorX('M312 -200', ard)
                ConveyorX('M310 1', ard)
        elif result == 2:
            if goToFireBase(data_string_firebase.get(list(data_string_firebase)[1])):
                testServo.servoVat1()
                ConveyorX('M312 -200', ard)
                ConveyorX('M310 1', ard)
        elif result == 3:
            if goToFireBase(data_string_firebase.get(list(data_string_firebase)[2])):
                testServo.servoVat2()
                ConveyorX('M312 -200', ard)
                ConveyorX('M310 1', ard)
        elif result == -1:
            if goToFireBase(data_string_firebase.get(list(data_string_firebase)[3])):
                testServo.servoVat3()
                ConveyorX('M312 -200', ard)
                ConveyorX('M310 1', ard)
        break        
    cap.release()

