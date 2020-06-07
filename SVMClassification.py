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
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import sem
from tensorflow.keras.applications.vgg16 import VGG16

model = VGG16(weights = 'imagenet', include_top = False)


p = Path("TRAINSET/")
p2 = Path("TestSet/")
dirs = p.glob("*")
labels_dict = {'key': 0, 'card': 1, 'keychain': 2, 'led': 3}
result_test = [[1], [1], [3], [3], [2], [2], [0], [0]]

def drawImg(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    return


image_data = []
labels = []
test_data = []
test_labels = []

for folder_dir in dirs:
    #print(folder_dir)
    label = str(folder_dir).split("\\")[-1][:-1]
    #print(label)

    for img_path in folder_dir.glob("*.jpg"):
        img = image.load_img(img_path, target_size=(32, 32))
        img_array = image.img_to_array(img)
        image_data.append(img_array)
        #print(labels_dict[label])
        labels.append(labels_dict[label])

for test_path in p2.glob("*.jpg"):
    test = image.load_img(test_path, target_size=(32, 32))
    test_array = image.img_to_array(test)
    test_data.append(test_array)

print('Image data :')
print(len(image_data))
print('Labels: ' )
print(len(labels))
print('Test data: ')
print(len(test_data))

## Convert data into numpy array

image_data = np.array(image_data, dtype='float32')/255.0
test_data = np.array(test_data, dtype='float32')/255.0
labels = np.array(labels)

print("Convert data into numpy array: ")
print(image_data.shape, labels.shape)

combined = list(zip(image_data, labels))
random.shuffle(combined)

image_data[:], labels[:] = zip(*combined)



for i in range(8):
    drawImg(test_data[i])

M = image_data.shape[0]
image_data = image_data.reshape(M,-1)
print("Data conversion for One vs One classification: ")
print(image_data.shape)
print(labels.shape)

N = test_data.shape[0]


test_data = test_data.reshape(N,-1)
print("TestSet conversion for One vs One classification: ")
print(test_data.shape)

print("Number of class: ")
number_of_classes = len(np.unique(labels))
print(number_of_classes)


def classWiseData(x, y):
    data = {}

    for i in range(number_of_classes):
        data[i] = []

    for i in range(x.shape[0]):
        data[y[i]].append(x[i])

    for k in data.keys():
        data[k] = np.array(data[k])

    return data

data = classWiseData(image_data, labels)

print(data[0].shape[0])
print(data[1].shape[0])
print(data[2].shape[0])
print(data[3].shape[0])


svm_classifier = svm.SVC(kernel='rbf', probability=True)
svm_classifier.fit(image_data, labels)
ypred_sklearn1 = svm_classifier.predict(test_data)
ypred_sklearn = svm_classifier.predict_proba(test_data)
print(ypred_sklearn1)
print(ypred_sklearn)


joblib.dump(svm_classifier, 'model_name.npy')




