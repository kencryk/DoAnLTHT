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

p = Path("Images/")
p2 = Path("TestSet/")
dirs = p.glob("*")
labels_dict = {'key': 0, 'pen': 1}
result_test = [[1], [1], [1], [0], [0], [1], [1], [0], [0], [0]]

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



for i in range(10):
    drawImg(test_data[i])

#for i in range(5):
    #drawImg(image_data[i])


class SVM:
    def __init__(self, C=1.0):
        self.C = C
        self.W = 0
        self.b = 0

    def hingeLoss(self, W, b, X, Y):
        loss = 0.0

        loss += .5 * np.dot(W, W.T)

        m = X.shape[0]

        for i in range(m):
            ti = Y[i] * (np.dot(W, X[i].T) + b)
            loss += self.C * max(0, (1 - ti))

        return loss[0][0]

    def fit(self, X, Y, batch_size=50, learning_rate=0.001, maxItr=500):

        no_of_features = X.shape[1]
        no_of_samples = X.shape[0]

        n = learning_rate
        c = self.C

        # Init the model parameters
        W = np.zeros((1, no_of_features))
        bias = 0

        # Initial Loss

        # Training from here...
        # Weight and Bias update rule
        losses = []

        for i in range(maxItr):
            # Training Loop

            l = self.hingeLoss(W, bias, X, Y)
            losses.append(l)
            ids = np.arange(no_of_samples)
            np.random.shuffle(ids)

            # Batch Gradient Descent(Paper) with random shuffling
            for batch_start in range(0, no_of_samples, batch_size):
                # Assume 0 gradient for the batch
                gradw = 0
                gradb = 0

                # Iterate over all examples in the mini batch
                for j in range(batch_start, batch_start + batch_size):
                    if j < no_of_samples:
                        i = ids[j]
                        ti = Y[i] * (np.dot(W, X[i].T) + bias)

                        if ti > 1:
                            gradw += 0
                            gradb += 0
                        else:
                            gradw += c * Y[i] * X[i]
                            gradb += c * Y[i]

                # Gradient for the batch is ready! Update W,B
                W = W - n * W + n * gradw
                bias = bias + n * gradb

        self.W = W
        self.b = bias
        return W, bias, losses

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
#print(data[2].shape[0])
#print(data[3].shape[0])

"""Combines data of two classes into a single matrix"""


def getDataPairForSVM(d1, d2):
    l1, l2 = d1.shape[0], d2.shape[0]
    samples = l1 + l2
    features = d1.shape[1]

    data_pair = np.zeros((samples, features))
    data_labels = np.zeros((samples,))

    data_pair[:l1, :] = d1
    data_pair[l1:, :] = d2

    data_labels[:l1] = -1
    data_labels[l1:] = 1

    return data_pair, data_labels

mySVM = SVM()
xp, yp = getDataPairForSVM(data[0], data[1])
w,b,loss = mySVM.fit(xp,yp,learning_rate=0.00001,maxItr=1000)
#plt.plot(loss)


def trainSVMs(x, y):
    svm_classifiers = {}

    for i in range(number_of_classes):
        svm_classifiers[i] = {}
        for j in range(i + 1, number_of_classes):
            xpair, ypair = getDataPairForSVM(data[i], data[j])
            wts, b, loss = mySVM.fit(xpair, ypair, learning_rate=0.00001, maxItr=1000)
            svm_classifiers[i][j] = (wts, b)

            #plt.plot(loss)
            #plt.show()

    return svm_classifiers

svm_classifiers = trainSVMs(image_data, labels)

#cats_dogs = svm_classifiers[0][1]
#cats_humans = svm_classifiers[0][3]
#print(cats_dogs[0].shape)
#print(cats_dogs[1])
keys_pens = svm_classifiers[0][1]
#keys_usbs = svm_classifiers[0][2]
print(keys_pens[0].shape)
print(keys_pens[1])

def binaryPredict(x,w,b):
    z = np.dot(x,w.T) + b
    if z >= 0:
        return 1
    else:
        return -1


def predict(x):
    count = np.zeros((number_of_classes,))

    for i in range(number_of_classes):
        for j in range(i + 1, number_of_classes):
            w, b = svm_classifiers[i][j]

            # Take a majority prediction
            z = binaryPredict(x, w, b)

            if z == 1:
                count[j] += 1
            else:
                count[i] += 1

    final_prediction = np.argmax(count)
    return final_prediction

for i in range(10):
    print("Du doan ket qua cua anh thu " + str(i+1) + " trong TestSet")
    print(predict(test_data[i]))
    test_labels.append(predict(test_data[i]))
#print(labels[0])




def accuracy(x, y):
    pred = []
    count = 0

    for i in range(x.shape[0]):
        prediction = predict(x[i])
        pred.append(prediction)
        if prediction == y[i]:
            count += 1

    return count / x.shape[0], pred

acc, ypred = accuracy(image_data, labels)

#acc, ypred = accuracy(test_data, test_labels)
print('Do chinh xac cua thuat toan doi voi TrainSet: ')
print(str(acc * 100) + '%')


svm_classifier = svm.SVC(kernel='linear', C=1.0)
svm_classifier.fit(image_data, labels)
ypred_sklearn = svm_classifier.predict(image_data)
#print(ypred_sklearn)
print('Do chinh xac cua thuat toan su dung ham cua Sklearn:')
print(str(svm_classifier.score(image_data,labels) * 100) + '%')

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


cnf_matrix = confusion_matrix(labels, ypred)
print(cnf_matrix)
plot_confusion_matrix(cnf_matrix, [0,1],normalize=False,title='Confusion matrix',cmap=plt.cm.Blues)

joblib.dump(svm_classifier, 'model_name.npy')

#cnf_matrix_sklearn = confusion_matrix(labels, ypred_sklearn)
#print(cnf_matrix_sklearn)
#plot_confusion_matrix(cnf_matrix_sklearn, [0,1],normalize=False,title='Confusion matrix',cmap=plt.cm.Blues)

print("Do chinh xac cua thuat toan: ")
acc = accuracy_score(result_test, ypred_sklearn)
print(acc)





