from sklearn import svm
from tensorflow.keras.applications.vgg16 import VGG16
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from pathlib import Path

model = VGG16(weights = 'imagenet', include_top = False)
vgg16_feature_list = []
p = Path("TRAINSET/")
p2 = Path("TestSet/")
dirs = p.glob("*")
labels_dict = {'key': 0, 'card': 1, 'keychain': 2, 'led': 3}

labels = []
test_data = []
test_labels = []
listImages = []
for test_path in p2.glob("*.jpg"):
    test = image.load_img(test_path, target_size=(224, 224))
    test_array = image.img_to_array(test)
    test_data.append(test_array)

for folder_dir in dirs:
    label = str(folder_dir).split("\\")[-1][:-1]


    for img_path in folder_dir.glob("*.jpg"):
        img = image.load_img(img_path, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        vgg16_feature = model.predict(img_data)
        listImages.append(vgg16_feature)
        labels.append(labels_dict[label])

labels = np.array(labels)

#
# vgg16_feature = model.predict(img_data)
print(listImages.shape)
# vgg16_feature_np = np.array(vgg16_feature)
# print(vgg16_feature_np.shape)
#vgg16_feature_list.append(vgg16_feature_np.flatten())
#vgg16_feature_list.append(vgg16_feature_np)


listImages = np.array(listImages, dtype='float32')/255.0
M = listImages.shape[0]
listImages = listImages.reshape(M,-1)
print(listImages.shape)
print(labels.shape)


vgg16_feature_list_np = np.array(vgg16_feature_list)
#print(vgg16_feature_list_np.shape)
svm_classifier = svm.SVC(kernel='linear', probability=True)
svm_classifier.fit(listImages, labels)

ypred_sklearn1 = svm_classifier.predict(test_data)
ypred_sklearn = svm_classifier.predict_proba(test_data)
print(ypred_sklearn1)
print(ypred_sklearn)






