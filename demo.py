from sklearn import svm
from keras.applications.vgg16 import VGG16
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from pathlib import Path

# Add Model VGG 16:
model = VGG16(weights = 'imagenet', include_top = False)
vgg16_feature_list = []
# Define Path:
p = Path("Dataset/")
p2 = Path("TestSet/")
dirs = p.glob("*")

# Define Label
labels_dict = {'key': 0, 'pen': 1, 'keychain': 2, 'led': 3}

labels = []
test_data = []
test_labels = []
listImages = []

#Predict Test
# for test_path in p2.glob("*.jpg"):
#     test = image.load_img(test_path, target_size=(224, 224))
#     test_array = image.img_to_array(test)
#     test_data.append(test_array)
# test_data = np.array(test_data, dtype='float32')/255.0
# print(test_data.shape)
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
print(test_data.shape)
# Training:
for folder_dir in dirs:
    label = str(folder_dir).split("\\")[-1][:-1]


    for img_path in folder_dir.glob("*.jpg"):
        # Using target size: 224 x 224: Follow the model VGG16
        img = image.load_img(img_path, target_size=(224, 224))
        img_data = image.img_to_array(img)
        # Add another Dimension for array
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        # Add feature to img
        vgg16_feature = model.predict(img_data)
        listImages.append(vgg16_feature)
        labels.append(labels_dict[label])

labels = np.array(labels)
listImages = np.array(listImages)
#print(listImages)
        #
        # vgg16_feature = model.predict(img_data)
print(listImages.shape)
        # vgg16_feature_np = np.array(vgg16_feature)
        # print(vgg16_feature_np.shape)
        #vgg16_feature_list.append(vgg16_feature_np.flatten())
        #vgg16_feature_list.append(vgg16_feature_np)


listImages = np.array(listImages, dtype='float32')/255.0
# Reshape:
M = listImages.shape[0]
listImages = listImages.reshape(M,-1)
print(listImages.shape)
print(labels.shape)


vgg16_feature_list_np = np.array(vgg16_feature_list)
#print(vgg16_feature_list_np.shape)
svm_classifier = svm.SVC(kernel='linear', probability=True)
svm_classifier.fit(listImages, labels)


N = test_data.shape[0]
test_data = test_data.reshape(N,-1)
print (test_data.shape)
ypred_sklearn1 = svm_classifier.predict(test_data)
ypred_sklearn = svm_classifier.predict_proba(test_data)
print(ypred_sklearn1)
print(ypred_sklearn)






