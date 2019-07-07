import numpy as np # linear algebra
import json
from matplotlib import pyplot as plt
from skimage import color
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score
from sklearn import preprocessing
from subprocess import check_output
import random

# Input data files has to be available in the same folder
# outfilePos.txt contains hog features of positive images containing pedestrain
hogFile = open('outfilePos.txt', 'r') 

flag = 0
count = 0

hog_features = []
featVec = []
labels = []
for line in hogFile:
    line = line.strip()
    hist = line.split(" ")
    for elem in hist:
        featVec.append(float(elem))
    hog_features.append(featVec)
    count += 1
    labels.append(1)
    featVec = []

hogFile.close()
print("Number of Files:"+str(count))

# outfileNeg.txt contains hog features of negative images containing no-pedestrain
hogFile = open('outfileNeg.txt', 'r')

featVec = []
for line in hogFile:
    if(count == 0):
        break
    line = line.strip()
    hist = line.split(" ")
    for elem in hist:
        featVec.append(float(elem))
    hog_features.append(featVec)
    labels.append(0)
    featVec = []
    count -= 1
    
hogFile.close()

labels = np.array(labels).reshape(len(labels),1)

#Using Linear SVM
clf = svm.LinearSVC()
hog_features = np.array(hog_features)
data_frame = np.hstack((hog_features,labels))
np.random.shuffle(data_frame)

#percentage of data used for training
percentage = 80
partition = int(len(hog_features)*percentage/100)

x_train, x_test = data_frame[:partition,:-1],  data_frame[partition:,:-1]
y_train, y_test = data_frame[:partition,-1:].ravel() , data_frame[partition:,-1:].ravel()

x_train = np.nan_to_num(x_train)
X_scaled = preprocessing.scale(x_train)
clf.fit(X_scaled,y_train)

weightFile = open('svmweights.txt', 'w+')
l = clf.coef_[0]
string = " "
for i in range(len(l)):
    string = string +" "+str(l[i])
string += " "+str(clf.intercept_)[1:-1]
weightFile.write(string)
weightFile.close()

x_test = np.nan_to_num(x_test)
y_pred = clf.predict(x_test)

print("Accuracy: "+str(accuracy_score(y_test, y_pred)))
print('\n')
print(classification_report(y_test, y_pred))
