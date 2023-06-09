# -*- coding: utf-8 -*-
"""Final Model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15HZwJUYUUg4jh5DN5R2PLUa_F1bceAIF

#Image Classification with SIFT/Hog/Daisy Feature and Neural Networks
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog, daisy, SIFT
from PIL import Image
from skimage import exposure

#Image Classification with SIFT/Hog/Daisy Feature and Neural Networks

"""## 1. Dataset Preparation

mount google drive
"""

from google.colab import drive
drive.mount('/content/drive')

"""Extract data"""

!unzip /content/drive/MyDrive/Al.zip

"""## Load Image Data"""

image_dir = '/content/Al/face-detection/images/gender/'

# Get all .jpg & .png images in the male folder
male_filenames = [os.path.join(image_dir, 'male', filename) for filename in os.listdir(image_dir + 'male') if filename.endswith('.jpg') or filename.endswith('.png')]

# Get all .jpg & .png images in the female folder
female_filenames = [os.path.join(image_dir, 'female', filename) for filename in os.listdir(image_dir + 'female') if filename.endswith('.jpg') or filename.endswith('.png')]

# Get the images using the file names
male_images = [Image.open(filename) for filename in male_filenames]
female_images = [Image.open(filename) for filename in female_filenames]

# make labels
female_labels = [1 for i in range(len(female_images))]
male_labels = [0 for i in range(len(male_images))]

"""show image"""

plt.imshow(male_images[1], cmap='gray')

plt.imshow(female_images[1], cmap='gray')

"""## Test extract feature from image

### HOG Feature
"""

feature, hog_img = hog(male_images[0], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=True, channel_axis=-1)

feature.shape

plt.bar(list(range(feature.shape[0])), feature)

feature, hog_img = hog(female_images[0], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=True, channel_axis=-1)

feature.shape

plt.bar(list(range(feature.shape[0])), feature)

"""### DAISY Feature"""

# Extract the DAISY descriptor
feature, daisy_image = daisy(male_images[0].convert('L'), step=180, radius=58, rings=2, histograms=6, orientations=8, visualize=True)

feature = np.array(feature).flatten()
feature.shape

plt.bar(list(range(feature.shape[0])), feature)

feature, daisy_img = daisy(female_images[0].convert('L'), step=180, radius=58, rings=2, histograms=6, orientations=8, visualize=True)

feature = np.array(feature).flatten()
feature.shape

plt.bar(list(range(feature.shape[0])), feature)

"""## feature extraction algorithm to be selected"""

# feature extraction algorithm to be selected
feature_type = input("Which Feature tool to use? \n 1-SIFT \n 2-SURF \n 3-ORB \n Enter as a number: ")

def get_descriptors_extractor(image):
  image = image.resize((128, 128))
  if feature_type=='1':
    # Extract the HOG descriptor
    feature, hog_img = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=True, channel_axis=-1)
    return feature

  elif feature_type=='2':
    # Extract the DAISY descriptor
    feature, daisy_img = daisy(image.convert('L'), step=180, radius=58, rings=2, histograms=6, orientations=8, visualize=True)
    return np.array(feature).flatten()

  elif feature_type=='3':
    # Extract the SIFT descriptor
    descriptor_extractor = SIFT()

    # Extract the SIFT descriptor from the image
    descriptor_extractor.detect_and_extract(male_images[0].convert('L'))
    image_keypoints   = descriptor_extractor.keypoints
    # select just 20 keypoint from image since very large
    image_descriptor = descriptor_extractor.descriptors[:20, :]
    feature = np.array(image_descriptors).flatten()
    return feature

"""
## 2. Preprocessing Feature Extraction"""

# merge images
images = male_images + female_images
labels = male_labels + female_labels

feature = get_descriptors_extractor(images[0])
n_dims = feature.shape[0]

n_samples = len(images)
n_samples

"""

Create variable for dataset"""

from sklearn import datasets

X, y = datasets.make_classification(n_samples=n_samples, n_features=n_dims)

X.shape

"""**Get extracted feature from each image & put into dataset variable**

Time consuming process
"""

from tqdm import tqdm
for i in tqdm(range(n_samples)):
  X[i] = get_descriptors_extractor(images[i])
  y[i] = labels[i]

"""## Train-Test split"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

print(f'Train shape: {X_train.shape}')
print(f'Test shape: {X_test.shape}')

"""# Train Models"""

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA

from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from xgboost import plot_tree, plot_importance
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, precision_recall_curve
from mlxtend.plotting import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import KFold

from sklearn.feature_selection import RFE

import warnings
warnings.filterwarnings("ignore")

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, auc

def train(clf, X_train, y_train):
    clf.fit(X_train, y_train)
    return clf

def test(model_name, clf, X_train, y_train, X_test, y_test):
    y_pred = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)
    
    cm_test = confusion_matrix(y_pred, y_test)
    
    print(f'Accuracy for training set for {model_name} = {accuracy_score(y_train, y_pred_train)}\n')
    print(f'Accuracy for test set for {model_name} = {accuracy_score(y_test, y_pred)}')
    print(f'Precision for test set for {model_name} = {precision_score(y_test, y_pred)}')
    print(f'Recall for test set for {model_name} = {recall_score(y_test, y_pred)}')
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(False)
#     plt.plot(auc(y_test, y_pred))
    
    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(clf, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
        plt.grid(False)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

models = {
    'GaussianNB': GaussianNB(),
    'BernoulliNB': BernoulliNB(),
    'LinearDiscriminantAnalysis':LinearDiscriminantAnalysis(),
    'LogisticRegression': LogisticRegression(),
    'RandomForestClassifier': RandomForestClassifier(),
    'SupportVectorMachine': SVC(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'BaggingClassifier': BaggingClassifier(),
    'ExtraTreesClassifier': ExtraTreesClassifier(),
    'XGBClassifier': XGBClassifier(learning_rate= 1, max_depth= 5, n_estimators= 10),
    'Stochastic Gradient Descent':  SGDClassifier(max_iter=5000, random_state=0),
    'Neural Nets': MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5000, 10), random_state=1),
}

modelNames = ["GaussianNB", 'BernoulliNB', 'LinearDiscriminantAnalysis','LogisticRegression','RandomForestClassifier','SupportVectorMachine',
             'DecisionTreeClassifier', 'KNeighborsClassifier','GradientBoostingClassifier', 'AdaBoostClassifier', 'BaggingClassifier','XGBClassifier',
              'ExtraTreesClassifier',
             'Stochastic Gradient Descent', 'Neural Nets']

trainScores = []
validationScores = []
testScores = []

for m in models:
    model = models[m]
    model.fit(X_train, y_train)
#     score = model.score(X_valid, y_valid)

    print(f'\n{m}\n') 
    train_score = model.score(X_train, y_train)
    print(f'Train score of trained model: {train_score*100}')
    trainScores.append(train_score*100)

#     validation_score = model.score(X_valid, y_valid)
#     print(f'Validation score of trained model: {validation_score*100}')
#     validationScores.append(validation_score*100)

    test_score = model.score(X_test, y_test)
    print(f'Test score of trained model: {test_score*100}')
    testScores.append(test_score*100)
    print(" ")

    y_predictions = model.predict(X_test)
    cm = confusion_matrix(y_predictions, y_test)

    tn = cm[0,0]
    fp = cm[0,1]
    tp = cm[1,1]
    fn = cm[1,0]
    accuracy  = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp)
    recall    = tp / (tp + fn)
    f1score  = 2 * precision * recall / (precision + recall)
    specificity = tn / (tn + fp)
    print(f'Accuracy : {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall   : {recall}')
    print(f'F1 score : {f1score}')
    print(f'Specificity : {specificity}')
    print("") 
    print(f'Classification Report: \n{classification_report(y_predictions, y_test)}\n')
    print("")
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_predictions)
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    
    # Plot non-normalized confusion matrix
    fig, ax = plot_confusion_matrix(conf_mat=cm,
                                show_absolute=True,
                                show_normed=True,
                                colorbar=True)
    plt.show()

    for m in range (1):
        current = modelNames[m]
        modelNames.remove(modelNames[m])

    preds = model.predict(X_test)
    confusion_matr = confusion_matrix(y_test, preds) #normalize = 'true'
    print("############################################################################")
    print("")
    print("")
    print("")

modelNames

modelNames = ["GaussianNB", 'BernoulliNB', 'LinearDiscriminantAnalysis','LogisticRegression','RandomForestClassifier','SupportVectorMachine',
             'DecisionTreeClassifier', 'KNeighborsClassifier','GradientBoostingClassifier', 'AdaBoostClassifier', 'BaggingClassifier','XGBClassifier',
              'ExtraTreesClassifier',
             'Stochastic Gradient Descent', 'Neural Nets']

for i in range(len(modelNames)):
    print(f'Accuracy of {modelNames[i]} -----> {testScores[i]}')
     
     
import pandas as pd
data = {'Model': modelNames, 'Accuracy': testScores}  
result = pd.DataFrame(data)
result

# function to add value labels
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], round(y[i], 3), ha = 'center',
                 bbox = dict(facecolor = 'gold', alpha =.9), weight='bold')

Names = ["GNB", 'BNB', 'LDA','LR','RF','SVM',
             'DT', 'KNN','GBC', 'AdaB', 'BC','XGB',
              'ETC', 'SGD', 'ANN']
fig = plt.figure(figsize=(15,6))
plt.title("Classifier Comprison",fontweight='bold')
plt.ylabel("Accuracy", fontweight="bold")
plt.xticks(weight="bold")
plt.yticks(weight="bold")
plt.bar(Names, testScores, color=['darkcyan', 'darkred', 'darkgreen', 'darkblue', 'darkcyan', 'violet', 'cyan', 'red', 'green', 'blue', 'purple'])

plt.grid(axis='y')
addlabels(Names, testScores)
plt.legend()

"""### Select Best Model and Save"""

best_model = result.Model.iloc[np.argmax(result.Accuracy)]
print(f'Best model is: {best_model}')
model = models[best_model]
model.fit(X_train, y_train)

import pickle

filename = "/content/drive/MyDrive/Model_Image/final_model.pickle"

# save model
pickle.dump(model, open(filename, "wb"))

# load model
loaded_model = pickle.load(open(filename, "rb"))

test_path = '9.jpg'
from PIL import Image

image = Image.open(test_path)
# Resize image
image = image.resize((128, 128))
feature = get_descriptors_extractor(image).reshape(1, -1)

y_pred = loaded_model.predict(feature)
if y_pred==0:
  print('Male')
else:
  print('Female')

