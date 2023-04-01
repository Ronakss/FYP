import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from PIL import Image
from skimage import exposure
import pickle
import PIL


#Image Classification with HOG Feature and Neural Networks

## 1. Dataset Preparation



## Load Image Data"""

image_dir = 'images/'

# Get all .jpg & .png images in the male folder
male_filenames = [os.path.join(image_dir, 'gender/male', filename) for filename in os.listdir(image_dir + 'gender/male') if filename.endswith('.jpg') or filename.endswith('.png')]

# Get all .jpg & .png images in the female folder
female_filenames = [os.path.join(image_dir, 'gender/female', filename) for filename in os.listdir(image_dir + 'gender/female') if filename.endswith('.jpg') or filename.endswith('.png')]

# Get the images using the file names
male_images = [Image.open(filename) for filename in male_filenames]
female_images = [Image.open(filename) for filename in female_filenames]

# make labels
female_labels = [0 for i in range(len(female_images))]
male_labels = [1 for i in range(len(male_images))]

"""show image"""

plt.imshow(male_images[1], cmap='gray')

plt.imshow(female_images[1], cmap='gray')

"""## Test extract HOG feature from image"""

## calculates the Histogram of Oriented Gradients (HOG) features for an image using the hog() 
#function with some specific parameters. The output of the function includes an 
#array of HOG features and a visualization of the HOG features overlaid on the input image.

feature, hog_img = hog(male_images[0], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=True, channel_axis=-1)

feature.shape

plt.bar(list(range(feature.shape[0])), feature)

#calculates the Histogram of Oriented Gradients (HOG) features for an image using the hog()
  #function with some specific parameters. The output of the function includes an array of
  # HOG features and a visualization of the HOG features overlaid on the input image.

feature, hog_img = hog(female_images[0], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=True, channel_axis=-1)

feature.shape

plt.bar(list(range(feature.shape[0])), feature)

"""
## 2. Preprocessing using HOG Feature Extraction"""

# merge images labed female and male images 
images = male_images + female_images
labels = male_labels + female_labels

n_dims = feature.shape[0]
n_dims

n_samples = len(images)
n_samples

"""

Create variable for dataset"""

from sklearn import datasets

#The code generates a synthetic dataset for classification tasks using the make_classification() 
#function from the sklearn.datasets module. The output is two arrays:
# X containing feature values and y containing class labels for each sample.

X, y = datasets.make_classification(n_samples=n_samples, n_features=n_dims)

X.shape

#HOG) features for each image in a dataset and storing them in 
#X, while also assigning the corresponding labels to y. The progress of the loop is displayed using the tqdm library.

"""


**Get HOG feature from each image & put into dataset variable**"""

from tqdm import tqdm
for i in tqdm(range(n_samples)):
    X[i], _ = hog(images[i], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=True, channel_axis=-1)
    y[i] = labels[i]

"""## Train-Test split"""

from sklearn.model_selection import train_test_split

#code is splitting the dataset into training and testing sets using the train_test_split() function from the sklearn.model_selection module.
#The train_test_split() function is being called with four parameters:
#X: the array of feature values for each sample.
#y: the array of class labels for each sample.
#test_size=0.2: the proportion of the dataset to include in the testing set. In this case, 20% of the samples will be used for testing.
#shuffle=True: whether to shuffle the samples before splitting them into training and testing sets.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

print(f'Train shape: {X_train.shape}')
print(f'Test shape: {X_test.shape}')

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

#These functions and classes can be used to train, evaluate and analyze the performance of a classification model.

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, auc

# code defines a function named train which takes a classifier object (clf) SVC  along with the training data 
#(X_train and y_train) and trains the classifier using the fit() method. The function returns the trained classifier object.

def train(clf, X_train, y_train):
    clf.fit(X_train, y_train)
    return clf

#The function first uses the trained classifier object to predict 
#the class labels for the test set and the training set. It then computes the confusion matrix,
# accuracy score, precision score, and recall score for the test set using the predicted and actual class labels

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

modelNames = ["GaussianNB", 'BernoulliNB', 'LinearDiscriminantAnalysis','LogisticRegression','RandomForestClassifier','SupportVectorMachine',
             'DecisionTreeClassifier', 'KNeighborsClassifier','GradientBoostingClassifier', 'AdaBoostClassifier', 'BaggingClassifier','XGBClassifier',
              'ExtraTreesClassifier',
             'Stochastic Gradient Descent', 'Neural Nets']

for i in range(len(modelNames)):
    print(f'Accuracy of {modelNames[i]} -----> {testScores[i]}')
     
     
import pandas as pd
data = {'Model': modelNames, 'Accuracy': testScores}  
result = pd.DataFrame(data)


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
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import hog

filename = "/Users/ronakshakari/Desktop/project/model/hog_model.pickle"

# load model
loaded_model = pickle.load(open(filename, "rb"))

test_path = '21.jpg'
image = Image.open(test_path)
plt.imshow(image, cmap='gray')

# Resize image
image = image.resize((128, 128))
feature, hog_img = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=True, channel_axis=-1)

feature = np.array(feature).flatten().reshape(1, -1)
y_pred = loaded_model.predict(feature)

if y_pred==1:
    print('Male')
else:
    print('Female')
