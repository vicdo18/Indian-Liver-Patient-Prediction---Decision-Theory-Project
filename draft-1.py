import os
import numpy as np
import pandas as pd

# visualization

import matplotlib.pyplot as plt
import seaborn as sns

# preprocessing

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder   # for gender 

from sklearn import model_selection
from sklearn import metrics

from sklearn.model_selection import cross_val_score

# Importing the data from the csv file.


df = pd.read_csv('C:/Users/vixky/Desktop/Project Θεωρια Αποφάσεων/Indian Liver Patient Dataset (ILPD).csv',sep=',',
names=['Age','Gender','Tb','Db','Alkphos','Sgpt','Sgot','Tp','Alb','Ag_ratio','Class']
)

#print(df)
#df.info()       # data in column 9 (alkphos) is missing 578/583

#print(df[df['Ag_ratio'].isnull()])                            # 4 rows with missing data in alkphos column
print('Mean of Ag_ratio before filling:', df['Ag_ratio'].mean())   # check mean before filling
df['Ag_ratio'].fillna(df['Ag_ratio'].mean(), inplace=True)  # fill missing data with mean


# print('How many missing values?',df['Ag_ratio'].isnull().sum())       # 0 missing values after filling with mean

# print(df.columns)
# print('*'*50)
# for i in df.columns :
#     print(i)
#     print(df[i].describe())
#     print('*'*50)


# Gender column into 0 and 1

le = LabelEncoder()        #sos male->0 female->1
label = 1-le.fit_transform(df['Gender'])      #fixed (1-le.. to invert values)
df['Gender'] = label                    # add new column with label encoded values
# with pd.option_context('display.max_rows', 583, 'display.max_columns', 11):          # print all rows and columns
#     print(df)
#print(df)

# check for imbalanced data

print('No. of patients with liver disease :',len(df[df['Class']==1]))
print('No. of patients without liver disease :',len(df[df['Class']==2]))

# sns.countplot(x='Class',data=df,palette='hls')
# plt.show()

X = df.iloc[:,0:10]   # Features (all columns except Class and gender)
y = df['Class']       # target variable

from sklearn.preprocessing import MinMaxScaler 

#Normalization [-1,1]

scaler=MinMaxScaler(feature_range=(-1,1))
scaled_values=df.drop(['Gender','Class'],axis=1)

scaled_values=scaler.fit_transform(scaled_values)
scaled_values=pd.DataFrame(scaled_values,columns=['Age','Tb','Db','Alkphos','Sgpt','Sgot','Tp','Alb','Ag_ratio'])

scaled_values.insert(1,'Gender',df['Gender'])
scaled_values['Class']=df['Class']
print("After scaling the data:")
print(scaled_values)

# Splitting the data into train and test sets

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)


# heatmap 

# plt.figure(figsize=(10,10))
# sns.heatmap(df.corr(),annot=True,fmt='.1f')
# plt.show()

# Naive Bayes

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

score = nb.score(X_test, y_test)
print("Overall score of the model is - ",score)
print("Model Report Card - ")
print(metrics.classification_report(y_test, y_pred, digits=3))
print("Accuracy score - ", metrics.accuracy_score(y_test,y_pred))


# 3o erwtima

# Cross validate model with 5fold stratified cross val randomly splits the training set into (5_splits) 5 distinct subsets called folds, 
# then it trains and evaluates the models 5 times, picking a different fold for evaluation every time and 
# training on the other 4 folds.

#k fold cross validation           , ---- email 

from sklearn import model_selection
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score , precision_score , accuracy_score , f1_score , roc_auc_score , confusion_matrix

vc_results = cross_validate(nb, X_train, y_train, cv=5, scoring=('accuracy', 'precision', 'recall', 'f1', 'roc_auc'))
print("!!Accuracy after validation: %0.2f (+/- %0.2f)" % (vc_results['test_accuracy'].mean(), vc_results['test_accuracy'].std() * 2))

#plot kfold progress

from sklearn.model_selection import learning_curve
train_sizes, train_scores, test_scores = learning_curve(nb, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 50))
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.5, 1.0])
plt.show()

# kfold = model_selection.KFold(n_splits=5, random_state=42, shuffle=True)
# results = model_selection.cross_val_score(nb, X_train, y_train, cv=kfold)
# print(" !!!Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# plt results from kfold

# plt.figure(figsize=(10,10))
# plt.plot(vc_results)
# plt.show()




# cvscore = cross_val_score(nb, X, y, cv=5)
# print("Cross-validated scores:", cvscore)
# #print('Accuracy: %0.2f (+/- %0.2f)' % (cvscore.mean(), cvscore.std() * 2))

#plot for the 5 folds

# plt.plot(score)
# plt.title('5-fold cross validation')
# plt.xlabel('Fold')
# plt.ylabel('Accuracy')
# plt.show()

#accuracy after cross validation
# print("Accuracy after cross validation: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))

# geometric mean metric

# from sklearn.metrics import make_scorer
# from sklearn.model_selection import cross_validate

# def geometric_mean(y_true, y_pred):
#     return metrics.fbeta_score(y_true, y_pred, beta=1, average='weighted')


# scoring = {'acc': 'accuracy', 
#               'prec_macro': 'precision_macro',
#                 'rec_micro': 'recall_macro',
#                 'f1_weighted': make_scorer(geometric_mean)}


#print sensitivity and specificity

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#plot confusion matrix 

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(nb, X_test, y_test)
plt.show()


# print('The geometric mean is {}'.format(geometric_mean(    # almost 51% accurate
#     y_test,
#     y_pred)))


# erwthma 4 
# SVM with Radial Basis Function (RBF) kernel 

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# SVM RBF training with kfold 

svm = SVC(kernel='rbf',  C=1.0 , gamma='scale')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

score = svm.score(X_test, y_test)
print("Overall score of the model is - ",score)
print("Model Report Card - ")
print(metrics.classification_report(y_test, y_pred, digits=3, zero_division=0)) #zero division warning - fixed
print("Accuracy score - ", metrics.accuracy_score(y_test,y_pred))

#plot confusion matrix for SVM RBF

plot_confusion_matrix(svm, X_test, y_test)
plt.show()

#geometric mean for SVM RBF 


def geometric_mean(y_true, y_pred):
    return metrics.fbeta_score(y_true, y_pred, beta=1, average='weighted')

print("The geometric mean  (SVC) is {}".format(geometric_mean(    
    y_test,
    y_pred)))

#print gamma 

print("Gamma is - ", svm.gamma)






