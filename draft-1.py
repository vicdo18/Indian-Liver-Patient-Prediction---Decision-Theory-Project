import os
import numpy as np
import pandas as pd

# visualization

import matplotlib.pyplot as plt
import seaborn as sns

# preprocessing

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder   # for sex 

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

# print('********')
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
df['Gender'] = label                    #add column to df , pws na to valw sthn thesh 2 ?
# with pd.option_context('display.max_rows', 583, 'display.max_columns', 11):          # print all rows and columns
#     print(df)
print(df)

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


# Cross validate model with 5fold stratified cross val randomly splits the training set into (5_splits) 5 distinct subsets called folds, 
# then it trains and evaluates the models 5 times, picking a different fold for evaluation every time and 
# training on the other 4 folds.


cvscore = cross_val_score(nb, X, y, cv=5)
print("Cross-validated scores:", cvscore)
#print('Accuracy: %0.2f (+/- %0.2f)' % (cvscore.mean(), cvscore.std() * 2))



# Geometric_Mean = sqrt (Sensitivity * Specificity)
# print("Geometric Mean - ", metrics.geometric_mean_score(y_test, y_pred))