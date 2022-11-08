import os
import numpy as np
import pandas as pd

# visualization

import matplotlib.pyplot as plt
import seaborn as sns

# preprocessing

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder   # for sex 
le = LabelEncoder()
from sklearn import model_selection
from sklearn import metrics

from sklearn.model_selection import cross_val_score

#importing 

df = pd.read_csv('C:/Users/vixky/Desktop/Project Θεωρια Αποφάσεων/Indian Liver Patient Dataset (ILPD).csv',sep=',',
names=['age','sex','tot_bilirubin','direct_bilirubin','tot_proteins','albumin','ag_ratio','sgpt','sgot','alkphos','class']
)

#print(df)
df.info()       # data in column 9 (Albumin_and_Globulin_Ratio) is missing 578/583
#df[df['alkphos'].isnull()]  # 4 rows with missing data (?)  idk how to refer to column 9 


# drop rows with missing data
#df = df.dropna()      steile mail 


# print(df.columns)
# print('*'*50)
# for i in df.columns :
#     print(i)
#     print(df[i].describe())
#     print('*'*50)


# Sex column into 0 and 1

label = le.fit_transform(df['sex'])
df.drop("sex", axis=1, inplace=True) #remove column from df 
df['sex'] = label                    #add column to df , pws na to valw sthn thesh 2 ?
print(df)

# check for imbalanced data

print('No. of patients with liver disease :',len(df[df['class']==1]))
print('No. of patients without liver disease :',len(df[df['class']==2]))

# correlation heatmap

def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(12, 12))
    colormap = sns.diverging_palette(154, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='black',
        annot_kws={'fontsize':11 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)

correlation_heatmap(df)
#plt.show()

# Splitting the data into train and test
X = df.iloc[:, :10]   # Features
y = df['class']       # target variable

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaled_values=scaler.fit_transform(X)
X.loc[:,:]=scaled_values

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)