#!/usr/bin/env python
# coding: utf-8

# In[337]:


# Let`s import all packages that we may need:
from sklearn.model_selection import cross_val_predict, KFold, cross_val_score, train_test_split
import matplotlib.pyplot as plt 
import sys 
from scipy.stats import randint
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph. 
from sklearn.preprocessing import StandardScaler, MinMaxScaler # for normalization
from sklearn.pipeline import Pipeline # pipeline making
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics # for the check the error and accuracy of the model
from sklearn.metrics import mean_squared_error,r2_score
import keras
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import SGD 
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.layers import LSTM, Dropout, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingClassifier
import tensorflow as tf
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from matplotlib import pyplot as plt
from keras.optimizers import SGD
import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import itertools as it


# In[338]:


c1 = pd.read_csv('FD1.csv')
c2 = pd.read_csv('FD5.csv')
#rf_ = pd.read_csv('rf.csv')


# In[339]:


c_1 = c1.loc[c1['VWC_1'] * c1['VWC_2'] != 0]
c_2 = c2.loc[c2['VWC_1'] * c2['VWC_2'] != 0]


# In[340]:


def lookback(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    dff = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(dff.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(dff.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# In[341]:


import copy
c1_30 = copy.deepcopy(c_1)
c1_60 = copy.deepcopy(c_1)
c2_30 = copy.deepcopy(c_2)
c2_60 = copy.deepcopy(c_2)
#rf = copy.deepcopy(rf_)


# In[342]:


c1_30.drop(c_1.columns[[0,2,3,4]], axis=1, inplace=True)
c1_60.drop(c_2.columns[[0,1,2,4]], axis=1, inplace=True)
c2_30.drop(c_1.columns[[0,2,3,4]], axis=1, inplace=True)
c2_60.drop(c_2.columns[[0,1,2,4]], axis=1, inplace=True)
#rf.drop(rf_.columns[[0]], axis=1, inplace=True)


# In[343]:


values_1 = c1_30.values
values_2 = c1_60.values
values_3 = c2_30.values
values_4 = c2_60.values
#values_5 = rf.values
scaler = MinMaxScaler(feature_range=(0, 1))


# In[344]:


#nfuture = 30
lookbackperiod = 1
#forcasting = lookbackperiod+nfuture


# In[345]:


scaled1 = scaler.fit_transform(values_1)
scaled2 = scaler.fit_transform(values_2)
scaled3 = scaler.fit_transform(values_3)
scaled4 = scaler.fit_transform(values_4)
#scaled5 = scaler.fit_transform(values_5)


# In[346]:


#shape_0 = scaled1.shape[0]
#shape_01 = scaled5.shape[1]
#scaled_05 = scaled5[:shape_0,:]


# In[347]:


#scaled_5 = scaler.fit_transform(scaled_05)
#scaled_1 = scaler.fit_transform(scaled1)
#scaled = np.concatenate((scaled_1,scaled_5), axis=1)
reframed1 = lookback(scaled1, lookbackperiod, 1)
print(reframed1.head())


# In[348]:


#shape_00 = scaled3.shape[0]
#scaled_005 = scaled5[:shape_00,:]


# In[349]:


#scaled_50 = scaler.fit_transform(scaled_005)
#scaled_3 = scaler.fit_transform(scaled3)
#scaled = np.concatenate((scaled_3,scaled_50), axis=1)
reframed2 = lookback(scaled3, lookbackperiod, 1)
print(reframed2.head())


# In[350]:


#kf = KFold(n_splits=1) # Define the split - into 2 folds
#kf.get_n_splits(reframed)# returns the number of splitting iterations in the cross-validator
#print (kf.get_n_splits(reframed))
#print(kf)


# In[351]:


features1 = reframed1.values
features2 = reframed2.values
truth1 = values_2[lookbackperiod:]
truth2 = values_4[lookbackperiod:]


# In[352]:


#truthsize = truth.shape[0]


# In[353]:


#features = features[0:truthsize]


# In[354]:


features1.shape, truth1.shape, features2.shape, truth2.shape


# In[ ]:


cvscores = []
yhat_array = []
y_array = []
     
#train_features = features[train_index]
#test_features = features[test_index]
#train_truth = truth[train_index]
#test_truth = truth[test_index]

train_features = features1
test_features = features2
train_truth = truth1
test_truth = truth2

train_features, train_truth = train_features[:, :], train_truth[:,]
test_features, test_truth = test_features[:, :], test_truth[:,]
train_features = train_features.reshape((train_features.shape[0], 1, train_features.shape[1]))
test_features = test_features.reshape((test_features.shape[0], 1, test_features.shape[1]))
print(train_features.shape, train_truth.shape, test_features.shape, test_truth.shape)

model = Sequential()
model.add(LSTM(100, input_shape=(train_features.shape[1], train_features.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam', metrics=['mae'])
history = model.fit(train_features, train_truth, epochs=1, batch_size=36, verbose=2, shuffle=False)
scores = model.evaluate(test_features, test_truth, verbose=0)
ypred = model.predict(test_features)

yhat_array.append(ypred)
y_array.append(test_truth)

print (test_truth.shape)
print (ypred.shape)

size_1= len(ypred)
size_2= len(test_truth)
aa_1=[x for x in range(size_1)]
aa_2=[x for x in range(size_2)]
plt.plot(aa_2, test_truth, marker='.', label="Actual")
plt.plot(aa_1, ypred, 'r', label="Predicted")
plt.title('Predicted vs. Actual Soil Moisture Reading', size=15)
plt.ylabel('Soil Moisture Reading (% v/v)', size=15)
plt.xlabel('Time step (minutes)', size=15)
plt.legend(fontsize=15)
plt.show()

print((model.metrics_names[1], scores[1]))


# In[ ]:


new_y_array = np.concatenate(y_array, axis=0)


# In[ ]:


new_yhat_array = np.concatenate(yhat_array, axis=0)


# In[ ]:


size= new_yhat_array.shape[0]
aa=[x for x in range(size)]
plt.plot(aa, new_y_array, marker='', color='blue', label="Actual")
plt.plot(aa, new_yhat_array, label="Predicted", color= 'red')
plt.ylabel('Soil Moisture Reading (% v/v)', size=13)
plt.xlabel('Time step (minutes)', size=13)
plt.legend(loc='upper right', fontsize=13)
plt.show()


# In[ ]:




