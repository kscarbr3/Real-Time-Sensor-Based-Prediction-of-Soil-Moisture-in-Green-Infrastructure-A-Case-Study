#!/usr/bin/env python
# coding: utf-8

# In[229]:


# Let`s import all packages that we may need:
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt 
import sys 
import numpy as np # linear algebra
from scipy.stats import randint
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph. 
from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn.preprocessing import StandardScaler # for normalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline # pipeline making
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics # for the check the error and accuracy of the model
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import KFold
## for Deep-learing:
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import SGD 
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import itertools
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout
# MLP for Pima Indians Dataset with 10-fold cross validation via sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingClassifier
# MLP with manual validation set
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import KFold
import numpy as np

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import  numpy as np
import itertools as it


# In[230]:


c1 = pd.read_csv('FD3.csv')
c2 = pd.read_csv('FD3.csv')
#rf_ = pd.read_csv('rf.csv')


# In[231]:


c_1 = c1.loc[c1['VWC_1'] * c1['VWC_2'] != 0]
c_2 = c2.loc[c2['VWC_1'] * c2['VWC_2'] != 0]


# In[232]:


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


# In[233]:


import copy
c1_30 = copy.deepcopy(c_1)
c2_30 = copy.deepcopy(c_2)
#rf = copy.deepcopy(rf_)

c1_30.drop(c_1.columns[[0,2,3,4]], axis=1, inplace=True)
c2_30.drop(c_2.columns[[0,2,3,4]], axis=1, inplace=True)
#rf.drop(rf_.columns[[0]], axis=1, inplace=True)


# In[234]:


values_1 = c1_30.values
values_2 = c2_30.values
#values_3 = rf.values
scaler = MinMaxScaler(feature_range=(0, 1))


# In[235]:


#nfuture = 120
lookbackperiod = 1
#forcasting = lookbackperiod+nfuture


# In[236]:


scaled_1 = scaler.fit_transform(values_1)
#scaled3 = scaler.fit_transform(values_3)


# In[237]:


#shape_0 = scaled_1.shape[0]
#shape_1 = scaled3.shape[1]
#scaled_03 = scaled3[:shape_0,:]


# In[238]:


#scaled3 = scaler.fit_transform(scaled_03)
scaled1 = scaler.fit_transform(scaled_1)
#scaled = np.concatenate((scaled1,scaled3), axis=1)
reframed = lookback(scaled1, lookbackperiod, 1)
print(reframed.head())


# In[239]:


kf = KFold(n_splits=10) # Define the split - into 2 folds
kf.get_n_splits(reframed)# returns the number of splitting iterations in the cross-validator
print (kf.get_n_splits(reframed))
print(kf)


# In[240]:


features = reframed.values
truth = values_2[lookbackperiod:]


# In[241]:


truthsize = truth.shape[0]


# In[242]:


features = features[0:truthsize]


# In[243]:


features.shape, truth.shape


# In[244]:


cvscores = []
yhat_array = []
y_array = []
for train_index, test_index in kf.split(features):
    print("TRAIN:", train_index, "TEST:", test_index)
     
    train_features = features[train_index]
    test_features = features[test_index]
    train_truth = truth[train_index]
    test_truth = truth[test_index]
 
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
    history = model.fit(train_features, train_truth, epochs=1, batch_size=36, verbose=2, shuffle=False, validation_split=0.1)
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
     
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
     
print("%.2f%% (± %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
 
 


# In[245]:


new_y_array = np.concatenate(y_array, axis=0)


# In[246]:


new_yhat_array = np.concatenate(yhat_array, axis=0)


# In[247]:


size= new_yhat_array.shape[0]
aa=[x for x in range(size)]
plt.plot(aa, new_y_array, marker='', color='blue', label="Actual")
plt.plot(aa, new_yhat_array, label="Predicted", color= 'red')
plt.ylabel('Soil Moisture Reading (% v/v)', size=13)
plt.xlabel('Time step (minutes)', size=13)
plt.legend(loc='upper right', fontsize=13)
plt.show()


# In[ ]:




