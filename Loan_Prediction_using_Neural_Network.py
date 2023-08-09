#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('tensorflow_version', '1.x')


# # Steps to build a Neural Network using Keras
# 
# <ol>1. Loading the dataset</ol>
# <ol>2. Creating training and validation set</ol>
# <ol>3. Defining the architecture of the model</ol>
# <ol>4. Compiling the model (defining loss function, optimizer)</ol>
# <ol>5. Training the model</ol>
# <ol>6. Evaluating model performance on training and validation set</ol>

# ## 1. Loading the dataset

# In[ ]:


# importing the required libraries
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# check version on sklearn
print('Version of sklearn:', sklearn.__version__)


# In[ ]:


# loading the pre-processed dataset
data = pd.read_csv('loan_prediction_data.csv')


# In[5]:


# looking at the first five rows of the dataset
data.head()


# In[6]:


# checking missing values
data.isnull().sum()


# In[7]:


# checking the data type
data.dtypes


# In[ ]:


# removing the loan_ID since these are just the unique values
data = data.drop('Loan_ID', axis=1)


# In[9]:


# looking at the shape of the data
data.shape


# In[ ]:


# separating the independent and dependent variables

# storing all the independent variables as X
X = data.drop('Loan_Status', axis=1)

# storing the dependent variable as y
y = data['Loan_Status']


# In[11]:


# shape of independent and dependent variables
X.shape, y.shape


# ## 2. Creating training and validation set

# In[ ]:


# Creating training and validation set

# stratify will make sure that the distribution of classes in train and validation set it similar
# random state to regenerate the same train and validation set
# test size 0.2 will keep 20% data in validation and remaining 80% in train set

X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=data['Loan_Status'],random_state=10,test_size=0.2)


# In[13]:


# shape of training and validation set
(X_train.shape, y_train.shape), (X_test.shape, y_test.shape)


# ## 3. Defining the architecture of the model

# In[14]:


# checking the version of keras
import keras
print(keras.__version__)


# In[15]:


# checking the version of tensorflow
import tensorflow as tf
print(tf.__version__)


# ### a. Create a model
# 
# <img src='https://drive.google.com/uc?id=1iZNZ3kwSHRNf-Irn3DZmMuBb6K-Lro7w'>

# In[ ]:


# importing the sequential model
from keras.models import Sequential


# ### b. Defining different layers
# 
# <img src='https://drive.google.com/uc?id=16X6De2hua1XJBe3dfmUUeGTgP6PbXEpc'>

# In[ ]:


# importing different layers from keras
from keras.layers import InputLayer, Dense 


# <img src='https://drive.google.com/uc?id=1tsy4B6G0UN4-J4L4roOdoWQiZMUdgw2a'>

# In[18]:


# number of input neurons
X_train.shapea


# In[19]:


# number of features in the data
X_train.shape[1]


# In[ ]:


# defining input neurons
input_neurons = X_train.shape[1]


# <img src='https://drive.google.com/uc?id=1xL_hM9rGItZjsZ8Lofwzw_9fZUi4bgJo'>

# In[ ]:


# number of output neurons

# since loan prediction is a binary classification problem, we will have single neuron in the output layer 


# In[ ]:


# define number of output neurons
output_neurons = 1


# In[ ]:


# number of hidden layers and hidden neurons

# It is a hyperparameter and we can pick the hidden layers and hidden neurons on our own


# In[ ]:


# define hidden layers and neuron in each layer
number_of_hidden_layers = 2
neuron_hidden_layer_1 = 10
neuron_hidden_layer_2 = 5


# In[ ]:


# activation function of different layers

# for now I have picked relu as an activation function for hidden layers, you can change it as well
# since it is a binary classification problem, I have used sigmoid activation function in the final layer


# In[26]:


# defining the architecture of the model
model = Sequential()
model.add(InputLayer(input_shape=(input_neurons,)))
model.add(Dense(units=neuron_hidden_layer_1, activation='relu'))
model.add(Dense(units=neuron_hidden_layer_2, activation='relu'))
model.add(Dense(units=output_neurons, activation='sigmoid'))


# In[27]:





# In[28]:


# number of parameters between input and first hidden layer

input_neurons*neuron_hidden_layer_1


# In[29]:


# number of parameters between input and first hidden layer

# adding the bias for each neuron of first hidden layer

input_neurons*neuron_hidden_layer_1 + 10


# In[30]:


# number of parameters between first and second hidden layer

neuron_hidden_layer_1*neuron_hidden_layer_2 + 5


# In[31]:


# number of parameters between second hidden and output layer

neuron_hidden_layer_2*output_neurons + 1


# ## 4. Compiling the model (defining loss function, optimizer)

# In[32]:


# compiling the model

# loss as binary_crossentropy, since we have binary classification problem
# defining the optimizer as adam
# Evaluation metric as accuracy

model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])


# ## 5. Training the model

# In[33]:


# training the model

# passing the independent and dependent features for training set for training the model

# validation data will be evaluated at the end of each epoch

# setting the epochs as 50

# storing the trained model in model_history variable which will be used to visualize the training process

model_history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50)


# ## 6. Evaluating model performance on validation set

# In[ ]:


# getting predictions for the validation set
prediction = model.predict_classes(X_test)


# In[35]:


# calculating the accuracy on validation set
accuracy_score(y_test, prediction)


# ### Visualizing the model performance

# In[36]:


# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[37]:


# summarize history for accuracy
plt.plot(model_history.history['acc'])
plt.plot(model_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[1]:


#functional API 


# In[ ]:


from keras import Input,Model


# In[ ]:


#Defining architecture of the model using functional API
x=Input(shape=(input_neurons,))
hidden1=Dense(units=neuron_hidden_layer_1,activation='relu')(x)
hidden2=Dense(units=neuron_hidden_layer_2,activation='relu')(hidden1)

Output=Dense(units=output_neurons,activation='sigmoid')(hidden2)

model_functional=Model(x,Output)


# In[ ]:


model_functional.summary()


# In[ ]:


#compiling the model
model_functional.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])


# In[ ]:


model_histry=model_functional.fit(X_train,y_train,Validation_data=(X_test,y_test),epochs=50)


# In[ ]:


predictions=model_functional.predict(X_test)


# In[ ]:


#convert it into 1 or 0 accounring to ur ceriteria
predictions =predictions.reshape(123,)


# In[ ]:


prediction_int=predictions>0.5


# In[ ]:


accuracy_score(y_test,prediction_int)

