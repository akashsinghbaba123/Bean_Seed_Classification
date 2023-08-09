#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the necessary library
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import math
import datetime as dt 
import warnings 
warnings.filterwarnings('ignore')


# In[2]:


#load our NYC TAXI TRIP duration dataset
data=pd.read_csv("C:/Users/AKASH/Desktop/nyc_taxi_trip_duration.csv")


# In[3]:


#Looking at top 5 rows of dataset
data.head()


# In[4]:


#finding the null values in the dataset
data.isna().sum()


# In[5]:


#data type in the dataset 
data.dtypes


# In[6]:


# Converting the pickup_datetime and dropoff_datetime to datetime datatype

data['pickup_datetime']=pd.to_datetime(data.pickup_datetime)
data['dropoff_datetime']=pd.to_datetime(data.dropoff_datetime)

# Starting pickup date and ending pickup date
data['pickup_datetime'].max(),data['pickup_datetime'].min()


# In[7]:


# Feature engineering to add some datetime features 
data['pickup_dayofweek']=data['pickup_datetime'].dt.dayofweek
data['week_of_year'] = data['pickup_datetime'].apply(lambda x: x.isocalendar()[1])
data['pickup_hour']=data['pickup_datetime'].dt.hour
data['pickup_minute']=data['pickup_datetime'].dt.minute
data['distance']=((data['dropoff_latitude']-data['pickup_latitude'])**2+(data['dropoff_longitude']-data['pickup_longitude'])**2)**0.5
data['is_holiday'] = data['pickup_datetime'].dt.date.isin([pd.to_datetime('2016-01-01').date(), pd.to_datetime('2016-01-18').date(), pd.to_datetime('2016-02-15').date(), pd.to_datetime('2016-05-30').date()]).astype(int)


# In[8]:


#Calculating the haversine distance array
def haversine_array(lat1,log1,lat2,log2):
    """
    Calculate the haversine distance between two points on the Earth's surface
    with given latitude and longitude coordinates.
    """
    
    lat1,log1,lat2,log2=map(np.radians,(lat1,log1,lat2,log2))
    earth_radius=6371 #Km
    lat=lat2-lat1
    log=log2-log1
    # Calculate the haversine of half the differences in latitude and longitude
    d=np.sin(lat*0.5)**2+np.cos(lat1)*np.cos(lat2)*np.sin(log*0.5)**2
    # Calculate the great-circle distance (in kilometers)
    h=earth_radius*np.arcsin(np.sqrt(d))
    
    return h
    
    
    


# In[9]:


#Calculating the direction between two points using haversine formaula

def direction_array(lat1,log1,lat2,log2):
    earth_radius=6371#Km
    log_delta=np.radians(log2-log1)
    
    lat1,log1,lat2,log2=map(np.radians,(lat1,log1,lat2,log2))
    
    # y represents the north-south component, and x represents the east-west component.
    
    y=np.sin(log_delta)*np.cos(lat2)
    x=np.cos(lat1)*np.sin(lat2)-np.sin(lat1)*np.cos(lat2)*np.cos(log_delta)
    
    return np.degrees(np.arctan2(y,x))
    


# In[10]:


# New feature of haversin distance
data['haversine_distance']=haversine_array(data['pickup_latitude'].values,data['pickup_longitude'].values,data['dropoff_latitude'].values,data['dropoff_longitude'].values) 


# In[11]:


#new feature of direction array

data['direction']=direction_array(data['pickup_latitude'].values,data['pickup_longitude'].values,data['dropoff_latitude'].values,data['dropoff_longitude'].values)


# In[12]:


data.head()


# In[13]:


#Open Source Routing Machine or OSRM for each trip in our original dataset. 
#This will give us a very good estimate of distances between pickup and dropoff Points

fr1 = pd.read_csv('NYC_modeling/osrm/fastest_routes_train_part_1.zip',
                  usecols=['id', 'total_distance', 'total_travel_time'])
fr2 = pd.read_csv('NYC_modeling/osrm/fastest_routes_train_part_2.zip',
                  usecols=['id', 'total_distance', 'total_travel_time'])

data_street = pd.concat((fr1, fr2))
data = data.merge(data_street, how='left', on='id')
data_street.head()


# In[14]:


#binning the latitude and longitude as it might contains some noisy data
data['pickup_latitude']=np.round(data['pickup_latitude'],3)
data['pickup_longitude']=np.round(data['pickup_longitude'],3)
data['dropoff_latitude']=np.round(data['dropoff_latitude'],3)
data['dropoff_longitude']=np.round(data['dropoff_longitude'],3)

#on hot encoding the vendor_id
data['vendor_id']=data['vendor_id']-1

data['vendor_id']


# In[15]:


#we are taking log to normalize the data
data['trip_duration']=np.log(data['trip_duration'].values+1)
sns.distplot(data['trip_duration'],kde=False,bins=200)
plt.show()
data.isna().sum()


# In[16]:


#filling the two null value in total_distance and total_travel_time
data.fillna(0,inplace=True)


# In[17]:


# taking out the dependent variable 'trip_duration'
y=data['trip_duration']
# taking out the independent variable 
x=data.drop(['id', 'pickup_datetime', 'dropoff_datetime', 'trip_duration','store_and_fwd_flag'],axis=1)


# In[18]:


# columns name and shape of new dataset
data.columns,data.shape


# In[19]:


#checking the shape of independent and dependent variable 
x.shape,y.shape


# In[20]:


x.head()


# In[21]:


#Importing the machine learning algorithm for model generation 
from sklearn.model_selection import train_test_split as  TTS
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import Ridge ,Lasso 


# In[22]:


#Import module to scale data
from sklearn.preprocessing import StandardScaler 
ss=StandardScaler()
#scaling the independent variables for better model
scaled=ss.fit_transform(x)
x=pd.DataFrame(scaled,columns=x.columns)


# In[23]:


# spiliting x and y (independet and dependent variable) into  train and test 
x_train,x_test,y_train,y_test=TTS(x,y,test_size=1/3,random_state=42)


# In[24]:


# Creating instance of Linear Regresssion
lr=LR()
#fiting the model
lr.fit(x_train,y_train)


# In[25]:


#predicting on test data set
pred=lr.predict(x_test)


# In[26]:


#Importing the evaluation metrics 
from sklearn.metrics import mean_squared_error as mse,r2_score

#Evaluating the error on predicted values by r2_score 
score=r2_score(y_test,pred)
score


# In the above score of the model , we can see 0.524 is score of the model which is correct predicted

# In[27]:


#Importing Ridge and Lasso linear model to improve the model 
#Importing  KFold

from sklearn.linear_model import Ridge,Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error,r2_score
from math import sqrt

# Set the different values of alpha to be tested
alpha_values = [0, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20, 25]

# Set the number of folds for cross-validation
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)


# Perform grid search over different alpha values
for alpha in alpha_values:
    r2_scores = []
    for train_index, test_index in kf.split(x,y):
        x_train_fold, x_test_fold = x.loc[train_index], x.loc[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        model = Ridge(alpha=alpha)
        model.fit(x_train_fold, y_train_fold)
        y_pred = model.predict(x_test_fold)
        r2_s =r2_score(y_test_fold,y_pred)
        r2_scores.append(r2_s)
    avg_r2 = np.mean(r2_scores)
    print(f"alpha: {alpha}, aveg r2_score: {avg_r2}")



# In the above r2_score, we can see the alpha and kfold together do not improve the model performance

# In[28]:


#We have define a function which takes input linear model instance,independent variable,dependent variable
# It return the r2_score for the given model 

def model_score(ml_model,x,y,rstate = 11):
    i = 1
    r2_scores = [] 
    x=x
    y=y
    #Creating instance of KFold() and puting n_splits=5 
    kf = KFold(n_splits=5,random_state=rstate,shuffle=True)
    for train_index,test_index in kf.split(x,y):
        print('\n{} of kfold {}'.format(i,kf.n_splits))
        x_train,x_test = x.loc[train_index],x.loc[test_index]
        y_train,y_test = y[train_index],y[test_index]

        model = ml_model
        model.fit(x_train, y_train)
        pred_test = model.predict(x_test)
      
        r2_score1 =r2_score(y_test, pred_test)
        sufix = ""
        msg = ""
        
        msg += "Valid r2_score: {:.5f}".format(r2_score1)
        print("{}".format(msg))
        # Save scores
        r2_scores.append(r2_score1)
        i+=1
    return r2_scores


# In[29]:


#Calling function model_score and passing linear regression instance,independent and dependent variable

lr=model_score(LR(),x,y)


# In the above score we can see the model started improves slightly more then previous models 

# In[30]:


#Importing the decision tree regressor
from sklearn.tree import DecisionTreeRegressor

#Creating instance of decisiontreeregressor
#putting hyperparameter 

dtr=DecisionTreeRegressor(random_state=12,max_depth=15,min_samples_leaf=25, min_samples_split=25)


# In[31]:


#Calling the function model_score()
dtr1=model_score(dtr,x,y)


# # Well!   We can see decision tree regressor perform best and it improve our model and the r2_score goes upto approx " 0.73 "

# In[32]:


#Importing the K-nn and creating the instance of it 
from sklearn.neighbors import KNeighborsRegressor as KNN

#creating the instance of KNN 
knn=KNN(n_neighbors=10)


# In[33]:


#fititng the model
knn.fit(x_train,y_train)


# In[34]:


#predicting on the test dataset
predict =knn.predict(x_test)   


# In[35]:


#checking the r2_score
score=r2_score(y_test,predict)


# In[36]:


score 


# We can see that knn is not performing well on this data so we will take DecisionTreeRegressor

# In[37]:





# In[ ]:




