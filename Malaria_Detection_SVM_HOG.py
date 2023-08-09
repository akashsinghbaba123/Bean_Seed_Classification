#!/usr/bin/env python
# coding: utf-8


#   Data Preparation
#   Method1: Pixel values as features
#   Model1: Model building using Pixel Features
#   Method 2: HOG Features
#  Model2: Model building using HOG Features

# # 1. Introduction to Problem Statement
# 
# Malaria is one of the deadliest diseases. It is a life-threatening disease caused by parasites that are transmitted to people through the bites of infected female Anopheles mosquitoes. It is preventable and curable.


# In[1]:


get_ipython().system('ls')


# The cell images folder contains all the images of the dataset and the file train.csv contain image names belonging to dataset and their corresponding labels i.e. Parasitized/Uninfected.
# Now lets see the names of images present in our dataset.

# In[3]:


get_ipython().system('ls cell_images/')


# In[1]:


#Storing the base directory 
import os

base_dir = os.path.join('./cell_images')


# # 3. Understanding Data Set

# First of all, let's set the base directory for reading images as all the images of the dataset are present in this directory.

# In[2]:


import numpy as np
import pandas as pd


# Now lets import the train.csv and look at its contents.

# In[3]:


train_df = pd.read_csv('train.csv')
train_df.shape


# In[4]:


train_df.head()


# In[5]:


#understand the distribution of both classes in the training data set
train_df['label'].value_counts()  


# From the above cell we can observe that our train set consists of equal samples of both the classes thus we will not face any problem due to class imbalance in the dataset.
# 

# In[7]:


import matplotlib.pyplot as plt
from skimage.io import imread, imshow

plt.figure(figsize=(15,15))
fig, ax = plt.subplots(nrows=2, ncols=2)

for i in range(2):
    for j in range(2):
        #Reading files after concatenating file name with folder path
        image = imread(os.path.join(base_dir,train_df["filename"][i+2*j]))
        #showing some images of the dataset
        ax[i,j].imshow(image)
        ax[i,j].set_ylabel(train_df["label"][i+2*j])


# # 4. Data Preparation

# 1. since we have textual labels for our images i.e. Parasitized/Uninfected so we will convert them to numerical labels i.e. 0/1

# In[18]:


from sklearn import preprocessing

#create the LabelEncoder object
le = preprocessing.LabelEncoder()


# Fit label Encoder
le.fit(train_df['label'])

#transform textual labels
labels = le.transform(train_df['label'])

# print('0 - ',le.inverse_transform(0))    
# print('1 - ',le.inverse_transform(1))     


# 2. Here we split the dataset into training and validation sets.
# Training set is the subset of the dataset that is used for training and validation set is used to evaluate the performance of the model after every epoch.

# In[19]:


#import required functions
from sklearn.model_selection import train_test_split

#divide the dataset into training and validation sets
train_files, val_files, train_labels, val_labels = train_test_split(train_df['filename'],
                                                                    labels, 
                                                                    test_size=0.2, random_state=42,shuffle=True)
#check the shapes of training and validation sets
print(train_files.shape, val_files.shape)
print(train_labels.shape,val_labels.shape)


# In[21]:


from skimage.io import imread, imshow
from skimage.transform import resize


# In[22]:


shapes = []
for i in train_df['filename']:
    image = imread(os.path.join(base_dir,i))
    shapes.append(image.shape)

print('Minimum Dimensions - ',np.min(shapes,axis=0))
print('Maximum Dimensions - ',np.max(shapes,axis=0))
print('Average Dimensions - ',np.mean(shapes,axis=0))  


# We can see that there are images of different shapes. It is recommended to have images in shape size before going ahead with modeling process and it is also dependent on which feature extractor tool, you are using. 

# # 5. Method 1: Pixel Values as Features

# Here, we will use pixel value as a feature to classify images in Parasitized/Uninfected images. Although, we have looked at that these images are of different shapes so we will bring all the images in same shape by resizing to 40X40. And, after that convert two dimensional matrix to one dimensional vector. So, for each images, we will have 1600 (40X40) features.

# In[24]:


IMG_DIMS=(40,40)
train_features_pixel=[]

for i in train_files:
    image = imread(os.path.join(base_dir,i))
    image = resize(image,IMG_DIMS)
    features = np.reshape(image,(IMG_DIMS[0]*IMG_DIMS[1]*3))
    train_features_pixel.append(features)

train_features_pixel = np.array(train_features_pixel)


# In[42]:


train_features_pixel.shape


# In[27]:


IMG_DIMS=(40,40)
val_features_pixel=[]

for i in val_files:
    image = imread(os.path.join(base_dir,i))
    image = resize(image,IMG_DIMS)
    features = np.reshape(image,(IMG_DIMS[0]*IMG_DIMS[1]*3))
    val_features_pixel.append(features)

val_features_pixel = np.array(val_features_pixel)


# In[29]:


val_features_pixel.shape


# # 6. Model1: Model building using Pixel Features

# ## Logistic Regression

# In[25]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#training the Logistic model
clf_lr_pixel = LogisticRegression()
clf_lr_pixel.fit(train_features_pixel,train_labels)


# In[46]:


preditions_train = clf_lr_pixel.predict(train_features_pixel)
print("Training: Model Accuracy - ",accuracy_score(train_labels,preditions_train)*100,'%')

predictions_val = clf_lr_pixel.predict(val_features_pixel)
print("Validation: Model Accuracy - ",accuracy_score(predictions_val,val_labels)*100,'%')


# ## Linear SVM

# In[47]:


from sklearn.svm import LinearSVC

#training the Logistic model
clf_svc_pixel = LinearSVC(random_state=102)
clf_svc_pixel.fit(train_features_pixel,train_labels)


# In[48]:


preditions_train = clf_svc_pixel.predict(train_features_pixel)
print("Training: Model Accuracy - ",accuracy_score(train_labels,preditions_train)*100,'%')

predictions_val = clf_svc_pixel.predict(val_features_pixel)
print("Validation: Model Accuracy - ",accuracy_score(predictions_val,val_labels)*100,'%')


# # 7. Method 2: HOG Features

# Your image size should be in 64X128(Width X Height) shape to extract HOG features from images. We will first change image shape size to 64X128 first and then extract HOG features.

# In[49]:


from skimage.feature import hog


# In[50]:


#Showing example of one image first
index= np.random.randint(0,1000)

image = imread(os.path.join(base_dir,train_files.iloc[index]))
IMG_DIMS = (128,64) # SkIMAGE takes input in HEIGHT X WIDTH format
image1 = resize(image,IMG_DIMS)
#calculating HOG features
features, hog_image = hog(image1, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=True, multichannel=True)


# In[51]:


#Original Image
imshow(image)


# In[52]:


#After Resize
imshow(image1)


# In[53]:


#Image with HOG Image (Look At the Edges)
imshow(hog_image)


# In[54]:


#Highlighting the HOG image using Matplotlib
import matplotlib.pyplot as plt
plt.imshow(hog_image, cmap="gray")


# In[55]:


#Shape of HOG Feature Vector
features.shape


# ## Calculate HOG features for both training and Validation images

# In[56]:


IMG_DIMS = (128,64)

#For Training Images

train_features_hog = []
for i in train_files:
    image = imread(os.path.join(base_dir,i))
    image = resize(image,IMG_DIMS)
    #calculating HOG features
    features, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=True, multichannel=True)
    hog_features = np.reshape(features,(features.shape[0]))
    train_features_hog.append(hog_features)

train_features_hog = np.array(train_features_hog)
  
#For Validation Images

val_features_hog = []
for i in val_files:
    image = imread(os.path.join(base_dir,i))
    image = resize(image,IMG_DIMS)
    #calculating HOG features
    features, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=True, multichannel=True)
    hog_features = np.reshape(features,(features.shape[0]))
    val_features_hog.append(hog_features)

val_features_hog = np.array(val_features_hog)

#checking the shape of the final lists after reading all the images
train_features_hog.shape, val_features_hog.shape


# # 8. Model2: Model building using HOG Features

# ## Logistic Regression

# In[58]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#training the Logistic model
clf_lr_hog = LogisticRegression()
clf_lr_hog.fit(train_features_hog,train_labels)


# In[59]:


preditions_train = clf_lr_hog.predict(train_features_hog)
print("Training: Model Accuracy - ",accuracy_score(train_labels,preditions_train)*100,'%')

predictions_val = clf_lr_hog.predict(val_features_hog)
print("Validation: Model Accuracy - ",accuracy_score(predictions_val,val_labels)*100,'%')


# ## Linear SVM

# In[60]:


from sklearn.svm import LinearSVC

#training the Logistic model
clf_svc_hog = LinearSVC()
clf_svc_hog.fit(train_features_hog,train_labels)


# In[61]:


preditions_train = clf_svc_hog.predict(train_features_hog)
print("Training: Model Accuracy - ",accuracy_score(train_labels,preditions_train)*100,'%')

predictions_val = clf_svc_hog.predict(val_features_hog)
print("Validation: Model Accuracy - ",accuracy_score(predictions_val,val_labels)*100,'%')


# In[4]:


#AKASH SINGH 

#Storing the base directory 
import os

base_dir = os.path.join('./cell_images')


# In[5]:


import pandas as pd 
import numpy as np 


# In[6]:


train_df=pd.read_csv('train.csv')


# In[7]:


train_df.head()


# In[10]:


from skimage.io import imshow,imread,imsave
import matplotlib.pyplot as plt 


# In[12]:


plt.figure(figsize=(15,15))
fig,ax=plt.subplots(nrows=3,ncols=3)

for i in range(3):
    
    for j in range(3):
        image=imread(os.path.join(base_dir,train_df['filename'][i+2*j]))
        
        ax[i][j].imshow(image)
        ax[i][j].set_ylabel(train_df['label'][i+2*j])


# In[13]:


#preprocessing
from sklearn.preprocessing import LabelEncoder


# In[14]:


le=LabelEncoder()


# In[15]:


labels=le.fit_transform(train_df['label'])


# In[17]:


from sklearn.model_selection import train_test_split,KFold


# In[18]:


x_train,x_test,y_train,y_test=train_test_split(train_df['filename'],labels,random_state=42,test_size=0.2,shuffle=True)


# In[19]:


x_train.shape,x_test.shape,y_train.shape,y_test.shape


# In[20]:


from skimage.io import imread,imshow
from skimage.transform import resize


# In[23]:


shapes=[]
for i in train_df['filename']:
    image=imread(os.path.join(base_dir,i))
    shapes.append(image.shape)
    
print("min shape",np.min(shapes,axis=0))
print("max shape",np.max(shapes,axis=0))
print("avg shape",np.mean(shapes,axis=0))


# In[24]:


#pixels as features 
IMG_RES=(40,40)
train_features=[]

for i in x_train:
    image=imread(os.path.join(base_dir,i))
    image=resize(image,IMG_RES)
    features=np.reshape(image,IMG_RES[0]*IMG_RES[1]*3)
    
    train_features.append(features)
    
train_features=np.array(train_features)
    


# In[25]:


test_features=[]
for i in x_test:
    image=imread(os.path.join(base_dir,i))
    image=resize(image,IMG_RES)
    features=np.reshape(image,IMG_RES[0]*IMG_RES[1]*3)
    
    test_features.append(features)
    
test_features=np.array(test_features)


# In[26]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[27]:


lr=LogisticRegression()


# In[29]:


lr.fit(train_features,y_train)


# In[30]:


predicttrain=lr.predict(train_features)
predicttest=lr.predict(test_features)


# In[32]:


score1=accuracy_score(y_train,predicttrain)
score2=accuracy_score(y_test,predicttest)


# In[33]:


print(" train score: ",score1)
print(" test score: ",score2)


# In[35]:


#linear svc
from sklearn.svm import LinearSVC


# In[37]:


svc=LinearSVC(random_state=102)


# In[38]:


svc.fit(train_features,y_train)


# In[39]:


predict_train=svc.predict(train_features)
predict_test=svc.predict(test_features)


# In[40]:


score1=accuracy_score(y_train,predict_train)
score2=accuracy_score(y_test,predict_test)


# In[41]:


print(score1)                                  
print(score2)                                                                         


# In[56]:


#hog features 
index=np.random.randint(1,10000)
HOG_RES=(128,64)


# In[57]:


from skimage.feature import hog


# In[58]:


image=imread(os.path.join(base_dir,train_df['filename'].iloc[index]))
image1=resize(image,HOG_RES)


# In[59]:


features, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=True, multichannel=True)


# In[61]:


imshow(image)


# In[62]:


imshow(image1)


# In[63]:


imshow(hog_image)


# In[68]:


#hog features for x_train and x_test
train_hog_features=[]
HOG_RES=(128,64)
for i in x_train:
    image=imread(os.path.join(base_dir,i))
    image1=resize(image,HOG_RES)
    features,hog_image=hog(image1,orientations=9,pixels_per_cell=(8,8),
                          cells_per_block=(2,2),visualize=True,multichannel=True)
    features_hog=np.reshape(features,(features.shape[0]))
    
    train_hog_features.append(features_hog)
train_hog_features=np.array(train_hog_features)


# In[69]:


test_hog_features=[]
HOG_RES=(128,64)
for i in x_test:
    image=imread(os.path.join(base_dir,i))
    image1=resize(image,HOG_RES)
    features,hog_image=hog(image1,orientations=9,pixels_per_cell=(8,8),
                          cells_per_block=(2,2),visualize=True,multichannel=True)
    features_hog=np.reshape(features,(features.shape[0]))
    
    test_hog_features.append(features_hog)
test_hog_features=np.array(test_hog_features)


# In[70]:


from sklearn.linear_model import LogisticRegression


# In[71]:


lr=LogisticRegression()


# In[72]:


lr.fit(train_hog_features,y_train)


# In[73]:


predicttrain=lr.predict(train_hog_features)
predicttest=lr.predict(test_hog_features)
score1=accuracy_score(y_train,predicttrain)
score2=accuracy_score(y_test,predicttest)


# In[74]:


score1 ,score2


# In[75]:


from sklearn.svm import LinearSVC


# In[77]:


svc=LinearSVC(random_state=102)


# In[78]:


svc.fit(train_hog_features,y_train)


# In[79]:


predicttrain=svc.predict(train_hog_features)
precittest=svc.predict(test_hog_features)


# In[80]:


score1=accuracy_score(y_train,predicttrain)
score2=accuracy_score(y_test,predicttest)
score1, score2


# In[ ]:




