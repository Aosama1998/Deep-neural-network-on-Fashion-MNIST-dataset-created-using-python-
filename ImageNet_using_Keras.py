#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


my_data = keras.my_datasets.fashion_mnist


# In[3]:


(train_images, train_labels), (test_images, test_labels) = my_data.load_my_data()


# In[4]:


Data_Types = ['T-shirt', 'Trousers', 'Pullovers', 'Dress', 'Coats',
               'Sandals', 'Shirts', 'Sneakers', 'Bags', 'Boots']


# In[5]:


train_images = train_images / 255.0
test_images = test_images / 255.0


# In[6]:
# for displaying results only 

plt.figure(figsize=(10,10))
for i in range(20):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(Data_Types[train_labels[i]])
plt.show()


# In[7]:
#defining the modes of our models and how we want it to operate

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation="softmax")
])


# In[8]:

#displaying information of our model after initializnig and setting it up to start running
model.summary()


# In[9]:
# Compiling the model

model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])


# In[10]:

model.fit(train_images, train_labels, epochs=10)
