import tensorflow as tf
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import random

# loading our pre-trained model 
model = tf.keras.applications.InceptionV3()
model.summary()

# creating new_model to fine tune earlier model
base_model = tf.keras.Model(inputs = model.layers[0].input , outputs = model.layers[-2].output)
new_model = tf.keras.Sequential()
new_model.add(base_model)
new_model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
new_model.summary()

new_model.compile(loss=tf.keras.losses.BinaryCrossentropy() , optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4), metrics =["accuracy"])

img_array = []
categories = ['close eyes','open eyes']
for category in categories:
    path = os.path.join('input/train', category)
    label = categories.index(category)
    images = os.listdir(path)
    for img in images:
        img_array.append([img,label])

# training on random sample of 1000 elements
rand_img_array = random.sample(img_array,1000)
train =[]
for i in range(0,1000):
    if(rand_img_array[i][1] == 0):
        img = cv2.imread(os.path.join('input/train/close eyes',rand_img_array[i][0]))
    else:
        img = cv2.imread(os.path.join('input/train/open eyes',rand_img_array[i][0]))
    img = cv2.resize(img,(299,299),3)
    img = img/255.0
    train.append([img,rand_img_array[i][1]])
X = []
y = []

for features,labels in train:
    X.append(features)
    y.append(labels)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)
X_train = np.asarray(X_train)   
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)   
y_test = np.asarray(y_test)

new_model.fit(X_train,y_train, epochs = 1, validation_data=(X_test,y_test))


# training on another 1000 random images
img_array2 = [x for x in img_array if (x not in rand_img_array)]
rand_img_array2 = random.sample(img_array2,1000)
del(img_array)
del(rand_img_array)
train =[]
for i in range(0,1000):
    if(rand_img_array2[i][1] == 0):
        img = cv2.imread(os.path.join('input/train/close eyes',rand_img_array2[i][0]))
    else:
        img = cv2.imread(os.path.join('input/train/open eyes',rand_img_array2[i][0]))
    img = cv2.resize(img,(299,299),3)
    img = img/255.0
    train.append([img,rand_img_array2[i][1]])
X = []
y = []

for features,labels in train:
    X.append(features)
    y.append(labels)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1)
X_train = np.asarray(X_train)   
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)   
y_test = np.asarray(y_test)

new_model.fit(X_train,y_train, epochs = 1, validation_data=(X_test,y_test))

new_model.save("my_new_model.keras")