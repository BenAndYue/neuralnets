import pandas as pd
import numpy as np

import matplotlib.pyplot as plt 
import cv2 as cv

from keras.layers import Conv2D, Input, LeakyReLU, Dense, Activation, Flatten, Dropout, MaxPool2D
from keras import models
from keras.optimizers import Adam,RMSprop 
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.models import model_from_json
import tensorflow as tf


np.random.seed(1) # seed
df_train = pd.read_csv("train.csv") # Loading Dataset


df_train = df_train.iloc[np.random.permutation(len(df_train))] # Random permutaion for dataset (seed is used to resample the same permutation every time)
# df_train.head(5)

sample_size = df_train.shape[0] # Training set size
validation_size = int(df_train.shape[0]*0.1) # Validation set size 

# train_x and train_y
train_x = np.asarray(df_train.iloc[:sample_size-validation_size,1:]).reshape([sample_size-validation_size,28,28,1]) # taking all columns expect column 0




train_y = np.asarray(df_train.iloc[:sample_size-validation_size,0]).reshape([sample_size-validation_size,1]) # taking column 0

# val_x and val_y
val_x = np.asarray(df_train.iloc[sample_size-validation_size:,1:]).reshape([validation_size,28,28,1])
val_y = np.asarray(df_train.iloc[sample_size-validation_size:,0]).reshape([validation_size,1])

# train_x.shape,train_y.shape
# loading test csv
df_test = pd.read_csv("test.csv")
test_x = np.asarray(df_test.iloc[:,:]).reshape([-1,28,28,1])


# normalize from pixel values in range [0,1]
train_x = train_x/255
val_x = val_x/255
test_x = test_x/255

# # plotting the first 30 images
# rows = 1 # defining no. of rows in figure
# cols = 1 # defining no. of colums in figure

# f = plt.figure(figsize=(2*cols,2*rows)) # defining a figure 
# for i in range(rows*cols): 
#     f.add_subplot(rows,cols,i+1) # adding sub plot to figure on each iteration
#     plt.imshow(train_x[i].reshape([28,28]),cmap="Blues") 
#     plt.axis("off")
#     print(train_y[i])
#     plt.title(str(train_y[i]), y=-0.15,color="black")
# plt.savefig("digits.png")


# # Cheacking frequency of digits in TRAINING and validation set
# counts = df_train.iloc[:sample_size-validation_size,:].groupby('label')['label'].count()
# # df_train.head(2)
# # counts
# f = plt.figure(figsize=(10,6))
# f.add_subplot(111)

# plt.bar(counts.index,counts.values,width = 0.8,color="orange")
# for i in counts.index:
#     plt.text(i,counts.values[i]+50,str(counts.values[i]),horizontalalignment='center',fontsize=14)

# plt.tick_params(labelsize = 14)
# plt.xticks(counts.index)
# plt.xlabel("Digits",fontsize=16)
# plt.ylabel("Frequency",fontsize=16)
# plt.title("Frequency Graph training set",fontsize=20)
# plt.savefig('digit_frequency_train.png')  
# plt.show()

# # df_train.iloc[sample_size-validation_index:,1:]
# # Cheacking frequency of digits in training and VALIDATION set
# counts = df_train.iloc[sample_size-validation_size:,:].groupby('label')['label'].count()
# # df_train.head(2)
# # counts
# f = plt.figure(figsize=(10,6))
# f.add_subplot(111)

# plt.bar(counts.index,counts.values,width = 0.8,color="orange")
# for i in counts.index:
#     plt.text(i,counts.values[i]+5,str(counts.values[i]),horizontalalignment='center',fontsize=14)

# plt.tick_params(labelsize = 14)
# plt.xticks(counts.index)
# plt.xlabel("Digits",fontsize=16)
# plt.ylabel("Frequency",fontsize=16)
# plt.title("Frequency Graph Validation set",fontsize=20)
# plt.savefig('digit_frequency_val.png')
# plt.show()

# printing the first image set to 1:1 to only print the first or selected first for testing
# plotting the first 30 images
rows = 1 # defining no. of rows in figure
cols = 1 # defining no. of colums in figure

f = plt.figure(figsize=(2*cols,2*rows)) # defining a figure 
for i in range(rows*cols): 
    f.add_subplot(rows,cols,i+1) # adding sub plot to figure on each iteration
    plt.imshow(train_x[i].reshape([28,28]),cmap="Blues") 
    plt.axis("off")
    plt.title(str(train_y[i]), y=-0.15,color="black")
plt.savefig("digits.png")

model = models.Sequential()
# Block 1
model.add(Conv2D(32,3, padding  ="same",input_shape=(28,28,1)))
model.add(LeakyReLU())
model.add(Conv2D(32,3, padding  ="same"))
model.add(LeakyReLU())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Block 2
model.add(Conv2D(64,3, padding  ="same"))
model.add(LeakyReLU())
model.add(Conv2D(64,3, padding  ="same"))
model.add(LeakyReLU())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation="sigmoid"))

initial_lr = 0.001
loss = "sparse_categorical_crossentropy"
model.compile(Adam(lr=initial_lr), loss=loss ,metrics=['accuracy'])
model.summary()

# training the model
epochs = 10
batch_size = 150
history_1 = model.fit(train_x,train_y,batch_size=batch_size,epochs=epochs)





# second try loading the model 
model.save('final_try.h5')

# loading model

# global model
modell = tf.keras.models.load_model('final_try.h5',compile=False)


# doesnt save correctly the model for a reason idk
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)



# how did the training go of the Keras ?

# Diffining Figure
f = plt.figure(figsize=(20,7))

#Adding Subplot 1 (For Accuracy)
f.add_subplot(121)

plt.plot(history_1.epoch,history_1.history['accuracy'],label = "accuracy") # Accuracy curve for training set
plt.plot(history_1.epoch,history_1.history['val_accuracy'],label = "val_accuracy") # Accuracy curve for validation set

plt.title("Accuracy Curve",fontsize=18)
plt.xlabel("Epochs",fontsize=15)
plt.ylabel("Accuracy",fontsize=15)
plt.grid(alpha=0.3)
plt.legend()

#Adding Subplot 1 (For Loss)
f.add_subplot(122)

plt.plot(history_1.epoch,history_1.history['loss'],label="loss") # Loss curve for training set
plt.plot(history_1.epoch,history_1.history['val_loss'],label="val_loss") # Loss curve for validation set

plt.title("Loss Curve",fontsize=18)
plt.xlabel("Epochs",fontsize=15)
plt.ylabel("Loss",fontsize=15)
plt.grid(alpha=0.3)
plt.legend()

plt.show()

# https://www.kaggle.com/tarunkr/digit-recognition-tutorial-cnn-99-67-accuracy#Buliding-Model

