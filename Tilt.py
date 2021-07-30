#!/usr/bin/env python
# coding: utf-8

# # Head Pose Estimation - Tilt

# In[1]:


import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE
import IPython.display as display
from PIL import Image
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Convolution2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
tf.keras.optimizers.RMSprop
get_ipython().run_line_magic('matplotlib', 'inline')
import os
tf.__version__


# In[2]:


#upload and unzip data file
import zipfile
with zipfile.ZipFile('./modified_data.zip', 'r') as zip_ref:
 zip_ref.extractall('./')


# In[3]:


#Reading training data
train = pd.read_csv("./train_data.csv")
#understand shape of dataframe
train.head()


# ## Understand Data and Preprocessing
# Checking shape of training data and creating a plot to analyse Bias in data

# In[4]:


train.shape


# In[5]:


#grouping
shape_data_group= train.groupby(["tilt"])
shape_data_group_plot= shape_data_group.count()


# In[6]:


#plotting bar chart
shape_data_group_plot.plot(kind='bar',color=('darkgreen'))
shape_data_group.count()


# In[7]:


train['tilt'] = train['tilt'].astype(str)


# In[8]:


#Splitting Data into TRAIN, TEST AND VALIDATION
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(train,test_size = 0.2)
training_data, validation_data = train_test_split(train_data,test_size = 0.2)

# Generate two data frames for training and validation #
print('Train size: {}, Test size: {}'.format(training_data.shape[0], validation_data.shape[0] ) ) 
N_train_images = training_data.shape[0]
N_val_images = validation_data.shape[0]


# In[9]:


training_data.shape


# In[10]:


validation_data.shape


# Inspecting the Data images in a random order

# In[11]:


from PIL import Image
import glob
image_list = []
for filepath in glob.glob('modified_data/*.jpg', recursive=True): #assuming gif
    headpose = filepath.split("/")[1]
    image_list.append((filepath, headpose))
headpose_data = pd.DataFrame(data=image_list, columns=['image_path', 'filename'])


# In[12]:


headpose_data.head()


# In[13]:


import cv2
import os

for filepath in glob.glob('modified_data/*.jpg', recursive=True):
    
    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(filepath), gray)


# In[14]:


r_inx = np.random.choice(100, 4)
rand_data = headpose_data.loc[r_inx,'image_path']

plt.figure(figsize=(16,4))
for i, image_path in enumerate(rand_data):
    im = np.asarray(Image.open(image_path))
    plt.subplot(1,4,i+1)
    plt.imshow(im,cmap='gray')
    plt.axis('off')
    
plt.show()


# ## Loading Function

# we are using rmsprop and categorical_crossentropy snce there are more than two categories in the output variable.The dataset is complicated hence we will be loading them in batches, and batch size is chosen as 16. The loading function converts channels into 3-channel images. Data normalization is done to bring it in pixel [0-1] value

# In[15]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator 

train_datagen = ImageDataGenerator(rescale=1./255, data_format='channels_last')
val_datagen = ImageDataGenerator(rescale=1./255, data_format='channels_last')

batch_size = 16

train_generator = train_datagen.flow_from_dataframe(
    dataframe=training_data,
    directory='./modified_data', 
    x_col="filename", 
    y_col="tilt", 
    target_size=(32, 32), 
    batch_size=batch_size, 
    class_mode='categorical')

validation_generator = val_datagen.flow_from_dataframe(
    dataframe=validation_data,
    directory='./modified_data', 
    x_col="filename", 
    y_col="tilt", 
    target_size=(32, 32), 
    batch_size=batch_size, 
    class_mode='categorical')

test_generator = val_datagen.flow_from_dataframe(
        dataframe=test_data,
        directory='./modified_data',
        x_col="filename",
        y_col="tilt",
        target_size=(32, 32),
        shuffle = False,
        batch_size=batch_size,
        class_mode='categorical')


# ## Training model 

# In[16]:


#function to train a model train the model
def train_model(model_, num_epoch=100, verbose=False):
    res = []
    for e in range(num_epoch):
        # print('Epoch', e)
        batches = 0

        loss_ = []
        acc_ = []

         # iterate over each batch
        for x,y in train_generator:
            loss, acc = model_.train_on_batch(x, y) # Update weights and return train loss, acc per batch
            loss_.append(loss)
            acc_.append(acc)
            batches += 1
            if batches >= N_train_images / batch_size:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break
        loss_ = np.mean(loss_)
        acc_ = np.mean(acc_)

        loss, acc = calculate_losses(model_, validation_generator, N_val_images, batch_size)
        if verbose:
            print("Training epoch {}: Loss = {}, Accuracy = {}".format(e, loss_, acc_))
            print("Validation epoch {}: Loss = {}, Accuracy = {}".format(e, loss, acc))

        res.append((e, loss_, acc_, loss, acc))
    return np.asarray(res)


# ## Accuracy and Loss Function

# In[17]:


def calculate_losses(model_, data_generator_, N_images, batch_size_): 
    loss_hold = []
    acc_hold = []
    batches = 0
    for x,y in data_generator_:
        loss,acc = model_.evaluate(x, y, verbose=0) 
        loss_hold.append(loss)
        acc_hold.append(acc)
        batches += 1
        if batches >= N_images / batch_size_:
            # we need to break the loop by hand because 
            # the generator loops indefinitely
            break
        
    return np.mean(loss_hold), np.mean(acc_hold)


# In[18]:


N_train_images = 1488
N_val_images = 372


# ## Model 1
# We have 32x32 features and nine classes, so the input layer must have 1024 units, and the output layer must have 9 units. We only have to define the hidden layers. We're only going to have one hidden layer for this project, and we'll give it 64 neurons.

# In[19]:


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Lambda
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Input
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.metrics import categorical_accuracy

# Input layer
input_ = Input(shape=(32, 32, 3)) # This is the input shape
input_slice = Lambda(lambda x: tf.expand_dims(x[:,:,:,0], -1, name=None))(input_)
x = Flatten()(input_slice)  # This will convert the 28x28 input to a vector of  dimension 784

# Hidden layer
h = Dense(64)(x)
h = Activation('sigmoid')(h)

# Output layer
out_ = Dense(9)(h)
out_ = Activation('softmax')(out_)

# Define model
model1 = Model(inputs=input_, outputs=out_)

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model1.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=[categorical_accuracy])


model1.summary()


# In[20]:


#training the model
res = train_model(model1, num_epoch= 75, verbose=1)


# In[21]:


#plot the results and plot the error loss for the training and accuracy data
def plot_results(res):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(res[:,0], res[:,1], 'r-')
    plt.plot(res[:,0], res[:,3], 'b-')
    plt.legend(['Train', 'Validation'])
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.ylim([0, np.max([5., np.max(res[:,1]), np.max(res[:,3])])])

    plt.subplot(1,2,2)
    plt.plot(res[:,0], res[:,2], 'r-')
    plt.plot(res[:,0], res[:,4], 'b-')
    plt.legend(['Train', 'Validation'])
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, np.max([1., np.max(res[:,2]), np.max(res[:,4])])])


# In[22]:


plot_results(res)


# In[23]:


#confusion matrix 
#Note, this code is taken straight from the SKLEARN website
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual class')
    plt.xlabel('Predicted class')


from collections import Counter
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import itertools

predictions = model1.predict_generator(test_generator,verbose=0)
print("Prediction shape is", predictions.shape)

predictedLabels =[]
for i in range(0,len(predictions[:,0])):
    predictedLabels.append(np.argmax(predictions[i]))

predictedLabels = np.asarray(predictedLabels)

true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

report = metrics.classification_report(true_classes, predictedLabels, target_names=class_labels)
print(report)
confusion_mtx = confusion_matrix(true_classes, predictedLabels)
plot_confusion_matrix(confusion_mtx, classes = range(5))


# ## Using Regularization 
# we are using categorical_crossentropy since the output variable contains nine classes.

# In[24]:


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Lambda
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Input
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.metrics import categorical_accuracy

# Input layer
input_ = Input(shape=(32, 32, 3)) # This is the input shape
input_slice = Lambda(lambda x: tf.expand_dims(x[:,:,:,0], -1, name=None))(input_)
x = Flatten()(input_slice)  # This will convert the 28x28 input to a vector of  dimension 784

# Hidden layer
# reg for weight matrix 1
h = Dense(64,kernel_regularizer=regularizers.l2(0.01))(x)
h = Activation('relu')(h)

# Output layer
# reg for weight matrix 2
out_ = Dense(9,kernel_regularizer=regularizers.l2(0.01))(h)
out_ = Activation('softmax')(out_)

# Define model
model_orig1 = Model(inputs=input_, outputs=out_)

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model_orig1.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=[categorical_accuracy])


model_orig1.summary()
# Creating a model for feature vizualization 
hidden_features = Model(inputs=input_, outputs=h)


# In[25]:


#training the model1
res = train_model(model_orig1, num_epoch=50, verbose=1)


# In[26]:


#plot the results and plot the error loss for the training and accuracy data
def plot_results(res):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(res[:,0], res[:,1], 'r-')
    plt.plot(res[:,0], res[:,3], 'b-')
    plt.legend(['Train', 'Validation'])
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.ylim([0, np.max([5., np.max(res[:,1]), np.max(res[:,3])])])

    plt.subplot(1,2,2)
    plt.plot(res[:,0], res[:,2], 'r-')
    plt.plot(res[:,0], res[:,4], 'b-')
    plt.legend(['Train', 'Validation'])
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, np.max([1., np.max(res[:,2]), np.max(res[:,4])])])


# In[27]:


plot_results(res)


# In[28]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Lambda, Input
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras import regularizers, optimizers

model_cnn = Sequential()

model_cnn.add(Conv2D(64, (5, 5), input_shape = (32,32,1)))
model_cnn.add(Activation('relu'))

model_cnn.add(Conv2D(64, (5, 5)))
model_cnn.add(Activation('relu'))

model_cnn.add(Flatten())  
model_cnn.add(Dense(128))
model_cnn.add(Activation('relu'))
model_cnn.add(Dropout(0.3))

model_cnn.add(Dense(9))
model_cnn.add(Activation('softmax'))


# ## Model 3
#  Using the final layer as Multi layer perceptron.

# In[29]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Lambda, Input,BatchNormalization, LeakyReLU
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras import regularizers, optimizers

Model3 = Sequential()

# input
Model3.add(Input(shape=(32, 32, 3)))

# Model1.add(Lambda(lambda x: tf.expand_dims(x[:,:,:,0], -1, name=None))) 
# this is a workaround. Dataloader automatically read one channel image as 3 channel 
#and we use Lambda layer to revert this back. Lambda layer can be used for operation 
#that does not involve trainianble weights

# Conv Layer 1
Model3.add(Conv2D(32, (3, 3)))
Model3.add(Activation('relu'))

# Conv Layer 2
Model3.add(Conv2D(64, (3, 3)))
Model3.add(Activation('relu'))
Model3.add(MaxPooling2D(pool_size=(2, 2)))

# MLP
Model3.add(Flatten())
Model3.add(Dense(64))
Model3.add(Activation('relu'))
Model3.add(Dense(9))
Model3.add(Activation('softmax'))

Model3.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=[categorical_accuracy])

Model3.summary()

res = train_model(Model3, num_epoch= 75, verbose=1)


# In[30]:


#plot the results and plot the error loss for the training and accuracy data
def plot_results(res):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(res[:,0], res[:,1], 'r-')
    plt.plot(res[:,0], res[:,3], 'b-')
    plt.legend(['Train', 'Validation'])
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.ylim([0, np.max([5., np.max(res[:,1]), np.max(res[:,3])])])

    plt.subplot(1,2,2)
    plt.plot(res[:,0], res[:,2], 'r-')
    plt.plot(res[:,0], res[:,4], 'b-')
    plt.legend(['Train', 'Validation'])
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, np.max([1., np.max(res[:,2]), np.max(res[:,4])])])


# In[31]:


plot_results(res)


# In[32]:


#confusion matrix 
#Note, this code is taken straight from the SKLEARN website
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual class')
    plt.xlabel('Predicted class')


from collections import Counter
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import itertools

predictions = Model3.predict_generator(test_generator,verbose=0)
print("Prediction shape is", predictions.shape)

predictedLabels =[]
for i in range(0,len(predictions[:,0])):
    predictedLabels.append(np.argmax(predictions[i]))

#predictedLabels = np.asarray(predictedLabels)

true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

report = metrics.classification_report(true_classes, predictedLabels, target_names=class_labels)
print(report)
confusion_mtx = confusion_matrix(true_classes, predictedLabels)
plot_confusion_matrix(confusion_mtx, classes = range(5))


# ## Model 4
# Using adam optimiser, it mitigates the saturation issue which sigmoid posses.

# In[33]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Lambda, Input
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras import regularizers, optimizers

model4 = Sequential()

model4.add(Conv2D(64, (5, 5), input_shape = (32,32,3)))
model4.add(Activation('relu'))

model4.add(Conv2D(64, (5, 5)))
model4.add(Activation('relu'))

model4.add(Flatten())  
model4.add(Dense(128))
model4.add(Activation('relu'))
model4.add(Dropout(0.3))

model4.add(Dense(9))
model4.add(Activation('softmax'))

adam = optimizers.Adam(learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False)
model4.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=[categorical_accuracy])
adam = optimizers.Adam(learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False)
model4.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=[categorical_accuracy])
model4.summary()


# In[34]:


#training the model4
res = train_model(model4, num_epoch=100, verbose=1)


# In[35]:


#plot the results and plot the error loss for the training and accuracy data
def plot_results(res):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(res[:,0], res[:,1], 'r-')
    plt.plot(res[:,0], res[:,3], 'b-')
    plt.legend(['Train', 'Validation'])
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.ylim([0, np.max([5., np.max(res[:,1]), np.max(res[:,3])])])

    plt.subplot(1,2,2)
    plt.plot(res[:,0], res[:,2], 'r-')
    plt.plot(res[:,0], res[:,4], 'b-')
    plt.legend(['Train', 'Validation'])
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, np.max([1., np.max(res[:,2]), np.max(res[:,4])])])


# In[36]:


plot_results(res)


# In[37]:


#confusion matrix 
#Note, this code is taken straight from the SKLEARN website
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual class')
    plt.xlabel('Predicted class')


from collections import Counter
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import itertools

predictions = model4.predict_generator(test_generator,verbose=0)
print("Prediction shape is", predictions.shape)

predictedLabels =[]
for i in range(0,len(predictions[:,0])):
    predictedLabels.append(np.argmax(predictions[i]))

#predictedLabels = np.asarray(predictedLabels)

true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

report = metrics.classification_report(true_classes, predictedLabels, target_names=class_labels)
print(report)
confusion_mtx = confusion_matrix(true_classes, predictedLabels)
plot_confusion_matrix(confusion_mtx, classes = range(5))


# ## Model 5 - with decay element in adam optimiser

# In[38]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Lambda, Input
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras import regularizers, optimizers

model5 = Sequential()

model5.add(Conv2D(32, (3, 3), input_shape = (32,32,3)))
model5.add(Activation('relu'))

model5.add(Conv2D(64, (3, 3)))
model5.add(Activation('relu'))

model5.add(Flatten())  
model5.add(Dense(128))
model5.add(Activation('relu'))
model5.add(Dropout(0.3))

model5.add(Dense(9))
model5.add(Activation('softmax'))

adam = optimizers.Adam(learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-08,
    decay=5e-05,
    amsgrad=False)
model5.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=[categorical_accuracy])
model5.summary()


# In[39]:


#training the model4
res = train_model(model5, num_epoch=100, verbose=1)


# In[40]:


#plot the results and plot the error loss for the training and accuracy data
def plot_results(res):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(res[:,0], res[:,1], 'r-')
    plt.plot(res[:,0], res[:,3], 'b-')
    plt.legend(['Train', 'Validation'])
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.ylim([0, np.max([5., np.max(res[:,1]), np.max(res[:,3])])])

    plt.subplot(1,2,2)
    plt.plot(res[:,0], res[:,2], 'r-')
    plt.plot(res[:,0], res[:,4], 'b-')
    plt.legend(['Train', 'Validation'])
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, np.max([1., np.max(res[:,2]), np.max(res[:,4])])])


# In[41]:


plot_results(res)


# In[42]:


#confusion matrix 
#Note, this code is taken straight from the SKLEARN website
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual class')
    plt.xlabel('Predicted class')


from collections import Counter
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import itertools

predictions = model5.predict_generator(test_generator,verbose=0)
print("Prediction shape is", predictions.shape)

predictedLabels =[]
for i in range(0,len(predictions[:,0])):
    predictedLabels.append(np.argmax(predictions[i]))

#predictedLabels = np.asarray(predictedLabels)

true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

report = metrics.classification_report(true_classes, predictedLabels, target_names=class_labels)
print(report)
confusion_mtx = confusion_matrix(true_classes, predictedLabels)
plot_confusion_matrix(confusion_mtx, classes = range(5))


# # Model 6 - Adding more layers

# In[43]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Lambda, Input
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras import regularizers, optimizers

model6 = Sequential()

model6.add(Conv2D(64, (3, 3), input_shape = (32,32,3)))
model6.add(Activation('relu'))

model6.add(Conv2D(64, (3, 3)))
model6.add(Activation('relu'))

model6.add(Conv2D(64, (3, 3)))
model6.add(Activation('relu'))

model6.add(Flatten())  
model6.add(Dense(128))
model6.add(Activation('relu'))
model6.add(Dropout(0.3))

model6.add(Dense(9))
model6.add(Activation('softmax'))

adam = optimizers.Adam(learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-08,
    decay=5e-05,
    amsgrad=False)
model6.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=[categorical_accuracy])
model6.summary()


# In[44]:


#training the model1
res = train_model(model6, num_epoch=100, verbose=1)


# In[45]:


#plot the results and plot the error loss for the training and accuracy data
def plot_results(res):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(res[:,0], res[:,1], 'r-')
    plt.plot(res[:,0], res[:,3], 'b-')
    plt.legend(['Train', 'Validation'])
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.ylim([0, np.max([5., np.max(res[:,1]), np.max(res[:,3])])])

    plt.subplot(1,2,2)
    plt.plot(res[:,0], res[:,2], 'r-')
    plt.plot(res[:,0], res[:,4], 'b-')
    plt.legend(['Train', 'Validation'])
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, np.max([1., np.max(res[:,2]), np.max(res[:,4])])])


# In[46]:


plot_results(res)


# In[47]:


#confusion matrix 
#Note, this code is taken straight from the SKLEARN website
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual class')
    plt.xlabel('Predicted class')


from collections import Counter
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import itertools

predictions = model6.predict_generator(test_generator,verbose=0)
print("Prediction shape is", predictions.shape)

predictedLabels =[]
for i in range(0,len(predictions[:,0])):
    predictedLabels.append(np.argmax(predictions[i]))

#predictedLabels = np.asarray(predictedLabels)

true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

report = metrics.classification_report(true_classes, predictedLabels, target_names=class_labels)
print(report)
confusion_mtx = confusion_matrix(true_classes, predictedLabels)
plot_confusion_matrix(confusion_mtx, classes = range(5))


# ## Model 7 - using adam optimiser and Dropouts

# In[48]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Lambda, Input
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras import regularizers, optimizers

model7 = Sequential()

# input
model7.add(Input(shape=(32, 32, 3)))
model7.add(Lambda(lambda x: tf.expand_dims(x[:,:,:,0], -1, name=None))) 

# Conv Layer 1
model7.add(Conv2D(64, (3, 3),))
model7.add(Activation('relu'))
model7.add(MaxPooling2D(pool_size=(2, 2)))

# Conv Layer 2 (no pooling)
model7.add(Conv2D(64, (3, 3),))
model7.add(Activation('relu'))

# Conv Layer 3
model7.add(Conv2D(128, (3, 3)))
model7.add(Activation('relu'))
model7.add(MaxPooling2D(pool_size=(2, 2)))

# MLP
model7.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model7.add(Dense(128))
model7.add(Activation('relu'))
model7.add(Dropout(0.5))
model7.add(Dense(9))
model7.add(Activation('softmax'))

adam = optimizers.Adam(learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False)
model7.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=[categorical_accuracy])

# save the weights so that we can start from the same place when tring different configurations
# model_cnn.save_weights('model.h5')

model7.summary()


# In[49]:


#training the model1
res = train_model(model7, num_epoch=100, verbose=1)


# In[50]:


#plot the results and plot the error loss for the training and accuracy data
def plot_results(res):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(res[:,0], res[:,1], 'r-')
    plt.plot(res[:,0], res[:,3], 'b-')
    plt.legend(['Train', 'Validation'])
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.ylim([0, np.max([5., np.max(res[:,1]), np.max(res[:,3])])])

    plt.subplot(1,2,2)
    plt.plot(res[:,0], res[:,2], 'r-')
    plt.plot(res[:,0], res[:,4], 'b-')
    plt.legend(['Train', 'Validation'])
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, np.max([1., np.max(res[:,2]), np.max(res[:,4])])])


# In[51]:


plot_results(res)


# In[52]:


#confusion matrix 
#Note, this code is taken straight from the SKLEARN website
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual class')
    plt.xlabel('Predicted class')


from collections import Counter
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import itertools

predictions = model7.predict_generator(test_generator,verbose=0)
print("Prediction shape is", predictions.shape)

predictedLabels =[]
for i in range(0,len(predictions[:,0])):
    predictedLabels.append(np.argmax(predictions[i]))

#predictedLabels = np.asarray(predictedLabels)

true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

report = metrics.classification_report(true_classes, predictedLabels, target_names=class_labels)
print(report)
confusion_mtx = confusion_matrix(true_classes, predictedLabels)
plot_confusion_matrix(confusion_mtx, classes = range(5))


# ## Model 8 
# Using Drop outs reduces the complexity. 

# In[53]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Lambda, Input
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras import regularizers, optimizers

model8 = Sequential()

# input
model8.add(Input(shape=(32, 32, 3)))
model8.add(Lambda(lambda x: tf.expand_dims(x[:,:,:,0], -1, name=None))) 

# Conv Layer 1
model8.add(Conv2D(64, (3, 3),))
model8.add(Activation('relu'))
model8.add(Dropout(0.5))
model8.add(MaxPooling2D(pool_size=(2, 2)))

# Conv Layer 2 (no pooling)
model8.add(Conv2D(64, (3, 3),))
model8.add(Activation('relu'))
model8.add(Dropout(0.5))

# Conv Layer 3
model8.add(Conv2D(128, (3, 3)))
model8.add(Activation('relu'))
model8.add(Dropout(0.5))
model8.add(MaxPooling2D(pool_size=(2, 2)))

# MLP
model8.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model8.add(Dense(128))
model8.add(Activation('relu'))
model8.add(Dropout(0.5))
model8.add(Dense(9))
model8.add(Activation('softmax'))

adam = optimizers.Adam(learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False)
model8.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=[categorical_accuracy])

# save the weights so that we can start from the same place when tring different configurations
# model_cnn.save_weights('model.h5')

model8.summary()


# In[54]:


#training the model1
res = train_model(model8, num_epoch=100, verbose=1)


# In[55]:


#plot the results and plot the error loss for the training and accuracy data
def plot_results(res):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(res[:,0], res[:,1], 'r-')
    plt.plot(res[:,0], res[:,3], 'b-')
    plt.legend(['Train', 'Validation'])
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.ylim([0, np.max([5., np.max(res[:,1]), np.max(res[:,3])])])

    plt.subplot(1,2,2)
    plt.plot(res[:,0], res[:,2], 'r-')
    plt.plot(res[:,0], res[:,4], 'b-')
    plt.legend(['Train', 'Validation'])
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, np.max([1., np.max(res[:,2]), np.max(res[:,4])])])


# In[56]:


plot_results(res)


# In[57]:


#confusion matrix 
#Note, this code is taken straight from the SKLEARN website
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual class')
    plt.xlabel('Predicted class')


from collections import Counter
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import itertools

predictions = model8.predict_generator(test_generator,verbose=0)
print("Prediction shape is", predictions.shape)

predictedLabels =[]
for i in range(0,len(predictions[:,0])):
    predictedLabels.append(np.argmax(predictions[i]))

#predictedLabels = np.asarray(predictedLabels)

true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

report = metrics.classification_report(true_classes, predictedLabels, target_names=class_labels)
print(report)
confusion_mtx = confusion_matrix(true_classes, predictedLabels)
plot_confusion_matrix(confusion_mtx, classes = range(5))


# ## Prediction

# In[58]:


df = pd.read_csv('test_data.csv')
headpose_data
df.head()


# In[59]:


dfv2 = pd.merge(df,headpose_data, on='filename')


# In[60]:


dfv2.head()


# In[61]:


submission_df = dfv2[['filename','image_path']]


# In[62]:


new_datagen = ImageDataGenerator(rescale=1./255, data_format='channels_last')

new_generator = new_datagen.flow_from_dataframe(
    dataframe=submission_df,
    directory='./modified_data', 
    x_col="filename", 
    y_col="image_path", 
    target_size=(32, 32), 
    batch_size=batch_size,
    shuffle=False,
    class_mode='categorical')


# In[63]:


y_pred = model8.predict_generator(new_generator,verbose=1)
print("Test dataset shape", y_pred.shape)

Y_pred_labels =[]
for i in range(0,len(y_pred[:,0])):
    Y_pred_labels.append(np.argmax(predictions[i]))

Y_pred_labels = np.asarray(Y_pred_labels)


# In[64]:


for image, x in enumerate(Y_pred_labels):
    if x == 0:
        predictedLabels[image] = -15
    elif x == 1:
        predictedLabels[image] = -30
    elif x == 2:
        predictedLabels[image] = -60
    elif x == 3:
        predictedLabels[image] = -90
    elif x == 4:
        predictedLabels[image] = 0
    elif x == 5:
        predictedLabels[image] = 15
    elif x == 6:
        predictedLabels[image] = 30
    elif x == 7:
        predictedLabels[image] = 60
    else:
        predictedLabels[image] = 90


# In[65]:


import pandas as pd
df = pd.DataFrame(predictedLabels, columns=["tilt"])
result = pd.concat([submission_df['filename'],df], axis=1, sort=False)


# In[66]:


result.head()


# In[67]:


result.to_csv('s3763905_tilt_predictions.csv', index=False)


# In[ ]:




