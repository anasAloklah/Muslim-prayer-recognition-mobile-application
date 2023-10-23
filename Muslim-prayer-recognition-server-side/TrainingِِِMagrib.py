#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam

print(tf.__version__)

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# In[3]:


pd.read_csv('Magrib.csv')

# In[4]:


pd.read_csv('Magrib.csv')

# In[5]:


file = open('Magrib.csv')
lines = file.readlines()

processedList = []

for i, line in enumerate(lines):
    try:
        line = line.split(',')
        last = line[5].split(';')[0]
        last = last.strip()
        if last == '':
            break;
        temp = [line[0], line[1], line[2], line[3], line[4], last]
        processedList.append(temp)
    except:
        print('Error at line number: ', i)

# In[6]:


processedList

# In[7]:


columns = ['user', 'activity', 'time', 'x', 'y', 'z']
data = pd.DataFrame(data=processedList, columns=columns)
data.head()

# In[8]:


data.shape

# In[9]:


data.info()

# In[10]:


data.isnull().sum()

# In[11]:


data['activity'].value_counts()

# In[12]:


data['x'] = data['x'].astype('float')
data['y'] = data['y'].astype('float')
data['z'] = data['z'].astype('float')

# In[13]:


data.info()

# In[14]:


Fs = 20

# In[15]:


activities = data['activity'].value_counts().index

# In[16]:


activities


# In[17]:


def plot_activity(activity, data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 7), sharex=True)
    plot_axis(ax0, data['time'], data['x'], 'X-Axis')
    plot_axis(ax1, data['time'], data['y'], 'Y-Axis')
    plot_axis(ax2, data['time'], data['z'], 'Z-Axis')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()


def plot_axis(ax, x, y, title):
    ax.plot(x, y, 'g')
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)


"""for activity in activities:
    data_for_plot = data[(data['activity'] == activity)][:Fs * 10]
    plot_activity(activity, data_for_plot)
"""
# In[18]:


df = data.drop(['user', 'time'], axis=1).copy()
df.head()

# In[19]:


df['activity'].value_counts()

# In[20]:


Standing = df[df['activity'] == 'Standing'].copy()
Sitting = df[df['activity'] == 'Sitting'].copy()
#Prostartion = df[df['activity'] == 'Prostartion'].head(1500).copy()
Prostartion = df[df['activity'] == 'Prostration'].copy()
Bowing = df[df['activity'] == 'Bowing'].copy()

# In[21]:


balanced_data = pd.DataFrame()
balanced_data = balanced_data.append([Standing, Sitting, Prostartion, Bowing])
balanced_data.shape

# In[22]:


balanced_data['activity'].value_counts()

# In[23]:


balanced_data.head()

# In[24]:


from sklearn.preprocessing import LabelEncoder

# In[25]:


label = LabelEncoder()
balanced_data['label'] = label.fit_transform(balanced_data['activity'])
balanced_data.head()

# In[26]:


label.classes_

# In[27]:


X = balanced_data[['x', 'y', 'z']]
y = balanced_data['label']

# In[28]:


scaler = StandardScaler()
#X = scaler.fit_transform(X)

scaled_X = pd.DataFrame(data=X, columns=['x', 'y', 'z'])
scaled_X['label'] = y.values

scaled_X

# In[29]:


import scipy.stats as stats

# In[30]:


Fs = 20
frame_size = Fs * 4  # 80
hop_size = Fs * 2  # 40


# In[31]:


def get_frames(df, frame_size, hop_size):
    N_FEATURES = 3

    frames = []
    labels = []
    for i in range(0, len(df) - frame_size, hop_size):
        x = df['x'].values[i: i + frame_size]
        y = df['y'].values[i: i + frame_size]
        z = df['z'].values[i: i + frame_size]

        # Retrieve the most often used label in this segment
        label = stats.mode(df['label'][i: i + frame_size])[0][0]
        frames.append([x, y, z])
        labels.append(label)

    # Bring the segments into a better shape
    frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
    labels = np.asarray(labels)

    return frames, labels


# In[32]:


X, y = get_frames(scaled_X, frame_size, hop_size)

# In[33]:


X.shape, y.shape

# In[34]:


(3555 * 6) / 40

# In[35]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# In[36]:


X_train.shape, X_test.shape

# In[37]:


X_train[0].shape, X_test[0].shape

# In[38]:


X_train = X_train.reshape(1844, 80, 3, 1)
X_test = X_test.reshape(462, 80, 3, 1)

# In[39]:


X_train[0].shape, X_test[0].shape

# In[40]:


model = Sequential()
model.add(Conv2D(16, (2, 2), activation='relu', input_shape=X_train[0].shape))
model.add(Dropout(0.02))

model.add(Conv2D(32, (2, 2), activation='relu'))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.01))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(8, activation='softmax'))
model.load_weights('asar_weights.h5')
# In[41]:


model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# In[42]:


history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), verbose=1)

# In[43]:


history.history

# In[44]:


history.history


# In[45]:


def plot_learningCurve(history, epochs):
    # Plot training & validation accuracy values
    epoch_range = range(1, epochs + 1)
    plt.plot(epoch_range, history.history['loss'])
    plt.plot(epoch_range, history.history['val_loss'])
    # plt.plot(epoch_range, history.history['val_accuracy'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()
    # Plot training & validation loss values
    plt.plot(epoch_range, history.history['accuracy'])
    plt.plot(epoch_range, history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()


# In[46]:


#plot_learningCurve(history, 20)

# In[47]:


#from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report

# In[48]:


y_pred = model.predict_classes(X_test)

# In[49]:


print(confusion_matrix(y_test, y_pred))
print('\n')
print(classification_report(y_test, y_pred))

mat = confusion_matrix(y_test, y_pred)
#plot_confusion_matrix(conf_mat=mat, class_names=label.classes_, show_normed=True, figsize=(7, 7))
# plot_confusion_matrix(conf_mat=mat, class_names=label.classes_, show_normed=True, figsize=(7,7))


# In[283]:

model.save_weights('asar_weights.h5')
#model.save_weights('magrib_weights.h5')
model.save('model.h5')

# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:




