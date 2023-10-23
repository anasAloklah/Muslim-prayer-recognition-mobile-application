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


pd.read_csv('Isha.csv')

# In[4]:


pd.read_csv('Isha.csv')

# In[5]:


file = open('Isha.csv')
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
shap1=X_train[0].shape

X_train[0].shape, X_test[0].shape

# In[38]:


X_train = X_train.reshape(2445, 80, 3, 1)
X_test = X_test.reshape(612, 80, 3, 1)

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

#model.load_weights('asar_weights.h5')
#model.save_weights('isha_weights.h5')
model.save_weights('asar_weights.h5')
model.save('model.h5')
dataS="0.043086052;0.7467132;9.859121|0.020360291;0.73462725;9.874607|0.047271118;0.7451713;9.900895|0.024880545;0.72695625;9.942133|0.043766007;0.74341875;9.9259|0.030147787;0.7493468;9.915998|0.024009055;0.75040025;9.902475|0.03369121;0.72533774;9.894125|0.027131094;0.7492702;9.88539|0.01697968;0.74413705;9.86028|0.028864495;0.73942524;9.850148|0.03506069;0.74017227;9.865146|0.021452047;0.747058;9.841941|0.029831752;0.7431123;9.859676|0.017353173;0.7443765;9.852944|0.022476764;0.7422791;9.875393|0.02207454;0.7361404;9.847389|0.031996112;0.7488584;9.855214|0.034984075;0.7283449;9.861841|0.03196738;0.73364085;9.856354|0.028969841;0.7357957;9.8474|0.0267576;0.73412925;9.866783|0.027092787;0.7494617;9.867443|0.0424061;0.74043083;9.855349|0.032369606;0.7289769;9.877586|0.01934515;0.7332003;9.860079|0.00998861;0.7283449;9.860941|0.029132647;0.73404306;9.86029|0.007182605;0.7338132;9.851364|0.032905906;0.7386783;9.853271|0.021547815;0.7454778;9.862521|0.018454505;0.73038477;9.873679|0.032522835;0.71865314;9.845206|0.047520116;0.7235278;9.87568|0.036276944;0.7386208;9.857695|0.020685904;0.7460045;9.875526|0.029697677;0.73061454;9.854707|0.029257145;0.7290057;9.869216|0.030550014;0.7364852;9.848194|0.024985889;0.7324246;9.857369|0.030281862;0.7335929;9.835477|0.039130833;0.74314106;9.852648|0.023836672;0.72799057;9.8521595|0.018732235;0.72377676;9.853777|0.015830461;0.7408139;9.860692|0.004376601;0.730289;9.876082|0.027370514;0.7358148;9.8738785|0.035874717;0.7412544;9.875622|0.029745562;0.7357095;9.866774|0.014192827;0.72977185;9.861611|0.008446744;0.7324725;9.873697|0.0283665;0.72616136;9.861535|0.017190369;0.7456693;9.870595|0.020456059;0.73323864;9.862675|0.026048914;0.7274638;9.85825|0.012727576;0.7420876;9.861478|0.027964275;0.73682994;9.856325|0.009634268;0.7407756;9.856794|0.025771188;0.73903257;9.856268|0.044015;0.73325783;9.84648|0.0119039705;0.7243705;9.855789|0.021164743;0.735968;9.836099|0.0089543145;0.72597945;9.85394|0.021423317;0.74246114;9.86776|0.021021092;0.73344934;9.8910885|0.024679432;0.7305859;9.921716|0.015150508;0.7370885;9.926647|0.028663384;0.7463014;9.926351|0.02387498;0.7409671;9.913489|0.017602172;0.7297048;9.893818|0.042453982;0.73573816;9.885937|0.015945382;0.74675155;9.882057|0.032206804;0.73137116;9.852198|0.03331771;0.732032;9.864398|0.021729775;0.73377496;9.85055|0.019364303;0.7293025;9.8817425|0.028174965;0.72006094;9.862876|0.025158271;0.7273202;9.855233|0.030004134;0.7329705;9.864657|0.030731974;0.73041344;9.847754|"

dataS = dataS.rstrip('|')
rows = dataS.split('|')
arr = []
for row in rows:
    value = row.split(';')
    v=[]
    v.append(float(value[0]))
    v.append(float(value[1]))
    v.append(float(value[2]))
    arr.append(v)
freamArry=arr

list=[]
list.append(freamArry)
list=np.asarray(list).reshape(1, 80, 3, 1)
#list = list.reshape(1, 80, 3, 1)
prediction = model.predict_classes(list)  # predict the result
print(prediction)
res = ''
if prediction == [0]:
    res = 'standing'
elif prediction == [1]:
    res = 'Bowing'
elif prediction == [2]:
    res = 'prostration'
elif prediction == [3]:
    res = 'Sitting'

print(res)
# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:




