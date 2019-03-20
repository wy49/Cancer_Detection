
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import shutil as st
import multiprocessing as mp
import h5py


# In[ ]:


dtypes = {
        'id'            : 'str',
        'label'           : 'int8',
        }
train_labels = pd.read_csv('../input/train_labels.csv',dtype = dtypes)
train_labels.head()


# In[ ]:


#balance 0 and 1 in train data
np.random.seed(23333)
train_labels.groupby(['label']).size() #0:130908 1: 89117
index = np.where(train_labels['label']==0)[0]
index = np.random.choice(index,89117,replace=False)

index = np.concatenate([index,np.where(train_labels['label']==1)[0]])
index,len(index)


# In[ ]:


#create folders for flow from directory
if not os.path.exists('../working/train/true/'):
    os.makedirs('../working/train/true/')
if not os.path.exists('../working/train/false/'):
    os.makedirs('../working/train/false/')
if not os.path.exists('../working/validate/true/'):
    os.makedirs('../working/validate/true/')
if not os.path.exists('../working/validate/false/'):
    os.makedirs('../working/validate/false/')
    
print(len(os.listdir('../working/train/false')))


# In[ ]:


#split train to new train/validate
n = len(train_labels)
X_train, X_test, y_train, y_test = train_test_split(train_labels.iloc[index]['id'], train_labels.iloc[index]['label'], test_size=0.2, random_state=23333)


# In[ ]:


READ_FLAGS = os.O_RDONLY
WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
BUFFER_SIZE = 128*1024
# BUFFER_SIZE = 1024

def copyfile(src, dst):
    try:
        fin = os.open(src, READ_FLAGS)
        stat = os.fstat(fin)
        fout = os.open(dst, WRITE_FLAGS, stat.st_mode)
        for x in iter(lambda: os.read(fin, BUFFER_SIZE), b""):
            os.write(fout, x)
#             print('yes')
    finally:
        try: 
            os.close(fin)
#             print('fin')
        except: 
            pass
            print('nin')
        try: 
            os.close(fout)
#             print('fout')
        except:
            pass
            print('nout')


# In[ ]:


for name in X_train:
    label = train_labels[train_labels['id'] == name]['label'].values
    if label == 1:
        copyfile('../input/train/'+name+'.tif','../working/train/true/'+name+'.tif')
    elif label == 0:
        copyfile('../input/train/'+name+'.tif','../working/train/false/'+name+'.tif')
    else:
        break
        print('train folder error')


# In[ ]:


for name in X_test:
    label = train_labels[train_labels['id'] == name]['label'].values
    if label == 1:
        copyfile('../input/train/'+name+'.tif','../working/validate/true/'+name+'.tif')
    elif label == 0:
        copyfile('../input/train/'+name+'.tif','../working/validate/false/'+name+'.tif')
    else:
        break
        print('train folder error')


# In[ ]:


#copy and split data into train/validate folder
#SUPER SLOWWWWWW
# for name in X_train:
#     label = train_labels[train_labels['id'] == name]['label'].values
#     if label == 1:
#         st.copyfile('../input/train/'+name+'.tif','../working/train/true/'+name+'.tif')
#     elif label == 0:
#         st.copyfile('../input/train/'+name+'.tif','../working/train/false/'+name+'.tif')
#     else:
#         break
#         print('train folder error')
        
# for name in X_test:
#     label = train_labels[train_labels['id'] == name]['label'].values
#     if label == 1:
#         st.copyfile('../input/train/'+name+'.tif','../working/validate/true/'+name+'.tif')
#     elif label == 0:
#         st.copyfile('../input/train/'+name+'.tif','../working/validate/false/'+name+'.tif')
#     else:
#         break
#         print('validate folder error')
    


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(True,True,True,True,
        rescale=1./255,
        rotation_range=40,                           
#         shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip= True)

train_generator = train_datagen.flow_from_directory(
    directory="../working/train",
    target_size=(96, 96),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=23333
)


# In[ ]:


valid_datagen = ImageDataGenerator(True,True,True,True,
        rescale=1./255,
        rotation_range=40,                           
#         shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip= True)

valid_generator = valid_datagen.flow_from_directory(
    directory="../working/validate",
    target_size=(96, 96),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=23333
)


# In[ ]:


import gc
gc.collect()


# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,BatchNormalization,Dense, Dropout, Flatten

cnn1 = Sequential()
cnn1.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(96,96,3)))
cnn1.add(BatchNormalization())

cnn1.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
cnn1.add(BatchNormalization())
cnn1.add(MaxPooling2D(pool_size=(2, 2)))
cnn1.add(Dropout(0.25))

cnn1.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
cnn1.add(BatchNormalization())
cnn1.add(Dropout(0.25))

cnn1.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
cnn1.add(BatchNormalization())
cnn1.add(MaxPooling2D(pool_size=(2, 2)))
cnn1.add(Dropout(0.25))

cnn1.add(Flatten())

cnn1.add(Dense(512, activation='relu'))
cnn1.add(BatchNormalization())
cnn1.add(Dropout(0.5))

cnn1.add(Dense(128, activation='relu'))
cnn1.add(BatchNormalization())
cnn1.add(Dropout(0.5))

cnn1.add(Dense(2, activation='softmax'))

cnn1.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
# view raw


# In[ ]:


callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=2),
             keras.callbacks.ModelCheckpoint(filepath='cnn_3_8.h5', monitor='val_loss', save_best_only=True)]


# In[ ]:


cnn1.fit_generator(train_generator, 
                   steps_per_epoch=np.ceil(142587 / 32), 
                   epochs=5, 
                   verbose=1,
                   validation_data=valid_generator,
                   validation_steps = np.ceil(35647 / 32))

