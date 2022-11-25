# Facial Detection and Emotional Recognition 
# Author: Ryan 
# Deep Learning

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.layers import Flatten, Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
from keras.applications.mobilenet import MobileNet, preprocess_input 
from keras.losses import categorical_crossentropy

base_model = MobileNet( input_shape=(224,224,3), include_top= False )

for layer in base_model.layers:
  layer.trainable = False


x = Flatten()(base_model.output)
x = Dense(units=7 , activation='softmax' )(x)

model = Model(base_model.input, x)
model.compile(optimizer='adam', loss= categorical_crossentropy , metrics=['accuracy'])

train_datagen = ImageDataGenerator(zoom_range = 0.2, shear_range = 0.2, horizontal_flip=True, rescale = 1./255)
train_data = train_datagen.flow_from_directory(directory= "Facial_Recognition_and_Emotion_Detection/train", target_size=(224,224), batch_size=64,)
train_data.class_indices

val_datagen = ImageDataGenerator(rescale = 1./255 )

val_data = val_datagen.flow_from_directory(directory="Facial_Recognition_and_Emotion_Detection/test", target_size=(224,224), batch_size=32,)

t_img , label = train_data.next()

def plotImages(img_arr, label):
  count = 0
  for im, l in zip(img_arr,label) :
    plt.imshow(im)
    plt.title(im.shape)
    plt.axis = False
    plt.show() 
    count += 1
    if count == 10:
      break

plotImages(t_img, label)

from keras.callbacks import ModelCheckpoint, EarlyStopping

es = EarlyStopping(monitor='val_accuracy', min_delta= 0.01 , patience= 5, verbose= 1, mode='auto')
mc = ModelCheckpoint(filepath="best_model.h5", monitor= 'val_accuracy', verbose= 1, save_best_only= True, mode = 'auto')
call_back = [es, mc]

hist = model.fit_generator(train_data, 
                           steps_per_epoch= 15, 
                           epochs= 30, 
                           validation_data= val_data, 
                           validation_steps= 8, 
                           callbacks=[es,mc])
                           
from keras.models import load_model
model = load_model("Facial_Recognition_and_Emotion_Detection/test/happy/PrivateTest_647018.jpg")

# Credits Edureka