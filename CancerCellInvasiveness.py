
#import cv2
import pickle
import tensorflow as tf
from tensorflow import keras
from keras import utils
from keras import layers
from keras import backend
from keras import datasets
from keras import metrics
import keras.models
from keras.models import Sequential
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten, Conv2D, Activation, Dense
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
#from keras.callbacks import ModelCheckpoint
from PIL import Image
from PIL import ImageEnhance

from keras import backend as K

import itertools
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import math

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import os
import csv
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import KFold
import shutil

import pandas as pd
import plotly.express as px


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def image_combine(image_root_path):
    for i in range(1, 21):
        img_pre = Image.open(image_root_path + '/Before/' + str(i) + '.png')
        img_post = Image.open(image_root_path + '/After/' + str(i) + '.png')

        img_pre = img_pre.convert('L')
        img_post = img_post.convert('L')
        
        enhancer = ImageEnhance.Contrast(img_pre)
        img_pre = enhancer.enhance(3)
        #img_pre.save(image_root_path + '/Combine/' + str(i) + 'preEn.png')
        
        enhancer = ImageEnhance.Contrast(img_post)
        img_post = enhancer.enhance(3)
        #img_post.save(image_root_path + '/Combine/' + str(i) + 'postEn.png')
        
        img_Comb = Image.new("RGB", img_pre.size)
        img_Comb_R = Image.new("RGB", img_pre.size)
        img_Comb_G = Image.new("RGB", img_pre.size)
        img_Comb_B = Image.new("RGB", img_pre.size)
        
        pixels_pre = list(img_pre.getdata())
        pixels_post = list(img_post.getdata())
        pixels_Comb = list(img_Comb.getdata())
        
        pixels_Comb_R = list(img_Comb.getdata())
        pixels_Comb_G = list(img_Comb.getdata())
        pixels_Comb_B = list(img_Comb.getdata())
        
        for j in range(len(pixels_pre)):
            #pixels_Comb[j] = (min(pixels_pre[j] + 0, 255), min(pixels_post[j] + 0, 255), int((pixels_pre[j] + pixels_post[j])/2))
            pixels_Comb[j] = (min(pixels_pre[j] + 0, 255), min(pixels_post[j] + 0, 255), int((pixels_pre[j] + pixels_post[j])/4))
            pixels_Comb_R[j] = (min(pixels_pre[j] + 0, 255),0,0)
            pixels_Comb_G[j] = (0, min(pixels_post[j] + 0, 255),0)
            pixels_Comb_B[j] = (0,0, int((pixels_pre[j] + pixels_post[j])/2))
            
        
        img_Comb.putdata(pixels_Comb)
        img_Comb.save(image_root_path + '/Combine/' + str(i) + '.png')
        img_Comb_R.putdata(pixels_Comb_R)
        #img_Comb_R.save(image_root_path + '/Combine/' + str(i) + '_R.png')
        img_Comb_G.putdata(pixels_Comb_G)
        #img_Comb_G.save(image_root_path + '/Combine/' + str(i) + '_G.png')
        img_Comb_B.putdata(pixels_Comb_B)
        #img_Comb_B.save(image_root_path + '/Combine/' + str(i) + '_B.png')

    return


def data_aug_TRUE(image_root_path):

    data_aug_gen = ImageDataGenerator(rotation_range=40,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      #rescale=1./255,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      vertical_flip=True,
                                      horizontal_flip=True,
                                      fill_mode='nearest')

    for i in range(1, 21):
        img = load_img(image_root_path + str(i) + '.png')
        imgArray = img_to_array(img)
        imgArray = imgArray.reshape((1,) + imgArray.shape)

        j = 0
        for batch in data_aug_gen.flow(imgArray,
                                       batch_size=1,
                                       save_to_dir='D:/Google Drive/00.with_HGLim/ML1/01.Experiment/Training/True/'+str(i),
                                       #save_prefix='Training',
                                       save_format='png'):
            j += 1
            if j > 200:
                break
    
    """        
    for i in range(4, 6):
        img = load_img(image_root_path + str(i) + '.png')
        imgArray = img_to_array(img)
        imgArray = imgArray.reshape((1,) + imgArray.shape)

        i = 0
        for batch in data_aug_gen.flow(imgArray,
                                       batch_size=1,
                                       save_to_dir=image_root_path + '/Aug',
                                       save_prefix='Testing',
                                       save_format='png'):
            i += 1
            if i > 100:
                break
    """
    
    return


def data_aug_FALSE(image_root_path):

    data_aug_gen = ImageDataGenerator(rotation_range=40,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      #rescale=1./255,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      vertical_flip=True,
                                      horizontal_flip=True,
                                      fill_mode='nearest')

    for i in range(1, 21):
        img = load_img(image_root_path + str(i) + '.png')
        imgArray = img_to_array(img)
        imgArray = imgArray.reshape((1,) + imgArray.shape)

        j = 0
        for batch in data_aug_gen.flow(imgArray,
                                       batch_size=1,
                                       save_to_dir='D:/Google Drive/00.with_HGLim/ML1/01.Experiment/Training/False/'+str(i),
                                       #save_prefix='Training',
                                       save_format='png'):
            j += 1
            if j > 200:
                break
    
    """        
    for i in range(4, 6):
        img = load_img(image_root_path + str(i) + '.png')
        imgArray = img_to_array(img)
        imgArray = imgArray.reshape((1,) + imgArray.shape)

        i = 0
        for batch in data_aug_gen.flow(imgArray,
                                       batch_size=1,
                                       save_to_dir=image_root_path + '/Aug',
                                       save_prefix='Testing',
                                       save_format='png'):
            i += 1
            if i > 100:
                break
    """
    
    return





def Fold(image_root_path):
    
    for i in range(0,5):
        for j in range(1,21):
            
            file_list = os.listdir(image_root_path+'/Training/True/'+str(j))
            if j%5 ==i:
                for k in file_list:
                    shutil.copy(image_root_path+'/Training/True/'+str(j)+'/'+str(k), image_root_path+'/Fold_'+str(i)+'/Testing/True/I'+str(j)+str(k))
            else:
                for k in file_list:
                    shutil.copy(image_root_path+'/Training/True/'+str(j)+'/'+str(k), image_root_path+'/Fold_'+str(i)+'/Training/True/I'+str(j)+str(k))
                
            file_list = os.listdir(image_root_path+'/Training/False/'+str(j))
            if j%5 ==i:
                for k in file_list:
                    shutil.copy(image_root_path+'/Training/False/'+str(j)+'/'+str(k), image_root_path+'/Fold_'+str(i)+'/Testing/False/I'+str(j)+str(k))
            else:
                for k in file_list:
                    shutil.copy(image_root_path+'/Training/False/'+str(j)+'/'+str(k), image_root_path+'/Fold_'+str(i)+'/Training/False/I'+str(j)+str(k))
            
            
    return





def plt_show_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    #plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training Error', 'Testing Error'], loc=0)


def plt_show_acc(history):
    plt.plot(history.history['val_precision_m'])
    plt.plot(history.history['val_recall_m'])
    plt.plot(history.history['val_f1_m'])
    plt.plot(history.history['val_accuracy'])
    #plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Precision', 'Recall', 'F1 Measure', 'Accuracy'], loc=0)


def prediction(model,image_root_path):
    batch_size = 4

    train_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    
    for i in range(0,5):
        train_generator = train_datagen.flow_from_directory(
                          'D:/Fold_'+str(i)+ '/Training',  
                          target_size=(150, 150),  
                          batch_size=batch_size,
                          class_mode='binary') 
        
        validation_generator = validation_datagen.flow_from_directory(
                               'D:/Fold_'+str(i)+ '/Testing',
                               target_size=(150, 150),
                               batch_size=batch_size,
                               class_mode='binary')
        
        csv_logger = CSVLogger('log' + '_Fold_' + str(i) + '.csv', append=True, separator=',') # log save path
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
        mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)
        
        history = model.fit_generator(train_generator,
                                      steps_per_epoch=1000/batch_size,
                                      validation_data=validation_generator,
                                      epochs=200,
                                      callbacks=[mc, csv_logger])
                                      #callbacks=[early_stopping, mc, csv_logger])
                        
        model.save_weights('first_try.h5')  # model save path
    
        plt.figure(figsize=(5, 5))
        plt_show_loss(history)
        plt.show()
        plt.savefig('Loss_Fold_' + str(i) + '.png')
    
        plt.figure(figsize=(5, 5))
        plt_show_acc(history)
        plt.show()
        plt.savefig('Acc_Fold_' + str(i) + '.png')


    return history


def build_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(150, 150, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #model.add(Conv2D(64, kernel_size=(3, 3)))
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    sgd = keras.optimizers.SGD(lr=0.00025)
    rmsprop = keras.optimizers.RMSprop(learning_rate=0.00025)
    adagrad = keras.optimizers.Adagrad(learning_rate=0.025)
    adadelta = keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95)
    adam = keras.optimizers.Adam(learning_rate=0.0025)

    model.compile(loss = 'binary_crossentropy',
                  optimizer = rmsprop,
                  metrics = [f1_m, precision_m, recall_m, 'accuracy'])

    return model


def prediction_FOLD(model,image_root_path,fold):
    batch_size = 8

    train_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    
    for i in range(fold,fold+1):
        train_generator = train_datagen.flow_from_directory(
                          'D:/Fold_'+str(i)+ '/Training',  
                          target_size=(150, 150),  
                          batch_size=batch_size,
                          class_mode='binary') 
        
        validation_generator = validation_datagen.flow_from_directory(
                               'D:/Fold_'+str(i)+ '/Testing',
                               target_size=(150, 150),
                               batch_size=batch_size,
                               class_mode='binary')
        
        csv_logger = CSVLogger('log' + '_Fold_' + str(i) + '_adadelta.csv', append=True, separator=',') # log save path
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
        mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)
        
        history = model.fit_generator(train_generator,
                                      steps_per_epoch=1000/batch_size,
                                      validation_data=validation_generator,
                                      epochs=100,
                                      callbacks=[mc, csv_logger])
                                      #callbacks=[early_stopping, mc, csv_logger])
                        
        model.save_weights('first_try.h5')  # model save path
    
        plt.figure(figsize=(5, 5))
        plt_show_loss(history)
        plt.show()
    
        plt.figure(figsize=(5, 5))
        plt_show_acc(history)
        plt.show()
        

    return history



image_path = 'D:/Google Drive/00.with_HGLim/ML1/01.Experiment'

#image_combine(image_path + '/Deform_less')
#image_combine(image_path + '/Deform_more')

#data_aug_FALSE(image_path + '/Deform_less/Combine/')
#data_aug_TRUE(image_path + '/Deform_more/Combine/')

#Fold(image_path)

#history = prediction(build_model(),image_path)


#model = build_model()
#history = prediction(build_model(),image_path)
#history = prediction_FOLD(build_model(),image_path,0)
#history = prediction_FOLD(build_model(),image_path,1)
#history = prediction_FOLD(build_model(),image_path,2)
#history = prediction_FOLD(build_model(),image_path,3)
#history = prediction_FOLD(build_model(),image_path,4)

df = pd.read_csv('D:/Google Drive/00.with_HGLim/ML1/01.Experiment/log_Fold_1_adam.csv')

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "50"

plt.figure(figsize=(20, 20))
plt.plot(df['loss'],linewidth=7)
plt.plot(df['val_loss'],linewidth=7)
    #plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training Error', 'Testing Error'], loc=0)


plt.figure(figsize=(20, 20))
plt.plot(df['val_precision_m'],linewidth=7)
plt.plot(df['val_recall_m'],linewidth=7)
plt.plot(df['val_f1_m'],linewidth=7)
plt.plot(df['val_accuracy'],linewidth=7)
    #plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Precision', 'Recall', 'F1 Measure', 'Accuracy'], loc=0)

fig = px.line(df, x = 'epoch', y = 'val_accuracy', title='aa')
fig.show()

#model.summary()

#data_aug(image_path + '/Deform_more/Combine/')
#image_combine(image_path + '/Deform_more')


#history = prediction(build_model(),image_path)





