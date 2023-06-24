# -*- coding: utf-8 -*-
"""
Created on Tue May 16 15:47:17 2023

@author: Nagihan
"""
import json
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image



# initialize image data gen for train and test(valid)

train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)

# all  train images ön işleme
#klasör tabanlı görüntü verilerini burada flo_fromdirectory ile alıyoruz
#verilere ön işleme yapıyoruz, boyutlar 48x48, renk gri ve sınıfı kategori tipi ayarlıyoruz
#train_data_gen işlemini girdiğimiz dizindeki görsellere uygula

train_generator = train_data_gen.flow_from_directory(
    'data/train',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical')

# all test images ön işleme

validation_generator = validation_data_gen.flow_from_directory(
    'data/test',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical')

# CREATE CNN MODEL /structure
# sinir ağı modelimizi oluşturuyoruz 
# sıralı katman halinde bir yapı kuruyoruz sequential ile
emotion_model = Sequential()

# Özellikleri saptar, filtre uygular
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
# veri küçültür, öğrenme işini huzlandırır
emotion_model.add(MaxPooling2D(pool_size=(2, 2))) #filtreleme uygular 2x2 lik/ boyut azaltmak için en büyükleri alır maxpooling
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))

emotion_model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten()) # fully connected yani sınıflandırma kısmı için model girdisini oluşturur/ vektör haline getirir
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.25)) #modelin ezberlemesini önlemek için kullanıldı
emotion_model.add(Dense(7, activation='softmax')) #çıkış katmanı 7 düğümden oluşuyor çünkü 7 sınıfımız mevcut/ her girdi için 7 boyutlu bir vektör çıkacak ve elde edilen değerin hangi sınıfa ait olup olmadığı belli olucak / çıkış katmanı aktivasyon fonk olarak softmax kullanıldı


# compile model
#modelimizi derliyoruz

emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

# Train the model

emotion_model_info = emotion_model.fit_generator(
    train_generator,
    steps_per_epoch=28709 // 64,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=7178 // 64)

# save model structure

model_json = emotion_model.to_json()
with open("emotion_model.json", "w") as json_file:
    json_file.write(model_json)
    
# save trained model weight in .h5 file
# öğrenmeler h5 dosyasına kaydedildi

emotion_model.save_weights('emotion_model.h5')
