# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 17:05:41 2023

@author: alexa
"""

import pandas as pd
import numpy as np
from sklearn import model_selection as ms
import prepareData as prep
from sklearn.ensemble import RandomForestRegressor as forest
from sklearn.metrics import r2_score as r2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import random as rn

direct='Ваше расположение файла'
ds=pd.read_excel(io=direct+'\Разметка данных Итог.xlsx', sheet_name='ds2_no_med')

replace_dict=prep.cleaning.strToIndex([ds['Правонарушение'], ds['Вид деятельности'], ds['Регионы'], ds['Пол']])
#print((replace_dict))

ds.info()


#%%
#создание функционально зависимых предсказываемых параметров с наложением шума для теста способности моделей находить объяснения
ds['Приведенный срок 2']=(ds['Правонарушение']/(ds['Пол']+1)  + pow(ds['Регионы'],0.5)+ pow(ds['Вид деятельности'],0.8)+
                              ((pow(ds['Год рождения'],2)-3940*ds['Год рождения'])/pow(10,6)+10 ))*abs(np.random.normal(1, 0.8))*(1+rn.randrange(0, 100)/100)

ds['Штраф 2']=100*(ds['Правонарушение']  + pow(ds['Регионы'],0.5)+ pow(ds['Вид деятельности'],0.8)+
                              ((pow(ds['Год рождения'],2)-3940*ds['Год рождения'])/pow(10,6)+10 ))*abs(np.random.normal(1, 0.8))*(1+rn.randrange(0, 100)/100)
#print(ds['Приведенный срок 2'])
#ds.info()
#%%		
trainSet, testSet=ms.train_test_split(ds, train_size=0.8)
x_train=np.array(trainSet[[ 'Пол',  'Год рождения', 'Вид деятельности', 'Регионы', 'Правонарушение']]) #'Пол', 'Год рождения', 'Вид деятельности', 'Регионы'
y_train=np.array(trainSet[['Приведенный срок' , 'Штраф']]) #, 'Штраф 2'

x_test=np.array(testSet[['Пол', 'Год рождения', 'Вид деятельности', 'Регионы', 'Правонарушение']]) #, 'Пол', 'Год рождения', 'Вид деятельности', 'Регионы'
y_test=np.array(testSet[['Приведенный срок'  , 'Штраф']])

#plt.scatter(ds['Правонарушение'], ds['Приведенный срок 2'])

#%%
#Модель деревьев решений
for i in range(20):
    
    tree=forest(n_estimators=300, max_features=1) #Объявление модели (400 - количество деревьев леса), max_ - метод определеиня? погуглить)
    tree.fit(x_train, y_train) #Тренировка
    print(r2(y_test, tree.predict(x_test)))
    pass




#%%
#Нейросеть
def build_model():
  model = keras.Sequential([
    keras.layers.Dense(4096, activation=tf.nn.relu,
                       input_shape=(x_train.shape[1],)),
    keras.layers.Dense(2048, activation=tf.nn.relu),
    keras.layers.Dense(1024, activation=tf.nn.relu),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(32, activation=tf.nn.relu),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(8, activation=tf.nn.relu),
    keras.layers.Flatten(),
    keras.layers.Dense(2)
  ])
    
  lr_schedule=keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.009,
    decay_steps=1000,
    decay_rate=0.8)

  optimizer =keras.optimizers.RMSprop(learning_rate=lr_schedule)

  model.compile(loss='mse',
                optimizer=optimizer,

                metrics=['mae'])
  return model

model = build_model()
model.summary()


#%%
train_history=list()
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 50==0:
        print(str(epoch)+' epoch '+str(model.evaluate(x_test, y_test, verbose=0)[1])
        +'  '+ '   loss=' +str(sum(y_test.T[0])/sum(model.predict(x_test).T[0])-1)
        + '   r2=' +str(r2(y_test, model.predict(x_test))), end='')
        print('')
        train_history.append([epoch, sum(y_test.T[0])/sum(model.predict(x_test).T[0])-1, r2(y_test, model.predict(x_test))])
    print('.',  end='')
    

EPOCHS = 5000
#%%
history = model.fit(x_train, y_train, epochs=EPOCHS, 
                    validation_split=0.2, verbose=0,  
                    callbacks=[PrintDot()])
#%%
gr=pd.DataFrame(train_history)
plt.plot(gr[0][30:90], gr[2][30:90])

#%%
print(train_history)






    
