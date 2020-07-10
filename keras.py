# -*- coding: utf-8 -*-
# author：albert time:2020/5/21
#keras

import numpy
import pandas as pd
from sklearn import preprocessing
import tensorflow as tf
import matplotlib.pyplot as plt

df_data = pd.read_excel('./titanic3.xls')
print(df_data)

selected_cols = ['survived', 'name', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
selected_df_data = df_data[selected_cols]

def prepare_data(df_data):
    df = df_data.drop(['name'], axis=1)
    age_mean = df['age'].mean()
    df['age'] = df['age'].fillna(age_mean)
    fare_mean = df['fare'].mean()
    df['fare'] = df['fare'].fillna(fare_mean)
    df['sex'] = df['sex'].map({'female': 0, 'male': 1}).astype(int)
    df['embarked'] = df['embarked'].fillna('S')
    df['embarked'] = df['embarked'].map({'C': 0, 'Q': 1, 'S': 2}).astype(int)

    ndarray_data = df.values
    features = ndarray_data[:, 1:]
    label = ndarray_data[:, 0]

    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    norm_features = minmax_scale.fit_transform(features)
    print(norm_features.shape)
    return norm_features, label

shuffled_df_data = selected_df_data.sample(frac=1)
x_data, y_data = prepare_data(shuffled_df_data)

train_size = int(len(x_data) * 0.8)
x_train = x_data[:train_size]
y_train = y_data[:train_size]

x_test = x_data[train_size:]
y_test = y_data[train_size:]

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=64, input_dim=7, use_bias=True, kernel_initializer='uniform', bias_initializer='zeros', activation='relu'))
model.add(tf.keras.layers.Dense(units=32, activation='sigmoid'))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

model.compile(optimizer=tf.keras.optimizers.Adam(0.003), loss='binary_crossentropy', metrics=['accuracy'])

train_history = model.fit(x=x_train, y=y_train, validation_split=0.2, epochs=100, batch_size=40, verbose=2)
print(train_history.history)
print(train_history.history.keys())

def visu_train_history(train_history, train_metric, validation_metric):
    plt.plot(train_history.history[train_metric])
    plt.plot(train_history.history[validation_metric])
    plt.title('TH')
    plt.xlabel('epoch')
    plt.ylabel(train_metric)
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

# visu_train_history(train_history, 'accuracy', 'val_accuracy')
# visu_train_history(train_history, 'loss', 'val_loss')