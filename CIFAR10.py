# -*- coding: utf-8 -*-
# author：albert time:2020/5/23
# CIFAR10图像检测

import tensorflow as tf
import matplotlib.pyplot as plt

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print('training data shape:', x_train.shape)
print('training labels shape:', y_train.shape)
print('test data shape:', x_test.shape)
print('test labels shape:', y_test.shape)

label_dict = {0: 'airplane', 1: 'automobile', 2: 'birl', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse',
              8: 'ship', 9: 'truck'}

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.Dropout(rate=0.3))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.Dropout(rate=0.3))
# model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(tf.keras.layers.Dropout(rate=0.3))


model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128, activation='relu'))

model.add(tf.keras.layers.Dense(10, activation='softmax'))

print(model.summary())

train_epochs = 5
batch_size = 100

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

checkpoint_filepath = './checkpoint/Cifar10.{epoch:02d}-{val_loss:.4f}.H5'
# callbacks = [
#     tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, verbose=0, save_freq='epoch'),
#     tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
# ]

# train_history = model.fit(x_train, y_train, validation_split=0.2, epochs=train_epochs, batch_size=batch_size, callbacks=callbacks, verbose=2)
train_history = model.fit(x_train, y_train, validation_split=0.2, epochs=train_epochs, batch_size=batch_size, verbose=2)


def visu_train_history(train_history, train_metric, validation_metric):
    plt.plot(train_history.history[train_metric])
    plt.plot(train_history.history[validation_metric])
    plt.title('TH')
    plt.xlabel('epoch')
    plt.ylabel(train_metric)
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


visu_train_history(train_history, 'loss', 'val_loss')
visu_train_history(train_history, 'accuracy', 'val_accuracy')

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('Ta', test_acc)
preds = model.predict_classes(x_test)

# model_filename = "./models/cifarCNNModel.h5"
# model.save_weights(model_filename)

def plot_images_labels_prediction(images, labels, preds, index, num=5):
    fig = plt.gcf()
    fig.set_size_inches(12, 6)
    if num > 10:
        num = 10
    for i in range(0, num):
        ax = plt.subplot(2, 5, i + 1)
        ax.imshow(images[index])
        title = str(i) + ',' + label_dict[labels[index][0]]
        if len(preds) > 0:
            title += "=>" + label_dict[preds[index]]
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        index = index + 1
    plt.show()

# plot_images_labels_prediction(x_test, y_test, preds, 0, 10)