# -*- coding: utf-8 -*-
# author：albert time:2020/6/8
#Deep Dream图

import tensorflow as tf
import numpy as np
import IPython.display as display
import PIL.Image
from tensorflow.keras.preprocessing import image
import time

def normalize_image(img):
    img = 255 * (img + 1.0) / 2.0
    return tf.cast(img, tf.uint8)

def show_image(img):
    display.display(PIL.Image.fromarray(np.array(img)))

def save_image(img, file_name):
    PIL.Image.fromarray(np.array(img)).save(file_name)

def calc_loss(img, model):
    channel = 13
    img = tf.expand_dims(img, axis=0)
    layer_activations = model(img)
    act = layer_activations[:, :, :, channel]
    loss = tf.math.reduce_mean(act)
    return loss

def render_deepdream(model, img, steps=100, step_size=0.01, verbose=1):
    for n in tf.range(steps):
        with tf.GradientTape() as tape:
            tape.watch(img)
            loss = calc_loss(img, model)

        gradients = tape.gradient(loss, img)
        gradients /= tf.math.reduce_std(gradients) + 1e-8
        img = img + gradients * step_size
        img = tf.clip_by_value(img, -1, 1)
        if (verbose == 1):
            if ((n+1) % 10 ==0):
                print("Step {}/{}, loss {}".format(n+1, steps, loss))
    return img

base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
layer_names = 'conv2d_85'
layers = base_model.get_layer(layer_names).output
dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)

img_noise = np.random.uniform(size=(300, 300, 3)) + 100.0
img_noise = img_noise.astype(np.float32)
show_image(normalize_image(img_noise))
img = tf.keras.applications.inception_v3.preprocess_input(img_noise)
img = tf.convert_to_tensor(img)

start = time.time()
dream_img = render_deepdream(dream_model, img, steps=100, step_size=0.01)
end = time.time()
end-start

dream_img = normalize_image(dream_img)
show_image(dream_img)

file_name = 'deepdream_{}.jpg'.format(layer_names)
save_image(dream_img, file_name)
