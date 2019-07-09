from keras.preprocessing import image
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
import os
import numpy as np
import tensorflow as tf
import keras
import glob
import time

t1 = time.time()
model = MobileNet()
model.summary()
model.layers.pop()
model = Model(inputs=model.inputs, outputs=model.layers[-5].output)

def process(image):
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    features = model.predict(image)
    return features

features = []

files = glob.glob('/home/duc-dn/test/*/*.jpg')
for i in range(len(files)):
    try:
        image = load_img(files[i], target_size=(224, 224))
        feature = process(image)
        print(i)
        features.append(feature[0])
    except :
        os.remove(files[i])

features = np.transpose(features)
features_shape = np.transpose(features)

np.savetxt('test.txt',features_shape)
print('Time run:')
print(time.time()-t1)