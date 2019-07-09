###
import os
from flask import Flask, request, render_template, url_for, redirect
from werkzeug.utils import secure_filename
###
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
import faiss
###
UPLOAD_FOLDER = '/static'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

###
t1 = time.time()
model = MobileNet()
#model.summary()
model.layers.pop()
model = Model(inputs=model.inputs, outputs=model.layers[-5].output)
###
features = np.loadtxt('test.txt')
features_shape = features.astype(np.float32)
dimension = 1024
n = 95276
nlist = 50
nprobe = 10
k = 3 
###
quantiser = faiss.IndexFlatL2(dimension)  
index = faiss.IndexIVFFlat(quantiser, dimension, nlist,   faiss.METRIC_L2)
index.train(features_shape) 
index.add(features_shape)    
_files = glob.glob('static/test/*/*.jpg')
###
graph = tf.get_default_graph() 
###
def process(image):
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    with graph.as_default():
		    features_image = model.predict(image)[0]
    # features_image = model.predict(image)
    return features_image

###
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route("/")
def fileFrontPage():
    return render_template('fileform.html')

# @app.route("/handleUpload", methods=['POST'])
# def handleFileUpload():
#     static_forder = glob.glob('static/image/*.jpg')
#     for i in range(len(static_forder)):
#         os.remove(static_forder[i])
#     if 'photo' in request.files:
#         photo = request.files['photo']
#         print(photo)
#         if photo.filename != '':            
#             photo.save(os.path.join('static/image', photo.filename))
#             print(photo.filename)

    
#     return redirect(url_for('fileFrontPage'))


# @app.route("/show", methods=['GET','POST'])
# def show():
#     file_in_folder = glob.glob('static/image/*.jpg')
#     input_image = []
#     if len(file_in_folder) != 0 :
#         for i in range(len(file_in_folder)):
#             input_image.append(file_in_folder[i])
#             img = load_img(file_in_folder[i], target_size=(224, 224))
#             feature = process(img)
#         print(feature)
#     print(input_image)
#     a = np.reshape(feature, (1, -1))
#     distances, indices = index.search(a, k)
#     print(indices)
#     indices[0].sort()
#     print(indices)
#     output = []

#     for i in range(len(_files)):
#         for j in range(len(indices[0])):
#             if i == indices[0][j]:
#                 output.append(_files[i])
#     print(output)
        
#     return render_template('show.html',image=input_image[0],image0=output[0],image1=output[1],image2=output[2])

@app.route("/handleUpload", methods=['POST'])
def handleFileUpload():
    static_forder = glob.glob('static/image/*.jpg')
    for i in range(len(static_forder)):
        os.remove(static_forder[i])
    if 'photo' in request.files:
        photo = request.files['photo']
        print(photo)
        if photo.filename != '':            
            photo.save(os.path.join('static/image', photo.filename))
            print(photo.filename)
    file_in_folder = glob.glob('static/image/*.jpg')
    input_image = []
    if len(file_in_folder) != 0 :
        for i in range(len(file_in_folder)):
            input_image.append(file_in_folder[i])
            img = load_img(file_in_folder[i], target_size=(224, 224))
            feature = process(img)
        print(feature)
    print(input_image)
    a = np.reshape(feature, (1, -1))
    distances, indices = index.search(a, k)
    print(indices)
    indices[0].sort()
    print(indices)
    output = []

    for i in range(len(_files)):
        for j in range(len(indices[0])):
            if i == indices[0][j]:
                output.append(_files[i])
    print(output)
        
    return render_template('show.html',image=input_image[0],image0=output[0],image1=output[1],image2=output[2])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9000,debug=True)  