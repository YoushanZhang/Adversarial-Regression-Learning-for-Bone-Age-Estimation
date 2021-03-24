import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0";  
import keras
from keras.models import *
from keras.layers import *
import numpy as np
from scipy.io import loadmat, savemat
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error


tf.set_random_seed(9)
np.random.seed(9)
from datetime import datetime
start=datetime.now()

def customLoss(yTrue,yPred):
    return K.abs(K.mean(yTrue) - K.mean(yPred))/K.mean(yTrue) *10 + keras.losses.mean_absolute_percentage_error(yTrue,yPred)

# make predictions from the tranined model
with tf.Graph().as_default():  
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        xx=loadmat('./boneage-test-dataset/' +'Bone_Inception_extract.mat') 
        X_test = xx['features']   
        y_all_test=xx['y']
        Y_test = np.expand_dims(y_all_test[:,0], axis=1)
        Gender_test = np.expand_dims(y_all_test[:,1], axis=1)
  
        model = tf.keras.models.load_model("best.model.h5", custom_objects={'customLoss': customLoss})

        
        preds = model.predict([X_test, Gender_test,X_test, Gender_test], batch_size=Y_test.shape[0], verbose=0)
        diff = np.expand_dims(preds[0].flatten(), axis=1) - Y_test
        print(mean_absolute_error(np.expand_dims(preds[0].flatten(), axis=1), Y_test))        
print(datetime.now()-start) 
