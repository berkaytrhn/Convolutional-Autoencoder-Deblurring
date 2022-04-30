from keras.models import load_model
import tensorflow as tf

from config import *
import cv2
import random

from config import *

def load_sisr_model():
    model=load_model(f"{MODEL_NAME}.h5")
    return model

def preprocess(image):
    expanded=tf.expand_dims(image, axis=0)
    tensor=tf.convert_to_tensor(expanded)
    return tensor

def predict(image, model):
    # 256x256 image received from api ready for model

    processed=preprocess(image)
    result=model.predict(processed)[0]
    # result is also normalized

    return result
