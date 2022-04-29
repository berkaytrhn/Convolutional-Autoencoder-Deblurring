from keras.models import load_model
import tensorflow as tf

from config import *
import cv2
import random


def preprocess(image):
    expanded=tf.expand_dims(image, axis=0)
    tensor=tf.convert_to_tensor(expanded)
    return tensor

def predict(image, model):
    # 256x256 image received from api ready for model


    processed=preprocess(image)
    result=model.predict(processed)[0]
    # result is also normalized
    cv2.imshow("test_final", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return result
