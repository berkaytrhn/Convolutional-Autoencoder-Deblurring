from keras.models import load_model

from config import *

def load_sisr_model():
    model=load_model(f"{MODEL_NAME}.h5")
    return model
