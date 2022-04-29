from keras.layers import Input, Conv2D, MaxPool2D, Conv2DTranspose, Add
from keras.models import Model
from keras.regularizers import L1

from config import *

def auto_encoder(regularization_coef=1e-10): #1e-5
    bias_coef=regularization_coef*100

    _input = Input(shape=(IMAGE_SIZE,IMAGE_SIZE,3))


    ####### ENCODER PART #######

    # 256 -> 128
    layer1=Conv2D(
        64, 
        kernel_size=(3,3), 
        padding="same", 
        activation="relu", 
        activity_regularizer=L1(l1=regularization_coef)
    )(_input)
    layer2=Conv2D(
        64, 
        kernel_size=(3,3), 
        padding="same", 
        activation="relu", 
        activity_regularizer=L1(l1=regularization_coef)
    )(layer1)
    layer3=MaxPool2D(padding="same")(layer2)

    # 128 -> 64
    layer4=Conv2D(
        128, 
        kernel_size=(3,3), 
        padding="same", 
        activation="relu", 
        activity_regularizer=L1(l1=regularization_coef)
    )(layer3)
    layer5=Conv2D(
        128, 
        kernel_size=(3,3), 
        padding="same", 
        activation="relu", 
        activity_regularizer=L1(l1=regularization_coef)
    )(layer4)
    layer6=MaxPool2D(padding="same")(layer5)

    # middle
    layer7=Conv2D(
        256, 
        kernel_size=(3, 3), 
        padding='same', 
        activation='relu', 
        activity_regularizer=L1(l1=regularization_coef)
    )(layer6)


    ####### DECODER PART #######
    
    
    # 64 -> 128
    #layer8=UpSampling2D()(layer76)
    layer8 = Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), padding="same")(layer7)
    layer9=Conv2D(
        128,
        kernel_size=(3,3),
        padding="same",
        activation="relu",
        activity_regularizer=L1(l1=regularization_coef)
    )(layer8)
    layer10=Conv2D(
        128,
        kernel_size=(3,3),
        padding="same",
        activation="relu",
        activity_regularizer=L1(l1=regularization_coef)
    )(layer9)
    layer11=Add()([layer5, layer10])

    # 128 -> 256
    #layer12=UpSampling2D()(layer11)
    layer12 = Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), padding="same")(layer11)
    layer13=Conv2D(
        64,
        kernel_size=(3,3),
        padding="same",
        activation="relu",
        activity_regularizer=L1(l1=regularization_coef)
    )(layer12)
    layer14=Conv2D(
        64,
        kernel_size=(3,3),
        padding="same",
        activation="relu",
        activity_regularizer=L1(l1=regularization_coef)
    )(layer13)
    layer15=Add()([layer14,layer2])

    # output layer
    res=Conv2D(
        3,
        kernel_size=(3,3),
        padding="same",
        activation="relu",
        activity_regularizer=L1(l1=regularization_coef)
    )(layer15)

    return Model(_input,res)
