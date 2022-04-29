from random import triangular
import cv2
import os
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import argparse

from config import *


def check_cache():
    if os.path.exists(TRAIN_HIGH_RES_SET) and os.path.exists(TRAIN_LOW_RES_SET):
        return True
    else:
        return False

def load_data(train_high_res_dir, train_low_res_dir, validation_high_res_dir, validation_low_res_dir):

    
    train_high_res_set=[]
    train_low_res_set=[]
    validation_high_res_set=[]
    validation_low_res_set=[]

    if check_cache():
        # dataset cached
        train_high_res_set=np.load(TRAIN_HIGH_RES_SET)
        train_low_res_set=np.load(TRAIN_LOW_RES_SET)
        validation_high_res_set=np.load(VALIDATION_HIGH_RES_SET)
        validation_low_res_set=np.load(VALIDATION_LOW_RES_SET)
    else:
        # dataset not cached
        # for train
        for file in tqdm(sorted(os.listdir(train_high_res_dir), key=lambda x:int(x.split(".")[0]))):    
            _path=os.path.join(train_high_res_dir,file)
            img=cv2.imread(_path).astype(np.float32)
            # convert 255 range
            img=cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img/=255.0
            train_high_res_set.append(img)
        
        for file in tqdm(sorted(os.listdir(train_low_res_dir), key=lambda x:int(x.split(".")[0].split("_")[-1]))):
            _path=os.path.join(train_low_res_dir,file)
            img=cv2.imread(_path).astype(np.float32)
            img/=255.0
            train_low_res_set.append(img)


        # for validation
        for file in tqdm(sorted(os.listdir(validation_high_res_dir), key=lambda x:int(x.split(".")[0]))):    
            _path=os.path.join(validation_high_res_dir,file)
            img=cv2.imread(_path).astype(np.float32)
            # convert 255 range
            img=cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img/=255.0
            validation_high_res_set.append(img)

        for file in tqdm(sorted(os.listdir(validation_low_res_dir), key=lambda x:int(x.split(".")[0].split("_")[-1]))):
            _path=os.path.join(validation_low_res_dir,file)
            img=cv2.imread(_path).astype(np.float32)
            img/=255.0
            validation_low_res_set.append(img)

        # train
        train_high_res_set=np.array(train_high_res_set)
        train_low_res_set=np.array(train_low_res_set)
        # validation
        validation_high_res_set=np.array(validation_high_res_set)
        validation_low_res_set=np.array(validation_low_res_set)

        # train
        np.save(TRAIN_HIGH_RES_SET, train_high_res_set)
        np.save(TRAIN_LOW_RES_SET, train_low_res_set)
        # validation
        np.save(VALIDATION_HIGH_RES_SET, validation_high_res_set)
        np.save(VALIDATION_LOW_RES_SET, validation_low_res_set)

    print(train_high_res_set.shape)
    print(train_low_res_set.shape)
    print(validation_high_res_set.shape)
    print(validation_low_res_set.shape)

    return tf.convert_to_tensor(train_high_res_set), tf.convert_to_tensor(train_low_res_set), tf.convert_to_tensor(validation_high_res_set), tf.convert_to_tensor(validation_low_res_set)



def main(args):
    train_dir=args.train_dir
    low_res_dir=args.low_res_dir
    load_data(train_dir, low_res_dir)

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dir", "-td", default=TRAIN_HIGH_RES_DATA_FOLDER) # can be also 'VALIDATION'
    parser.add_argument("--low_res_dir", "-ld", default=TRAIN_LOW_RES_DATA_FOLDER) # can be also 'VALIDATION'


    args=parser.parse_args()
    main(args)