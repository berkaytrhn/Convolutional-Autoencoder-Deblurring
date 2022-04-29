from keras.models import load_model
import tensorflow as tf

from config import *
import argparse

import cv2
import random



def preprocess(image):    
    height, width, _= image.shape

    row = random.randint(0, (height-IMAGE_SIZE))
    col = random.randint(0, (width-IMAGE_SIZE))


    smaller_size=int(IMAGE_SIZE*.4)
    cropped=tf.image.resize(image[row:row+IMAGE_SIZE,col:col+IMAGE_SIZE], (smaller_size, smaller_size))
    cropped=tf.image.resize(cropped, (IMAGE_SIZE, IMAGE_SIZE), method=tf.image.ResizeMethod.BICUBIC)
    cropped=cropped.numpy()
    cropped/=255.0

    cv2.imshow("Low Resolution Image", cropped)
    return tf.convert_to_tensor(tf.expand_dims(cropped, axis=0))

def predict(img, model):
    cv2.imshow("Original Image", img)
    img=preprocess(img)
    res=model.predict(img)[0]
    
    print(res.shape)
    cv2.imshow("Reconstructed", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main(args):
    model=load_model(f"{MODEL_NAME}.h5")

    img_path=args.path
    img=cv2.imread(img_path)
    predict(img, model)
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # test image path
    parser.add_argument("--path", "-p", required=True)

    args = parser.parse_args()

    main(args)