import argparse
import os
import random
import shutil
from tqdm import tqdm
import cv2

from config import *


def preprocess(path):
    low_res_size=int(IMAGE_SIZE*.4)
    image=cv2.imread(path)
    image=cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image=cv2.resize(image, (low_res_size,low_res_size))
    image=cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    return image 

def create_low_res_data(train_dir, low_res_dir):
    # traverse train data and create low_res data
    if os.path.exists(low_res_dir):
        return
    
    os.mkdir(low_res_dir)

    for file in tqdm(os.listdir(train_dir)):
        img_path=os.path.join(train_dir, file)
        final_path=os.path.join(low_res_dir, f"low_res_{file}")
        low_res=preprocess(img_path)
        cv2.imwrite(final_path, low_res)


def filter_subdata(path, k, train_dir):
    """
    RANDOM CHOOSING PART
    """
    directories=os.listdir(path)
    selected=[]
    # [x1, x2, x3]
    for i in range(k):
        _selected=random.choice(directories)
        selected.append(_selected)
        directories.remove(_selected)
    
    ############################################


    """
    CREATING DATASET PART
    """

    # final destination wil be -> training_data/counter.jpg

    # copy data as dataset
    os.mkdir(train_dir)
    counter=0
    for subdir in tqdm(selected):
        data_folder=os.path.join(path,subdir)
        # bmw_data/x/
        for file in tqdm(os.listdir(data_folder)):
            # bmw_data/x/y.jpg
            current_path = os.path.join(data_folder, file)
            final_dest=os.path.join(train_dir,f"{counter}.{file.split('.')[-1]}")
            
            shutil.copy(current_path, final_dest)
            

            counter+= 1

    

def main(args):
    path=args.path
    num_directories=args.num_dir
    train_dir=args.train_folder
    low_res_dir=args.low_res_folder



    if not os.path.exists(train_dir):
        filter_subdata(path, num_directories, train_dir)

    create_low_res_data(train_dir, low_res_dir)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", "-p", default=DATASET_FOLDER)
    parser.add_argument("--num_dir", "-nd", default=DATASET_FOLDER_COUNT, type=int)
    parser.add_argument("--train_folder", "-tf", default=TRAIN_HIGH_RES_DATA_FOLDER) # should run for also 'VALIDATION'
    parser.add_argument("--low_res_folder", "-bf", default=TRAIN_LOW_RES_DATA_FOLDER) # should run for also 'VALIDATION'

    args = parser.parse_args()

    main(args)