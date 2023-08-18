import os
from glob import glob
from tqdm import tqdm
import pandas as pd

def create_csv():

    train_image_cat_dir = "C:\\Users\\vchar\\OneDrive\\Desktop\ML Projects\\ComputerVision\\Cats_vs_Dogs\\train\\cat"
    train_image_dog_dir = "C:\\Users\\vchar\\OneDrive\\Desktop\ML Projects\\ComputerVision\\Cats_vs_Dogs\\train\\dog"

    cur_dir = os.getcwd()

    os.chdir(train_image_cat_dir)

    file_list = glob("*.jpg")

    class_list = ["cat" for i in range(len(file_list))]

    os.chdir(train_image_dog_dir)

    dog_file_list = glob("*.jpg")

    file_list.extend(dog_file_list)

    class_list.extend(["dog" for i in range(len(dog_file_list))])

    df = pd.DataFrame()
    df['file'] = file_list
    df['class'] = class_list

    os.chdir(cur_dir)

    df.to_csv("train.csv", index=False)


