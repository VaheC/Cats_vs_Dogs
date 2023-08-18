import os
import shutil
from glob import glob
from tqdm import tqdm

def arrange_images():

    train_image_dir = "C:\\Users\\vchar\\OneDrive\\Desktop\ML Projects\\ComputerVision\\Cats_vs_Dogs\\train"

    cur_dir = os.getcwd()

    os.chdir(train_image_dir)

    if not os.path.exists("cat"):
        os.mkdir("cat")

    if not os.path.exists("dog"):
        os.mkdir("dog")

    file_list = glob("*.jpg")

    for filename in tqdm(file_list):

        temp_filename_list = filename.split('.')

        temp_class = temp_filename_list[0]

        temp_filename = '.'.join(temp_filename_list[1:])

        shutil.move(filename, f"{temp_class}/{temp_filename}")

    os.chdir(cur_dir)