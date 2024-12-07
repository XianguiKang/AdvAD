"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import csv
import os

import numpy as np

import datetime
import time
from tqdm import tqdm

from pytorch_fid_score_new.fid_score_new import return_fid_from_data
from utils import logger
from PIL import Image

from natsort import index_natsorted

##load image metadata (Image_ID, true label, and target label)
def load_ground_truth(csv_filename):
    image_id_list = []
    label_ori_list = []
    label_tar_list = []

    with open(csv_filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            image_id_list.append(row['ImageId'])
            label_ori_list.append(int(row['TrueLabel']) - 1)
            label_tar_list.append(int(row['TargetClass']) - 1)

    sorted_indices = index_natsorted(image_id_list)
    image_id_list = [image_id_list[i] for i in sorted_indices]
    label_ori_list = [label_ori_list[i] for i in sorted_indices]
    label_tar_list = [label_tar_list[i] for i in sorted_indices]

    return image_id_list, label_ori_list, label_tar_list

def count_images_in_directory(directory):
    files = os.listdir(directory)
    file_count = len([f for f in files if os.path.isfile(os.path.join(directory, f))])
    return file_count

def eval_fid_from_npz(file_name=None, re_logger=True, quant=False):

    file_dir = 'attack_results/' + file_name
    # file_dir = 'visualization/' + file_name
    if re_logger:
        today = datetime.date.today()
        now = time.strftime("_%H%M%S")
        save_dir = file_dir + "/eval_" + (str(today).replace('-', '') + now)
        os.makedirs(save_dir, exist_ok=True)
        logger.configure(dir=save_dir)

    logger.log("******* Evaluating {} FID ({}) *******".format(file_name, "8-bit Quanted" if quant else "Un-quanted"))

    npz_file = os.path.join(file_dir, "adversarial_examples.npz")
    npz_data = np.load(npz_file)
    all_adv_images = npz_data['arr_0']
    image_num = all_adv_images.shape[0]

    if quant:
        all_adv_images = all_adv_images * 255.
        all_adv_images = np.clip(np.round(all_adv_images), 0, 255) / 255.

    images_root = "./dataset/images/"  # The clean images' root directory.
    image_id_list, label_ori_list, label_tar_list = load_ground_truth('./dataset/images.csv')

    image_size = all_adv_images.shape[2]

    all_clean_images = []
    # print("loading original images")
    for i in tqdm(range(image_num)):
        original_image = Image.open(images_root + image_id_list[i] + '.png').convert('RGB')
        original_image = original_image.resize((image_size, image_size), resample=Image.LANCZOS)
        original_image = np.array(original_image)
        all_clean_images.append(original_image)
    all_clean_images = np.stack(all_clean_images, axis=0)
    all_clean_images = all_clean_images.transpose(0, 3, 1, 2).astype(np.float32) / 255.0

    fid = return_fid_from_data(all_adv_images, all_clean_images)
    logger.log("FID: {}".format(fid))
    logger.log("******* Done *******")

if __name__ == "__main__":
    dir_list = [
        # "your results directory in attack_results/xxxx"
    ]
    for dir in dir_list:
        eval_fid_from_npz(file_name=dir, re_logger=True, quant=False)
        eval_fid_from_npz(file_name=dir, re_logger=True, quant=True)
