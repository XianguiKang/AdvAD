"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import csv
import os

import numpy as np
import torch as th

import datetime
import time
from tqdm import tqdm

from utils import logger
from PIL import Image

from attacked_models import model_selection
from art.estimators.classification import PyTorchClassifier
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

def eval_asr_from_npz(file_name=None, re_logger=True, quant=False, model_name=None):

    assert model_name is not None
    models_transfer_name = [model_name]

    file_dir = 'attack_results/' + file_name

    if re_logger:
        today = datetime.date.today()
        now = time.strftime("_%H%M%S")
        save_dir = file_dir + "/eval_" + (str(today).replace('-', '') + now)
        os.makedirs(save_dir, exist_ok=True)
        logger.configure(dir=save_dir)


    logger.log("******* Evaluating {} ASR ({}) *******".format(file_name, "8-bit Quanted" if quant else "Un-quanted"))

    npz_file = os.path.join(file_dir, "adversarial_examples.npz")
    npz_data = np.load(npz_file)
    all_adv_images = npz_data['arr_0']

    if quant:
        all_adv_images = all_adv_images * 255.
        all_adv_images = np.clip(np.round(all_adv_images), 0, 255) / 255.

    image_num = all_adv_images.shape[0]

    images_root = "./dataset/images/"  # The clean images' root directory.
    image_id_list, label_ori_list, label_tar_list = load_ground_truth('./dataset/images.csv')

    all_ori_labels = []
    all_tar_labels = []
    all_clean_images = []

    image_size = all_adv_images.shape[2]
    print("loading original images")
    for i in tqdm(range(image_num)):
        original_image = Image.open(images_root + image_id_list[i] + '.png').convert('RGB')
        original_image = original_image.resize((image_size, image_size), resample=Image.LANCZOS)
        original_image = np.array(original_image)
        all_clean_images.append(original_image)

        label_ori_now = label_ori_list[i]
        all_ori_labels.append(np.array(label_ori_now)[None])
        label_tar_now = label_tar_list[i]
        all_tar_labels.append(np.array(label_tar_now)[None])

    all_clean_images = np.stack(all_clean_images, axis=0)
    all_clean_images = all_clean_images.transpose(0, 3, 1, 2).astype(np.float32) / 255.0

    all_ori_labels = np.concatenate(all_ori_labels, axis=0)
    all_tar_labels = np.concatenate(all_tar_labels, axis=0)

    assert all_adv_images.shape[2] == all_adv_images.shape[3] == all_clean_images.shape[2] == all_clean_images.shape[3]
    img_res = all_adv_images.shape[2]
    logger.log("image res: {}x{}".format(img_res, img_res))
    logger.log("eval models: {}".format(models_transfer_name))

    model_transfer(
        all_clean_images,
        all_adv_images,
        all_ori_labels,
        all_tar_labels,
        res=img_res,
        models_transfer_name=models_transfer_name,
        nb_classes=1000,
    )

def model_transfer(clean_img, adv_img, label, label_tar, res, models_transfer_name, nb_classes =1000, save_path=None):

    all_clean_accuracy = []
    all_adv_accuracy = []

    for name in tqdm(models_transfer_name):
        # logger.log("*********transfer to {}********".format(name))
        model = model_selection(name)
        model.eval()

        preprocess_np = (np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]))

        f_model = PyTorchClassifier(
            model=model,
            clip_values=(0, 1),
            loss=th.nn.CrossEntropyLoss(),
            input_shape=(3, res, res),
            nb_classes=nb_classes,
            preprocessing=preprocess_np,
            device_type='gpu',
        )

        clean_pred = f_model.predict(clean_img, batch_size=50)
        accuracy = np.sum(np.argmax(clean_pred, axis=1) == label) / len(label)
        all_clean_accuracy.append(accuracy * 100)

        adv_pred = f_model.predict(adv_img, batch_size=50)
        accuracy = np.sum(np.argmax(adv_pred, axis=1) == label) / len(label)
        all_adv_accuracy.append(accuracy * 100)

    logger.log("***********************************************************")
    for i in range(len(models_transfer_name)):
        logger.log("{}\nclean acc: {:.2f}, adv acc: {:.2f}, untarget_asr: {:.2f}".format(
            models_transfer_name[i],
            all_clean_accuracy[i],
            all_adv_accuracy[i],
            100 - all_adv_accuracy[i],
        ))
    logger.log("***********************************************************")


if __name__ == "__main__":
    dir_list = [
        # "your results directory in attack_results/xxxx"
    ]
    for dir in dir_list:
        eval_asr_from_npz(file_name=dir, quant=False)
        eval_asr_from_npz(file_name=dir, quant=True)
