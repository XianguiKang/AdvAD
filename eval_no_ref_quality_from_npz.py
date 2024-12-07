import torch
import argparse
import numpy as np 
from PIL import Image
import random
import torchvision.transforms as transforms
import torchvision
import torch.utils.data as data
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import trange, tqdm
import numpy
import numpy as np
import math
import pyiqa
from utils import logger


import datetime
import time

parser = argparse.ArgumentParser(description='Test Image Quality!')
parser.add_argument('--img_path', type=str, default='./', help='cnn')
parser.add_argument('--metric', type=str, default='musiq-koniq', help='cnn')
parser.add_argument('--seed', type=int, default=42, help='cnn')


args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def img_loader(path):
    try:
        with open(path, 'rb') as f:
            img = Image.open(path).convert('RGB')
            return img
    except IOError:
        print('Cannot load image ' + path)

def eval_no_ref_quality_from_npz(file_name=None, re_logger=True, quant=False):
    file_dir = 'attack_results/' + file_name
    if re_logger:
        today = datetime.date.today()
        now = time.strftime("_%H%M%S")
        save_dir = file_dir + "/eval_" + (str(today).replace('-', '') + now)
        os.makedirs(save_dir, exist_ok=True)
        logger.configure(dir=save_dir)

    npz_file = os.path.join(file_dir, "adversarial_examples.npz")
    npz_data = np.load(npz_file)
    all_adv_images = npz_data['arr_0']
    all_adv_images = all_adv_images.transpose(0, 2, 3, 1) * 255.
    all_adv_images = np.clip(all_adv_images, 0, 255)

    if quant:
        all_adv_images = np.round(all_adv_images)

    all_adv_images = torch.from_numpy(all_adv_images).cuda()

    image_num = all_adv_images.shape[0]

    metrics = ['musiq-koniq']
    all_res = []
    lower_better = []

    logger.log("******* Evaluating {} No-ref quality ({}) *******".format(file_name, "8-bit Quanted" if quant else "Un-quanted"))

    for metric in metrics:
        iqa_metric = pyiqa.create_metric(metric, device=device)
        lower_better.append(iqa_metric.lower_better)
        res_now = 0
        for i in tqdm(range(image_num)):
            # img2 = torch.Tensor(np.array(img_loader(os.path.join(img2_root, f_list[i])).resize((224, 224)))).unsqueeze(0).to(device).permute(0, 3, 1, 2)/255.0
            img2 = all_adv_images[i].unsqueeze(0).permute(0, 3, 1, 2) / 255.0
            score_now = iqa_metric(img2)
            res_now += score_now
        res = res_now.item()/image_num
        all_res.append(res)

    for i in range(len(metrics)):
        logger.log("{} ({}): {:.4f}".format(metrics[i], lower_better[i], all_res[i]))
        # print('name', res_now/len(f_list))


if __name__ == "__main__":
    dir_list = [
        # "your results directory in attack_results/xxxx"
    ]
    for dir in dir_list:
        eval_no_ref_quality_from_npz(file_name=dir, re_logger=True, quant=False)
        eval_no_ref_quality_from_npz(file_name=dir, re_logger=True, quant=True)


