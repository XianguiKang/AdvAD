import csv
import os

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from natsort import ns, natsorted, index_natsorted

import numpy as np
import torch as th
from tqdm import tqdm

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

class ImageNet_Compatible(Dataset):
    def __init__(self, root, image_size):
        self.image_list = []
        self.id_list = []
        self.label_ori_list = []
        self.label_tar_list = []
        self.transform = transforms.Compose([
            # transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        image_id_list, label_ori_list, label_tar_list = load_ground_truth(os.path.join(root, 'images.csv'))
        images_root = os.path.join(root, 'images')

        print("loading images and labels")
        for i in tqdm(range(len(image_id_list))):
            self.label_ori_list.append(th.from_numpy(np.array(label_ori_list[i])))
            self.label_tar_list.append(th.from_numpy(np.array(label_tar_list[i])))
            original_image = Image.open(os.path.join(images_root, image_id_list[i] + '.png')).convert('RGB')

            original_image = original_image.resize((image_size, image_size), resample=Image.LANCZOS)

            original_image = self.transform(original_image)
            self.image_list.append(original_image)
        print("done")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image = self.image_list[index]
        label_ori = self.label_ori_list[index]
        label_tar = self.label_tar_list[index]

        return image, label_ori, label_tar

if __name__ == "__main__":
    image_dataset = ImageNet_Compatible(root="ImageNet-Compatible", image_size=224)
    data_loader = DataLoader(image_dataset, batch_size=32, shuffle=False)

    for i, (images, label_ori, label_tar) in enumerate(data_loader):
        print(images.shape)
        print(label_ori.shape)
        print(label_tar.shape)

