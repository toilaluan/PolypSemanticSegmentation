import torch
from torch.utils.data import Dataset
import os
import glob
import cv2
import albumentations as A
import config as cfg
from PIL import Image 
import numpy as np
import torchvision.transforms as T
from config import cfg

class DroneDataset(Dataset):
    def __init__(self, data_path, img_size, seg_ratio, cfg, mode = 'train'):
        self.mode = mode
        self.seg_ratio = seg_ratio
        self.img_size = img_size
        self.data_path = data_path
        self.img_list = self.load_txt(os.path.join(data_path, f'{mode}.txt'))
        self.img_aug = cfg['train_img_aug']
        self.map_aug = cfg['train_map_aug']
        self.transform = T.Compose([
            T.ToTensor(),
            # T.RandomCrop(img_size, pad_if_needed=True, fill=255),
            T.Normalize(cfg['image_mean'], cfg['image_std'])
        ])
        self.ele_transform = T.Compose([
            T.ToTensor(),
            # T.Normalize(0.5, 0.5)
        ])
    def __len__(self):
        return len(self.img_list)
    def read_mask(self, mask_path):
        image = cv2.imread(mask_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # lower boundary RED color range values; Hue (0 - 10)
        lower1 = np.array([0, 100, 20])
        upper1 = np.array([10, 255, 255])
        # upper boundary RED color range values; Hue (160 - 180)
        lower2 = np.array([160,100,20])
        upper2 = np.array([179,255,255])
        lower_mask = cv2.inRange(image, lower1, upper1)
        upper_mask = cv2.inRange(image, lower2, upper2)

        red_mask = lower_mask + upper_mask
        red_mask[red_mask != 0] = 2
        
        # boundary RED color range values; Hue (36 - 70)
        green_mask = cv2.inRange(image, (36, 25, 25), (70, 255,255))
        green_mask[green_mask != 0] = 1
        
        full_mask = cv2.bitwise_or(red_mask, green_mask)
        full_mask = full_mask.astype(np.uint8)
        # cv2.imwrite('debug.png', (full_mask*50).astype(np.uint8))
        
        return full_mask
    def __getitem__(self, index):
        img_name = self.img_list[index].rstrip().split('.')[0]
        img_path = os.path.join(self.data_path, f'train/{img_name}.jpeg')
        seg_path = os.path.join(self.data_path, f'train_gt/{img_name}.jpeg')

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_NEAREST)
        # seg = cv2.imread(seg_path)[:,:,0]
        seg = self.read_mask(seg_path)
        # seg = cv2.resize(seg, (self.img_size[0]//self.seg_ratio, self.img_size[0]//self.seg_ratio), interpolation=cv2.INTER_NEAREST)
        seg = cv2.resize(seg, (self.img_size), interpolation=cv2.INTER_NEAREST)

        if self.mode == 'train':
            img = self.img_aug(image = img)['image']
            transformed = self.map_aug(image = img, mask=seg)
            img = transformed['image']
            seg = transformed['mask']
        # cv2.imwrite('debug.png', img)
        img = self.transform(img)
        # print(img)
        seg = torch.tensor(seg).long()
        return img , seg
    def load_txt(self, path):
        with open(path, 'r') as f:
            data = f.readlines()
        return data
    def visualize(self, img, seg):
        cv2.imwrite("visualize/debug.png", img)
if __name__ == "__main__":
    data_path = '/home/kc/luantt/kaggle_data/dataset-medium'
    dataset = DroneDataset(data_path=data_path, mode='train', img_size=(224,224), seg_ratio=1, cfg=cfg)
    dataset[10]