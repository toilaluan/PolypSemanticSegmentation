from dataset import DroneDataset
import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.utils.data import DataLoader
from run import *
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation, MaskFormerForInstanceSegmentation
import albumentations as A
from config import cfg
from criterion import *
from FPN import FPNModel
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
# model = DeeplabV3plus(num_classes=6)
print(cfg)
# model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", num_labels=cfg['n_classes'], num_channels=3, ignore_mismatched_sizes=True)
model = FPNModel(cfg['n_classes'], 256)
model = nn.DataParallel(model)
cfg['model_name'] = 'convnext_fpn_aspp'
cfg['use_transformers'] = False
# print(model)
print(count_parameters(model ))
train_data = DroneDataset(cfg['data_path'], img_size=cfg['img_size'], seg_ratio = cfg['seg_ratio'], mode='train', cfg=cfg)
val_data = DroneDataset(cfg['data_path'], mode='valid', img_size=cfg['img_size'], seg_ratio = cfg['seg_ratio'], cfg=cfg)
train_loader = DataLoader(train_data, cfg['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_data, 1, num_workers=4, pin_memory=True)
criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
criterion = BorderLoss(device=torch.device(cfg['device']), criterion=criterion, interation=2, ratio=1.5)
criterion_aux = DiceLoss(mode='multiclass')
opt = torch.optim.Adam(model.parameters(), lr = cfg['lr'], weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt,0.95)
start_train(model, train_loader, val_loader, opt, criterion, scheduler, cfg, criterion_aux=criterion_aux)
