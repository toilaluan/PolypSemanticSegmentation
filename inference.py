import cv2
import torch
from FPN import FPNModel
import glob
from config import cfg
import os
import torch.nn as nn
from torchvision import transforms as T
import numpy as np
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation, MaskFormerForInstanceSegmentation
def load_checkpoint(model, path):
    state_dict = torch.load(path)
    # print(len(state_dict))
    new_state_dict = {}
    for k,v in state_dict.items():
        new_state_dict[k[7:]] = v
    model.load_state_dict(new_state_dict)

total_img_dir = glob.glob("/mnt/luantt/polyp/datasets/test/*")
checkpoint_path = '/mnt/luantt/polyp/checkpoints/convnext_fpn_gelu/best_miou.pth'
out_dir = 'output/'
device = torch.device('cuda')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
# model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", num_labels=3, ignore_mismatched_sizes=True)
model = FPNModel(cfg['n_classes'], 256)
model = nn.DataParallel(model)
model.load_state_dict(torch.load(checkpoint_path))
model.to(device)
model.eval()
count_1 = 0
count_0 = 0
for img_dir in total_img_dir:
    img_name = img_dir.split('/')[-1].split('.')[0]
    img = cv2.imread(img_dir)
    ori_h, ori_w, _ = img.shape
    img = cv2.resize(img, cfg['img_size'], interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = T.Compose([
            T.ToTensor(),
            T.Normalize(cfg['image_mean'],cfg['image_std'])
        ])
    img = transform(img).to(device) # C, H, W
    img = img.unsqueeze(0) # 1, C, H, W
    # img = torch.concat([img,img], dim=0)
    # print(img.shape)
    with torch.no_grad():
        output = model(img)# 1, 3, H, W
        # pred = output.logits.argmax(1) # 1, H, W
        pred = output.argmax(1)
        # print(pred.shape)
        # pred = pred.permute(2, 3, 1, 1).squeeze(-1)
        pred = T.Resize((ori_h, ori_w), interpolation=T.InterpolationMode.NEAREST)(pred)
        blank = torch.zeros((3, ori_h, ori_w))
        one_label = torch.where(pred==1, 1, 0)
        zero_label = torch.where(pred==2, 1, 0)
        print("One", one_label.sum())
        print("Zero", zero_label.sum())
        if (one_label.sum() > 0):
            count_1 += 1
        if zero_label.sum() > 0:
            count_0 += 1
        blank[0,:,:] = zero_label
        blank[1,:,:] = one_label
        blank = blank.permute(1,2,0)
        blank = blank.numpy().astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, img_name)+'.png', blank)
print(count_0, count_1)




