import albumentations as A
import cv2
cfg = {
    'data_path': '/mnt/luantt/polyp/datasets',
    'batch_size' : 12,
    'epochs': 150,
    'device': 'cuda',
    'lr': 5e-4,
    'img_size': (512,512),
    'seg_ratio': 4,
    'n_classes': 3,
    'model_name': 'convnext_fpn',
    'use_transformers': False,
    'log_interval': 10,
    'image_mean': (0.485,0.456,0.406),
    'image_std' : (0.229,0.224,0.225),
}
cfg['train_img_aug'] = A.Compose(
    (
        A.RGBShift(r_shift_limit=5, g_shift_limit=5, b_shift_limit=5, p = 0.5),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5, p=0.5),
        A.RandomBrightnessContrast(),
        A.GaussNoise(2),
    ) 
    )
cfg['train_map_aug'] = A.Compose(
    [
        # A.RandomScale((1, 1.5)),
        # A.RandomCrop(height=cfg['img_size'][0], width=cfg['img_size'][1], always_apply=True),
        A.RandomResizedCrop(height=cfg['img_size'][0], width = cfg['img_size'][1], scale = (0.5, 1), interpolation=cv2.INTER_NEAREST, p = 0.5),
        A.RandomRotate90(),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.OneOf([
                A.CLAHE (clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=1, shift_limit=0.5, interpolation=cv2.INTER_NEAREST, p=0.5),
        ], p=1.0),
    ]
    )
