import torch
from .metrics import *
from tqdm import tqdm
import numpy
import wandb
import pprint
import torch.nn.functional as F
import os
def start_train(model, train_loader, val_loader, opt, criterion, scheduler, cfg, criterion_aux):
    wandb.init(project="Polyp", config=cfg, name=cfg['model_name'])
    wandb.config = cfg
    device = torch.device(cfg['device'])
    model.train()
    model.to(device)
    epochs = cfg['epochs']
    best_miou = 0.
    for epoch in range(cfg['epochs']):
        print("START TRAINING EPOCH", epoch)
        epoch_loss = 0
        for i, (img, map) in enumerate(train_loader):
            # print(i)
            # break
            img = img.to(device)
            map = map.to(device)
            output = model(img)
            if cfg['use_transformers']:
                output = output.logits
            if cfg['seg_ratio'] > 1:
                output = F.interpolate(output, scale_factor = cfg['seg_ratio'], mode='nearest')
            # print(output.shape)
            # print(output.shape, map.shape)
            aux_loss = criterion_aux(output, map)
            loss = criterion(output, map) + aux_loss*0.2
            loss.backward()
            opt.step()
            opt.zero_grad()
            epoch_loss+=loss.item()
            if i % cfg['log_interval'] == 0:
                print(f'Epoch: [{epoch}/{epochs}], Lr: [{scheduler.get_last_lr()}], Step: [{i}/{len(train_loader)}], Epoch Loss : {epoch_loss/(i+1)}')
        ious, accs, dices, mloss = start_validate(model, val_loader, criterion, cfg, device)
        scheduler.step()
        infor = {
            'mDice': dices.mean(),
            'mIOU': ious.mean(), 
            'mLoss': mloss,
            'mIOU_class_1': ious[0].item(),
            'mIOU_class_2': ious[1].item(),
            'mIOU_class_3': ious[2].item(),
            # 'mIOU_class_4': ious[3].item(),
            # 'mIOU_class_5': ious[4].item(),
            # 'mIOU_class_6': ious[5].item(),
            'acc_class_1' : accs[0].item(),
            'acc_class_2' : accs[1].item(),
            'acc_class_3' : accs[2].item(),
            # 'acc_class_4' : accs[3].item(),
            # 'acc_class_5' : accs[4].item(),
            # 'acc_class_6' : accs[5].item(),
            }
        pprint.pprint(infor)
        wandb.log(infor)
        if not os.path.exists('checkpoints/'+cfg['model_name']):
            os.makedirs('checkpoints/'+cfg['model_name'])
        if (ious.mean() > best_miou):
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'cfg': cfg,
                'optimizer': opt.state_dict(),
            }, "checkpoints/{}/best_miou.ckpt".format(cfg['model_name']))
            best_miou = ious.mean()
            
        torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'cfg': cfg,
                'optimizer': opt.state_dict(),
            }, "checkpoints/{}/last.pth".format(cfg['model_name']))
def start_validate(model, val_loader, criterion, cfg, device):
    model.eval()
    total_loss = 0.
    total_intersect = torch.zeros((cfg['n_classes']))
    total_union = torch.zeros((cfg['n_classes']))
    total_area_label = torch.zeros((cfg['n_classes']))
    total_predict_area = torch.zeros((cfg['n_classes']))
    ious = torch.zeros((cfg['n_classes'],))
    i = 0
    for img, map in tqdm(val_loader):
        img = img.to(device)
        map = map.to(device)
        with torch.no_grad():
            out = model(img)
            if cfg['use_transformers']:
                out = out.logits
            if cfg['seg_ratio'] > 1:
                out = F.interpolate(out, scale_factor = cfg['seg_ratio'], mode='nearest')
            pred = out.argmax(dim=1)
            loss = criterion(out, map)
            pred = pred.cpu()
            map = map.cpu() 
            total_loss += loss.item()
            # print(pred.shape, map.shape)
            intersect, union, area_pred_label, area_label = intersect_and_union(pred, map, num_classes=cfg['n_classes'], ignore_index=255)
            total_intersect += intersect
            total_union += union
            total_predict_area += area_pred_label
            total_area_label += area_label
    ious = total_intersect/total_union
    accs = total_intersect/total_area_label
    dices = 2*total_intersect/(total_predict_area + total_area_label)
    # ious = ious.mean()
    model.train()
    return ious, accs, dices, total_loss/len(val_loader)


