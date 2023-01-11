import torch.nn as nn
import cv2
import torch
import numpy as np
class BorderLoss(nn.Module):
    def __init__(self, device, criterion, interation = 1, ratio = 1):
        super().__init__()
        self.criterion = criterion
        self.iteration = interation
        self.ratio = ratio
        self.device = device
    def forward(self, x, y):
        '''
        x shape: [N, H, W]
        '''
        #loss has shape N, H, W
        loss = self.criterion(x, y)
        mask = torch.zeros_like(y)
        # print(self.criterion)
        #iterate over batch size
        for i, img in enumerate(y):
            # img shape is [H, W]
            # print(img.shape)
            img = img.unsqueeze(2).detach().cpu().numpy()
            kernel = np.ones((3, 3), np.uint8)
            img_mask = (img > 0).astype(np.uint8)
            # print(img_mask.shape)
            erosion = cv2.erode(img_mask, kernel, iterations=self.iteration)
            dilation = cv2.dilate(img_mask, kernel, iterations=self.iteration)
            border = dilation - erosion
            # cv2.imwrite("debug.png", border*255)
            mask[i, :, :] = torch.tensor(border)
        mask = mask.to(self.device)
        mask = mask * self.ratio + (1-mask)
        # print(mask.shape, loss.shape)
        loss *= mask
        loss = loss.mean()
        return loss

if __name__ == '__main__':
    x = torch.zeros((1, 3, 224, 224))
    y = torch.ones((1,224,224)).long()
    criterion = nn.CrossEntropyLoss(reduction='none')
    border_criterion = BorderLoss(criterion=criterion, device=torch.device('cpu'), interation=100, ratio=2)
    print(border_criterion(x, y), criterion(x, y).mean())


            