import torch.nn as nn
import torch.nn.functional as F
import torch
from .resnet import build_backbone as resnet
from .convnext import build_backbone as convnext
from .aspp import ASPP
from .modules import DeformableConv2d
class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                                stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        out = self.relu(x)
        return out
class BasicDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,dilation=1, bias=True):
        super(BasicDown, self).__init__()
        self.sub1 = BasicBlock(in_channels, out_channels*2, 3, stride, 1, dilation, bias)
        self.de_conv = DeformableConv2d(out_channels*2, out_channels, 3, 1, 1, bias)
        # self.sub2 = BasicBlock(out_channels*2, out_channels, 1, 1, 0, 1, bias)
    def forward(self, x):
        x = self.sub1(x)
        x = self.de_conv(x)
        # x = self.sub2(x)
        return x
class FPN(nn.Module):
    def __init__(self, in_features, out_feature):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_feature
        self.top_layer = BasicDown(in_features[-1], out_feature, kernel_size=1)
        self.down_layer1 = BasicDown(in_features[-2], out_feature, kernel_size=1)
        self.down_layer2 = BasicDown(in_features[-3], out_feature, kernel_size=1)
        self.down_layer3 = BasicDown(in_features[-4], out_feature, kernel_size=1)

    def _upsample_add(self, x, y):
        x = F.interpolate(x, size=y.shape[2:], mode='bilinear')
        return x + y
    def forward(self, x):
        '''
        x is features 
        '''        
        x = list(x.values())
        # print(x)
        p5 = self.top_layer(x[-1])
        p4 = self._upsample_add(p5, self.down_layer1(x[-2]))
        p3 = self._upsample_add(p4, self.down_layer2(x[-3]))
        p2 = self._upsample_add(p3, self.down_layer3(x[-4]))
        return p2
class FPNModel(nn.Module):
    def __init__(self, n_classes, dim = 256):
        super().__init__()
        self.backbone = convnext()
        in_features = self._get_in_features()
        self.fpn = FPN(in_features, dim)
        # self.fpn = FeaturePyramidNetwork(in_channels_list=in_features, out_channels=dim)
        # self.head = ASPP(num_classes=n_classes, in_channels=dim, dim=dim)
        self.head = nn.Conv2d(in_channels=dim, out_channels=n_classes, kernel_size=1)
    def _get_in_features(self):
        in_features = []
        with torch.no_grad():
            x = torch.zeros((1,3,224,224))
            out = self.backbone(x)
            out = list(out.values())
            for l in out:
                in_features.append(l.shape[1])
        return in_features
    def forward(self, x):
        x = self.backbone(x)
        x = self.fpn(x)
        # for k, v in x.items():
        #     print(v.shape)
        # # print(x.shape)
        x = self.head(x)
        return x
if __name__ == '__main__':
    
    x = torch.zeros((1,3,224,224))
    model = FPNModel(3, 256)
    print(model(x).shape)
    
