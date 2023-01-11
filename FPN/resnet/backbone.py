import timm
from torchvision.models._utils import IntermediateLayerGetter
def build_backbone():
    model = timm.create_model('resnet50', pretrained=True)
    # print(model)
    model = IntermediateLayerGetter(model, {
        'layer1': 'layer1',
        'layer2' : 'layer2',
        'layer3': 'layer3',
        'layer4' : 'layer4'
    })
    return model

if __name__ == '__main__':
    import torch
    x = torch.zeros((1,3,224,224))
    model = build_backbone()
    out = model(x)
    for k, v in out.items():
        print(k, v.shape)