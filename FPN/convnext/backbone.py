import timm
import torch.nn as nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
def build_backbone():
    # print(timm.list_models('*convnext'))
    model = timm.create_model('convnext_tiny', pretrained=True)
    return_nodes = {
        'stages.0': 'layer1',
        'stages.1': 'layer2',
        'stages.2': 'layer3',
        'stages.3': 'layer4'
    }
    model = create_feature_extractor(model, return_nodes=return_nodes)
    return model
if __name__ == '__main__':
    import torch
    model = build_backbone()
    x = torch.zeros((1,3,224,224))
    out = model(x)
    for k, v in out.items():
        print(v.shape)