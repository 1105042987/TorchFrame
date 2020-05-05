import torchvision.models as models
import torch.nn as nn

def get_ResNet(name,num_classes,pretrained):
    assert name in ('resnet18','resnet34','resnet50','resnet101',
                    'resnet152','resnext50_32x4d', 'resnext101_32x8d',
                    'wide_resnet50_2', 'wide_resnet101_2')
    resnet = getattr(models,name)(pretrained=pretrained)    
    fc_features = resnet.fc.in_features
    resnet.fc = nn.Linear(fc_features, num_classes)
    return resnet
