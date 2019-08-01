import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet


class CEfficientNet(nn.Module):
    def __init__(self, num_classes=10, pretrained=False, model_name='efficientnet-b0'):
        super(CEfficientNet, self).__init__()
        if pretrained:
            self.features = EfficientNet.from_pretrained(model_name)
        else:
            self.features = EfficientNet.from_name(model_name)
        self.features._conv_stem.stride = (1, 1)
        fc_features = self.features._fc.in_features  
        self.features._fc = nn.Linear(fc_features, 10)  
    
    def forward(self, x):
        out = self.features(x)
        return out

if __name__ == "__main__":
    x = torch.randn(10, 3, 32, 32)
    net = CEfficientNet(10, True, model_name='efficientnet-b0')
    print(net)
    out = net(x)
    print(out.shape)