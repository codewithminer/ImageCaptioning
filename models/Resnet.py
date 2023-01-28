import torch
import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    """Load the pretrained Resnet-152 and replace top fc layer"""
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResNet, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)   # It “chains” outputs to inputs sequentially for each subsequent module, finally returning the output of the last module.
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)  # features size (1, 2048, 1, 1)
        features = features.reshape(features.size(0), -1)   # [x,y ... , z] [1, 2048]
        features = self.bn(self.linear(features))   # [1, 256]
        return features

