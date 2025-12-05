import torchvision.models as models
import torch.nn as nn


class DebrisClassifier(nn.Module):
    def __init__(self, backbone='resnet34', pretrained=True):
        super(DebrisClassifier, self).__init__()
        if backbone == 'resnet34':
            self.encoder = models.resnet34(pretrained=pretrained)
        elif backbone == 'resnet50':
            self.encoder = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError("Unsupported backbone")

        # Modify the final fully connected layer for binary classification
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, 1)  # Binary output
        self.sigmoid = nn.Sigmoid()  # Apply sigmoid to the output

    def forward(self, x):
        x = self.encoder(x)
        x = self.sigmoid(x)  # Sigmoid to output probability for binary classification
        return x