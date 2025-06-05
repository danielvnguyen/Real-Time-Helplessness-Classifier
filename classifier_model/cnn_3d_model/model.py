# 3D CNN Model
import torch.nn as nn
from torchinfo import summary

class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, double=False):
        super(Conv3DBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential() if not double else nn.Sequential(
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.MaxPool3d(kernel_size=2)
        self.norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.norm(x)
        return x

class HelplessnessClassifier(nn.Module):
    def __init__(self, input_channels=3):
        super(HelplessnessClassifier, self).__init__()

        # Sequence of 3D Convolutions
        self.cnn = nn.Sequential(
            Conv3DBlock(in_channels=input_channels, out_channels=32, double=True),
            Conv3DBlock(in_channels=32, out_channels=64, double=False),
            Conv3DBlock(in_channels=64, out_channels=128, double=False),
            Conv3DBlock(in_channels=128, out_channels=256, double=False),
            # Conv3DBlock(in_channels=256, out_channels=512, double=False),
        )

        # Final avg pool downsampling before fully connected layer
        self.avg_pool = nn.AdaptiveAvgPool3d(4)

        # Fully connected layer for classification
        self.fc = nn.Sequential(
            nn.Linear(in_features=256*4*4*4, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1) # flatten the input
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = HelplessnessClassifier().cuda()
    summary(model, (1, 3, 90, 224, 224))
