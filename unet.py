import torch
import torch.nn as nn
from torchsummary import summary

# TODO: implement attention block
class AttentionBlock(nn.Module):
    def __init__(self):
        pass

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channesl = out_channels
    
    def forward(self, x):
        residual = x
        if self.downsample:
            residual = self.downsample(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x += residual
        x = self.relu(x)
        return x
    
if __name__ == "__main__":
    model = ResidualBlock(32, 32).to("cpu")
    # summary(model, (32, 64, 64), "cpu")
    total_params = sum([p.numel() for p in model.parameters()])
    print("Total parameters = ", total_params)
    for name, param in model.state_dict().items():
        print(name, param.size())