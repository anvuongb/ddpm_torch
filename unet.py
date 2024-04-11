import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from torchvision.transforms import Resize


# TODO: implement temporal unet

class DownSampleBlock(nn.Module):
    def __init__(self):
        super(DownSampleBlock, self).__init__()

    def forward(self, x, use_conv=True):
        N, C, H, W = x.shape
        if use_conv:
            down = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=3, stride=2)
        else:
            down = nn.AvgPool2d(kernel_size=2, stride=2)
        x = down(x)

        assert x.shape[2] == H//2
        assert x.shape[3] == W//2

        return x

class UpSampleBlock(nn.Module):
    def __init__(self):
        super(UpSampleBlock, self).__init__()

    def forward(self, x, use_conv=True):
        N, C, H, W = x.shape
        resize = Resize((H*2, W*2))
        x = resize(x)
        if use_conv:
            conv = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=3, stride=1)
            x = conv(x)

        assert x.shape[2] == H*2
        assert x.shape[3] == W*2

        return x
class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_size=512, num_heads=8):
        super(SelfAttentionBlock, self).__init__()
        assert embed_size % num_heads == 0
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = int(embed_size / num_heads)
        self.sqrt_dims = np.sqrt(embed_size)
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.fc_out = nn.Linear(self.head_dim * self.num_heads, self.embed_size)   
    
    def forward(self, x):
        N, C, H, W = x.shape
        value_len, key_len, query_len = x.shape[1], x.shape[1], x.shape[1]

        # normalizing
        layer_norm = nn.LayerNorm([C, H, W])
        x = layer_norm(x)

        # split into heads
        values = x.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = x.reshape(N, key_len, self.num_heads, self.head_dim)
        queries = x.reshape(N, query_len, self.num_heads, self.head_dim)       

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape N x query_len x num_heads x head_dim
        # keys shape N x key_len x num_heads x head_dim
        # energy shape N x num_heads x query_len x key_len

        attention = torch.softmax(energy/self.sqrt_dims, dim=3)
        out = torch.einsum("nhql,nlhd->nqhd",[attention, values]).reshape(N, query_len, self.num_heads * self.head_dim)
        # attention shape N x num_heads x query_len x key_len
        # values shape N x value_len x num_heads x head_dim
        # einsum shape N x query_len x num_heads x head_dim
        # can do this because value_len, key_len, query_len are all the same
        # out flatten
        out = self.fc_out(out)
        return out
    
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
    total_params = sum([p.numel() for p in model.parameters()])
    print("Total parameters = ", total_params)
    for name, param in model.state_dict().items():
        print(name, param.size())

    model = SelfAttentionBlock(128, 2).to("cpu")
    total_params = sum([p.numel() for p in model.parameters()])
    print("Total parameters = ", total_params)
    for name, param in model.state_dict().items():
        print(name, param.size())