# reimplement UNET (again ...)
# simple implementation
# no attention

import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from torchvision.transforms import Resize


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super(ResidualBlock, self).__init__()
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)

        if self.is_res:
            if self.same_channels:
                x2 = x2 + x
            else:
                x2 = x2 + x1
            x2 = x2 / np.sqrt(2)

        return x2


class DownSampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(DownSampleBlock, self).__init__()
        self.down = nn.Sequential(
            ResidualBlock(in_channels, out_channels), nn.MaxPool2d(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(UpSampleBlock, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualBlock(out_channels, out_channels),
            ResidualBlock(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, skip], 1)
        return self.up(x)


class FCEmbedding(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int) -> None:
        super(FCEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.input_dim = input_dim

        self.emb = nn.Sequential(
            nn.Linear(input_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, self.input_dim)
        return self.emb(t)


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> None:
        half_dims = self.embed_dim // 2
        # this implements 1/(10000^(k/d))
        # using the fact that e^-ln(x)=1/x
        # why? probably faster in some ways?
        # TODO: benchmark this later
        embeddings = np.log(10000) / (half_dims - 1)
        embeddings = torch.exp(
            -embeddings * torch.arange(half_dims, device=t.device)
        )  # got 1/(10000^(k/d))
        embeddings = t[:, None] * embeddings[None, :]  # got frequencies
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        # ## print(embeddings.shape)
        return embeddings

class TimeEmbedding(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, use_sinusoidal:bool=False) -> None:
        super(TimeEmbedding, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.use_sinusoidal = use_sinusoidal

        if use_sinusoidal:
            self.model = SinusoidalPositionalEmbedding(embed_dim)
        else:
            self.model = FCEmbedding(input_dim, embed_dim)

    def forward(self, t:torch.Tensor) -> None:
        return self.model(t)
    

class Unet(nn.Module):
    def __init__(self, in_channels: int, n_features: int = 256, embed_method:str="linear") -> None:
        super(Unet, self).__init__()
        self.in_channels = in_channels
        self.n_features = n_features

        self.init_conv = ResidualBlock(in_channels, n_features, is_res=True)

        self.down1 = DownSampleBlock(n_features, n_features)
        self.down2 = DownSampleBlock(n_features, n_features * 2)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())
        
        if embed_method == "linear":
            self.t_emb1 = TimeEmbedding(1, 2 * n_features, use_sinusoidal=False)
            self.t_emb2 = TimeEmbedding(1, n_features, use_sinusoidal=False)
        
        if embed_method == "cosine":
            self.t_emb1 = TimeEmbedding(1, 2 * n_features, use_sinusoidal=True)
            self.t_emb2 = TimeEmbedding(1, n_features, use_sinusoidal=True)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_features, 2 * n_features, 7, 7),
            nn.GroupNorm(8, 2 * n_features),
            nn.ReLU(),
        )

        self.up1 = UpSampleBlock(4 * n_features, n_features)
        self.up2 = UpSampleBlock(2 * n_features, n_features)

        self.out = nn.Sequential(
            nn.Conv2d(2 * n_features, n_features, 3, 1, 1),
            nn.GroupNorm(8, n_features),
            nn.ReLU(),
            nn.Conv2d(n_features, self.in_channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        t_emb1 = self.t_emb1(t).view(-1, self.n_features * 2, 1, 1)
        t_emb2 = self.t_emb2(t).view(-1, self.n_features, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(up1 + t_emb1, down2)
        up3 = self.up2(up2 + t_emb2, down1)
        out = self.out(torch.cat((up3, x), 1))

        return out
