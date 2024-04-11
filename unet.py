import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from torchvision.transforms import Resize
import math

# TODO: implement temporal unet


class DownSampleBlock(nn.Module):
    def __init__(self, channels, use_conv=True):
        super(DownSampleBlock, self).__init__()
        self.use_conv = use_conv
        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self, x):
        N, C, H, W = x.shape
        if self.use_conv:
            x = self.conv(x)
        else:
            down = nn.AvgPool2d(kernel_size=2, stride=2)
            x = down(x)
        print(x.shape)
        assert x.shape[2] == H // 2
        assert x.shape[3] == W // 2

        return x


class UpSampleBlock(nn.Module):
    def __init__(self, channels, use_conv=True):
        super(UpSampleBlock, self).__init__()
        self.use_conv = use_conv
        if self.use_conv:
            self.conv = nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )

    def forward(self, x):
        N, C, H, W = x.shape
        resize = Resize((H * 2, W * 2))
        x = resize(x)
        if self.use_conv:
            x = self.conv(x)

        print(x.shape)
        assert x.shape[2] == H * 2
        assert x.shape[3] == W * 2

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

        attention = torch.softmax(energy / self.sqrt_dims, dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.num_heads * self.head_dim
        )
        # attention shape N x num_heads x query_len x key_len
        # values shape N x value_len x num_heads x head_dim
        # einsum shape N x query_len x num_heads x head_dim
        # can do this because value_len, key_len, query_len are all the same
        # out flatten
        out = self.fc_out(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim, stride=1):
        self.time_mlp = nn.Linear(time_embedding_dim, out_channels)
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.relu = nn.ReLU()
        self.out_channesl = out_channels
        # self.batch_norm_1 = nn.BatchNorm2d(out_channels)
        # self.batch_norm_2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, t_emb):
        # TODO: 
        # add normalization
        # add time dimension between conv1 and conv2
        residual = x 
        x = self.conv1(x)
        x = self.conv2(x)
        x += residual
        x = self.relu(x)
        return x


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.dim = dim

    def forward(self, t):
        half_dims = self.dim // 2
        # this implements 1/(10000^(k/d))
        # using the fact that e^-ln(x)=1/x
        # why? probably faster in some ways?
        # TODO: benchmark this later
        embeddings = np.log(10000) / (half_dims - 1)
        embeddings = torch.exp(
            torch.arange(-embeddings * half_dims, device=t.device)
        )  # got 1/(10000^(k/d))
        embeddings = t[:, None] * embeddings[:, None]  # got frequencies
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings

class UNet(nn.Module):
    def __init__(
        self,
        init_channels,
        in_channels,
        out_channels,
        num_res_blocks,
        attn_resolutions,
        input_img_resolution,
        channels_multipliers=(1, 2, 4, 8),
        dropout=0.0,
        resamp_with_conv=True,
    ):
        super(UNet, self).__init__()
        self.init_channels = init_channels
        self.time_emb_size = self.init_channels * 4
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.input_img_resolution = input_img_resolution
        self.channels_multipliers = channels_multipliers
        self.dropout = dropout
        self.resamp_with_conv = resamp_with_conv

        # DEFINE NETWORKS
        # STAGE 0: Time embedding
        self.t_emb = nn.Sequential(
            SinusoidalPositionalEmbedding(self.time_emb_size),
            nn.Linear(self.time_emb_size, self.time_emb_size),
            nn.ReLU(),
        )

        # STAGE 1: Downsampling
        self.conv_init = nn.Conv2d(
            in_channels=in_channels,
            out_channels=init_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        current_res = input_img_resolution
        in_channels_multipliers = (1,) + self.channels_multipliers
        self.down = nn.ModuleList()
        for i in range(len(self.channels_multipliers)):
            res_blocks = nn.ModuleList()
            attn_blocks = nn.ModuleList()
            block_dim_in = self.init_channels * in_channels_multipliers[i]
            block_dim_out = self.init_channels * self.channels_multipliers[i]
            for _ in range(self.num_res_blocks):
                res_blocks.append(ResidualBlock(block_dim_in, block_dim_out))
                block_dim_in = block_dim_out
                if current_res in attn_resolutions:
                    attn_blocks.append(SelfAttentionBlock(block_dim_in, 1))
            level_objects = nn.Module()
            level_objects.res_blocks = res_blocks
            level_objects.attn_blocks = attn_blocks 
            if i < len(self.channels_multipliers)-1:
                level_objects.scale = DownSampleBlock(block_dim_in, use_conv=True)
                current_res = current_res // 2
            self.down.append(level_objects)

        # STAGE 2: Middle
        # TODO: finish this stage
        self.middle = nn.Module()
        self.middle.res_block_1 = ResidualBlock()

if __name__ == "__main__":
    # model = ResidualBlock(32, 32).to("cpu")
    # total_params = sum([p.numel() for p in model.parameters()])
    # print("Total parameters = ", total_params)
    # for name, param in model.state_dict().items():
    #     print(name, param.size())

    # model = SelfAttentionBlock(128, 2).to("cpu")
    # total_params = sum([p.numel() for p in model.parameters()])
    # print("Total parameters = ", total_params)
    # for name, param in model.state_dict().items():
    #     print(name, param.size())

    test = torch.randn(size=(8, 32, 256, 256))
    downsample = DownSampleBlock(channels=32, use_conv=True)
    test = downsample(test)
    print(test.shape)

    test = torch.randn(size=(8, 32, 128, 128))
    upsample = UpSampleBlock(channels=32, use_conv=True)
    test = upsample(test)
    print(test.shape)
