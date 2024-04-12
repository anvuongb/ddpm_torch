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
    def __init__(self, in_channels, num_heads=8):
        super(SelfAttentionBlock, self).__init__()

        self.norm = nn.GroupNorm(
            num_channels=in_channels, num_groups=32, eps=1e-6, affine=True
        )

        self.conv_values = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            stride=1,
            kernel_size=1,
            padding=0,
        )
        self.conv_keys = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            stride=1,
            kernel_size=1,
            padding=0,
        )
        self.conv_queries = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            stride=1,
            kernel_size=1,
            padding=0,
        )

        self.conv_out = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            stride=1,
            kernel_size=1,
            padding=0,
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(x)
        q = self.conv_queries(h_)
        k = self.conv_keys(h_)
        v = self.conv_values(h_)

        # compute QK
        N, C, H, W = q.shape
        q = q.reshape(N, C, H * W)  # N x C x HW
        k = k.reshape(N, C, H * W)  # N x C x HW
        w_qk = torch.einsum("ncq,nck->nqk", [q, k])  # N x HW x HW
        w_qk *= int(C) ** (-0.5)  # scaled-dot sqrt(dim)
        w_qk = torch.nn.functional.softmax(w_qk, dim=2)

        # attend to V
        v = v.reshape(N, C, H * W)  # N x C x HW
        attn = torch.einsum("ncq,nqk->ncq", [v, w_qk])
        attn = attn.reshape(N, C, H, W)
        attn = self.conv_out(attn)

        return attn


# TODO: either this is wrong, or the construction of UNet is wrong, something is off
class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, time_embedding_dim, stride=1, dropout=0.1
    ):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        print("in_channels, out_channels", self.in_channels, self.out_channels)
        self.time_mlp = nn.Linear(time_embedding_dim, out_channels)
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
        self.batch_norm_1 = nn.BatchNorm2d(in_channels)
        self.batch_norm_2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.skip_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x, t_emb):
        t_emb = self.time_mlp(t_emb)

        residual = x
        print("x_shape", x.shape)
        x = self.batch_norm_1(x)
        x = self.conv1(x)
        t_emb = t_emb[:, :, None, None]
        x = x + t_emb

        x = self.batch_norm_2(x)
        x = self.dropout(x)
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            residual = self.skip_conv(residual)
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
            -embeddings * torch.arange(half_dims, device=t.device)
        )  # got 1/(10000^(k/d))
        embeddings = t[:, None] * embeddings[None, :]  # got frequencies
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        # print(embeddings.shape)
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
                res_blocks.append(
                    ResidualBlock(block_dim_in, block_dim_out, self.time_emb_size)
                )
                block_dim_in = block_dim_out
                if current_res in attn_resolutions:
                    attn_blocks.append(SelfAttentionBlock(block_dim_in, 1))
            level_objects = nn.Module()
            level_objects.res_blocks = res_blocks
            level_objects.attn_blocks = attn_blocks
            if i < len(self.channels_multipliers) - 1:
                level_objects.scale = DownSampleBlock(block_dim_in, use_conv=True)
                current_res = current_res // 2
            self.down.append(level_objects)

        # STAGE 2: Middle
        self.middle = nn.Module()
        self.middle.res_block_1 = ResidualBlock(
            in_channels=block_dim_in,
            out_channels=block_dim_in,
            time_embedding_dim=self.time_emb_size,
        )
        self.middle.attn_block = SelfAttentionBlock(block_dim_in, 1)
        self.middle.res_block_2 = ResidualBlock(
            in_channels=block_dim_in,
            out_channels=block_dim_in,
            time_embedding_dim=self.time_emb_size,
        )

        # STAGE 3: Upsampling
        self.up = nn.ModuleList()
        for i in reversed(range(len(self.channels_multipliers))):
            res_blocks = nn.ModuleList()
            attn_blocks = nn.ModuleList()
            block_dim_out = self.init_channels * self.channels_multipliers[i]
            skip_dim_in = self.init_channels * self.channels_multipliers[i]
            for j in range(self.num_res_blocks + 1):
                if j == self.num_res_blocks:
                    skip_dim_in = self.init_channels * in_channels_multipliers[i]
                res_blocks.append(
                    ResidualBlock(
                        block_dim_in + skip_dim_in, block_dim_out, self.time_emb_size
                    )
                )
                block_dim_in = block_dim_out
                if current_res in attn_resolutions:
                    attn_blocks.append(SelfAttentionBlock(block_dim_in, 1))
            level_objects = nn.Module()
            level_objects.res_blocks = res_blocks
            level_objects.attn_blocks = attn_blocks
            if i > 0:
                level_objects.scale = UpSampleBlock(block_dim_in, use_conv=True)
                current_res = current_res * 2
            self.up.insert(0, level_objects)

        # Final stage
        self.out_norm = nn.GroupNorm(num_groups=32, num_channels=block_dim_in)
        self.out_conv = nn.Conv2d(
            in_channels=block_dim_in,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x, t):
        assert x.shape[2] == x.shape[3] == self.input_img_resolution  # square image

        # STAGE 0: time embedding
        t_emb = self.t_emb(t)

        # STAGE 1: Downsampling
        h_list = [self.conv_init(x)]
        for i in range(len(self.channels_multipliers)):
            for j in range(self.num_res_blocks):
                h = self.down[i].res_blocks[j](h_list[-1], t_emb)
                if len(self.down[i].attn_blocks) > 0:
                    h = self.down[i].attn_blocks[j](h)
                h_list.append(h)
            if i < len(self.channels_multipliers) - 1:
                h = self.down[i].scale(h)
                h_list.append(h)

        # STAGE 2: Middle
        h = self.middle.res_block_1(h, t_emb)
        h = self.middle.attn_block(h)
        h = self.middle.res_block_2(h, t_emb)

        # STAGE 3: Upsampling
        for i in reversed(range(len(self.channels_multipliers))):
            for j in range(self.num_res_blocks + 1):
                h = self.up[i].res_blocks[j](torch.cat([h, h_list.pop()], dim=1), t_emb)
                if len(self.up[i].attn_blocks) > 0:
                    h = self.up[i].attn_blocks[j](h)
                h_list.append(h)
            if i > 0:
                h = self.up[i].scale(h)
                h_list.append(h)

        # Final stage
        h = self.out_norm(h)
        h = self.out_conv(h)

        return h


if __name__ == "__main__":
    model = UNet(
        init_channels=32,
        in_channels=3,
        out_channels=3,
        num_res_blocks=2,
        attn_resolutions=(16,),
        input_img_resolution=32,
        channels_multipliers=(1, 2, 2, 2),
    ).to("cpu")
    total_params = sum([p.numel() for p in model.parameters()])
    print("Total parameters = ", total_params)
    # for name, param in model.state_dict().items():
    #     print(name, param.size())

    test_data = np.random.randn(8, 3, 32, 32).astype(np.float32)
    test_data = torch.from_numpy(test_data).to("cpu")

    t = torch.from_numpy(np.array(np.arange(8))).to("cpu")

    torch.onnx.export(model, (test_data, t), f="unet.onnx")

    # model = ResidualBlock(32, 32, 512).to("cpu")
    # total_params = sum([p.numel() for p in model.parameters()])
    # print("Total parameters = ", total_params)
    # for name, param in model.state_dict().items():
    #     print(name, param.size())

    # model = SelfAttentionBlock(128, 2).to("cpu")
    # total_params = sum([p.numel() for p in model.parameters()])
    # print("Total parameters = ", total_params)
    # for name, param in model.state_dict().items():
    #     print(name, param.size())

    # test = torch.randn(size=(8, 32, 256, 256))
    # downsample = DownSampleBlock(channels=32, use_conv=True)
    # test = downsample(test)
    # print(test.shape)

    # test = torch.randn(size=(8, 32, 128, 128))
    # upsample = UpSampleBlock(channels=32, use_conv=True)
    # test = upsample(test)
    # print(test.shape)
