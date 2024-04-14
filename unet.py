import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from torchvision.transforms import Resize
import math

# TODO: implement temporal unet

# This downsample block is from
# https://github.com/pesser/pytorch_diffusion/blob/master/pytorch_diffusion/model.py#L54
class DownSampleBlock(nn.Module):
    def __init__(self, channels, use_conv=True):
        super().__init__()
        self.with_conv = use_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(channels,
                                        channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


# class DownSampleBlock(nn.Module):
#     def __init__(self, channels, use_conv=True):
#         super(DownSampleBlock, self).__init__()
#         self.use_conv = use_conv
#         self.conv = nn.Conv2d(
#             in_channels=channels,
#             out_channels=channels,
#             kernel_size=3,
#             stride=2,
#             padding=1,
#         )

#     def forward(self, x_in):
#         x = x_in
#         N, C, H, W = x.shape
#         if self.use_conv:
#             x = self.conv(x)
#         else:
#             down = nn.AvgPool2d(kernel_size=2, stride=2)
#             x = down(x)
#         # print(f"    down x_out.shape={x.shape}")
#         assert x.shape[2] == H // 2
#         assert x.shape[3] == W // 2

#         return x

# This upsample block is from
# https://github.com/pesser/pytorch_diffusion/blob/master/pytorch_diffusion/model.py#L36
class UpSampleBlock(nn.Module):
    def __init__(self, channels, use_conv=True):
        super().__init__()
        self.with_conv = use_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(channels,
                                        channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

# class UpSampleBlock(nn.Module):
#     def __init__(self, channels, use_conv=True):
#         super(UpSampleBlock, self).__init__()
#         self.use_conv = use_conv
#         if self.use_conv:
#             self.conv = nn.Conv2d(
#                 in_channels=channels,
#                 out_channels=channels,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#             )

#     def forward(self, x_in):
#         x = x_in
#         N, C, H, W = x.shape
#         # resize = Resize((H * 2, W * 2))
#         # x = resize(x)
#         x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
#         if self.use_conv:
#             x = self.conv(x)

#         # print(f"    up x_out.shape={x.shape}")
#         assert x.shape[2] == H * 2
#         assert x.shape[3] == W * 2

#         return x


# This attn block is from
# https://github.com/pesser/pytorch_diffusion/blob/master/pytorch_diffusion/model.py#L136
class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels, num_heads=8, name="none"):
        super().__init__()
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


# class SelfAttentionBlock(nn.Module):
#     def __init__(self, in_channels, num_heads=8, name="none"):
#         super(SelfAttentionBlock, self).__init__()
#         self.in_channels = in_channels
#         self.norm = nn.GroupNorm(
#             num_channels=in_channels, num_groups=32, eps=1e-6, affine=True
#         )

#         self.conv_values = nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=in_channels,
#             stride=1,
#             kernel_size=1,
#             padding=0,
#         )
#         self.conv_keys = nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=in_channels,
#             stride=1,
#             kernel_size=1,
#             padding=0,
#         )
#         self.conv_queries = nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=in_channels,
#             stride=1,
#             kernel_size=1,
#             padding=0,
#         )

#         self.conv_out = nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=in_channels,
#             stride=1,
#             kernel_size=1,
#             padding=0,
#         )

#     def forward(self, x):
#         # print(f"    attn in={self.in_channels} x_in.shape={x.shape}")
#         h_ = x
#         h_ = self.norm(h_)
#         q = self.conv_queries(h_)
#         k = self.conv_keys(h_)
#         v = self.conv_values(h_)

#         # compute QK
#         N, C, H, W = q.shape
#         q = q.reshape(N, C, H * W)  # N x C x HW
#         k = k.reshape(N, C, H * W)  # N x C x HW
#         w_qk = torch.einsum("ncq,nck->nqk", [q, k])  # N x HW x HW
#         w_qk = w_qk * int(C) ** (-0.5)  # scaled-dot sqrt(dim)
#         w_qk = torch.nn.functional.softmax(w_qk, dim=2)

#         # attend to V
#         v = v.reshape(N, C, H * W)  # N x C x HW
#         attn = torch.einsum("ncq,nqk->ncq", [v, w_qk])
#         attn = attn.reshape(N, C, H, W)
#         attn = self.conv_out(attn)
#         # print(f"    attn x_out.shape={x.shape}")
#         return attn


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)

# This res block is from
# https://github.com/pesser/pytorch_diffusion/blob/master/pytorch_diffusion/model.py#L76
class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        temb_channels,
        conv_shortcut=False,
        dropout=0.1,
        name="",
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = torch.nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = torch.nn.GroupNorm(
            num_groups=32, num_channels=out_channels, eps=1e-6, affine=True
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


# class ResidualBlock(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         time_embedding_dim,
#         stride=1,
#         dropout=0.1,
#         name="none",
#     ):
#         super(ResidualBlock, self).__init__()
#         self.name = name
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         # #print("in_channels, out_channels", self.in_channels, self.out_channels)
#         self.time_mlp = nn.Linear(time_embedding_dim, out_channels)
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=in_channels,
#                 out_channels=out_channels,
#                 kernel_size=3,
#                 stride=stride,
#                 padding=1,
#             ),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=out_channels,
#                 out_channels=out_channels,
#                 kernel_size=3,
#                 stride=stride,
#                 padding=1,
#             ),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#         )
#         self.relu = nn.ReLU()
#         self.batch_norm_1 = nn.BatchNorm2d(in_channels)
#         self.batch_norm_2 = nn.BatchNorm2d(out_channels)
#         self.dropout = nn.Dropout(dropout)
#         self.skip_conv = nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#         )

#     def forward(self, x_in, t_emb_in):
#         x = x_in
#         t_emb = self.time_mlp(t_emb_in)

#         # fmt: off
#         #print(f"{self.name}\n   in={self.in_channels}, out={self.out_channels}\n    x_in.shape={x.shape}")

#         residual = x
#         x = self.batch_norm_1(x)
#         x = self.conv1(x)
#         t_emb = t_emb[:, :, None, None]
#         x = x + t_emb

#         x = self.batch_norm_2(x)
#         x = self.dropout(x)
#         x = self.conv2(x)
#         if self.in_channels != self.out_channels:
#             residual = self.skip_conv(residual)
#         x = x + residual
#         # x = self.relu(x)
#         #print(f"    x_out.shape={x.shape}")
#         return x

# this emb is from
# https://github.com/pesser/pytorch_diffusion/blob/master/pytorch_diffusion/model.py#L6
def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb

# class SinusoidalPositionalEmbedding(nn.Module):
#     def __init__(self, dim):
#         super(SinusoidalPositionalEmbedding, self).__init__()
#         self.dim = dim

#     def forward(self, t):
#         half_dims = self.dim // 2
#         # this implements 1/(10000^(k/d))
#         # using the fact that e^-ln(x)=1/x
#         # why? probably faster in some ways?
#         # TODO: benchmark this later
#         embeddings = np.log(10000) / (half_dims - 1)
#         embeddings = torch.exp(
#             -embeddings * torch.arange(half_dims, device=t.device)
#         )  # got 1/(10000^(k/d))
#         embeddings = t[:, None] * embeddings[None, :]  # got frequencies
#         embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
#         # #print(embeddings.shape)
#         return embeddings


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
        dropout=0.1,
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
        # self.t_emb = nn.Sequential(
        #     SinusoidalPositionalEmbedding(self.time_emb_size),
        #     nn.Linear(self.time_emb_size, self.time_emb_size),
        #     nn.ReLU(),
        # )
        # ref impl
        self.t_emb = nn.Module()
        self.t_emb.dense = nn.ModuleList([
            torch.nn.Linear(self.init_channels,
                            self.time_emb_size),
            torch.nn.Linear(self.time_emb_size,
                            self.time_emb_size),
        ])

        # STAGE 1: Downsampling
        self.conv_init = nn.Conv2d(
            in_channels=in_channels,
            out_channels=init_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        block_dim_list = []

        current_res = input_img_resolution
        in_channels_multipliers = (1,) + self.channels_multipliers
        self.down = nn.ModuleList()
        for i in range(len(self.channels_multipliers)):
            res_blocks = nn.ModuleList()
            attn_blocks = nn.ModuleList()
            block_dim_in = self.init_channels * in_channels_multipliers[i]
            block_dim_out = self.init_channels * self.channels_multipliers[i]
            # print(f"down {i} with mult={self.channels_multipliers[i]}")
            for j in range(self.num_res_blocks):
                # print(f"     res block {j}, c_in={block_dim_in}, c_out={block_dim_out}")
                res_blocks.append(
                    ResidualBlock(
                        block_dim_in,
                        block_dim_out,
                        self.time_emb_size,
                        name=f"down{i}b{j}",
                    )
                )
                block_dim_list.append([block_dim_in, block_dim_out])
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
            block_dim_in,
            block_dim_in,
            self.time_emb_size,
            name="mid1",
        )
        self.middle.attn_block = SelfAttentionBlock(block_dim_in, 1)
        self.middle.res_block_2 = ResidualBlock(
            block_dim_in,
            block_dim_in,
            self.time_emb_size,
            name="mid2",
        )

        # STAGE 3: Upsampling
        # print(block_dim_list)
        self.up = nn.ModuleList()
        for i in reversed(range(len(self.channels_multipliers))):
            res_blocks = nn.ModuleList()
            attn_blocks = nn.ModuleList()
            block_dim_out = self.init_channels * self.channels_multipliers[i]
            skip_dim_in = self.init_channels * self.channels_multipliers[i]
            # skip_dim_in = 0
            # print(f"up {i} with mult={self.channels_multipliers[i]}")
            for j in reversed(range(self.num_res_blocks)):
                # if j == self.num_res_blocks:
                #     skip_dim_in = self.init_channels * in_channels_multipliers[i]
                block_dim_in = block_dim_list[i * self.num_res_blocks + j][1]
                block_dim_out = block_dim_list[i * self.num_res_blocks + j][0]
                if j < self.num_res_blocks - 1:
                    skip_dim_in = 0
                # fmt:off
                # print(f"     res block {self.num_res_blocks-j-1}, c_in={block_dim_in} + {skip_dim_in}, c_out={block_dim_out}")
                res_blocks.append(
                    ResidualBlock(
                        block_dim_in + skip_dim_in,
                        block_dim_out,
                        self.time_emb_size,
                        name=f"up{i}b{self.num_res_blocks - j-1}",
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

    def forward(self, x_in, t_in):
        x, t = x_in, t_in
        assert x.shape[2] == self.input_img_resolution  # square image
        assert x.shape[3] == self.input_img_resolution  # square image

        # STAGE 0: time embedding
        # t_emb = self.t_emb(t)
        # ref impl
        t_emb = get_timestep_embedding(t, self.init_channels)
        t_emb = self.t_emb.dense[0](t_emb)
        t_emb = nonlinearity(t_emb)
        t_emb = self.t_emb.dense[1](t_emb)

        # STAGE 1: Downsampling
        h_list = []
        h = self.conv_init(x)
        for i in range(len(self.channels_multipliers)):
            for j in range(self.num_res_blocks):
                h = self.down[i].res_blocks[j](h, t_emb)
                if len(self.down[i].attn_blocks) > 0:
                    h = self.down[i].attn_blocks[j](h)
                if j == self.num_res_blocks - 1:
                    h_list.append(h)
            if i < len(self.channels_multipliers) - 1:
                h = self.down[i].scale(h)
                # h_list.append(h)

        # STAGE 2: Middle
        h = self.middle.res_block_1(h, t_emb)
        h = self.middle.attn_block(h)
        h = self.middle.res_block_2(h, t_emb)

        # STAGE 3: Upsampling
        for i in reversed(range(len(self.channels_multipliers))):
            for j in range(self.num_res_blocks):
                if j == 0:
                    h = self.up[i].res_blocks[j](
                        torch.cat([h, h_list.pop()], dim=1), t_emb
                    )
                else:
                    h = self.up[i].res_blocks[j](h, t_emb)
                if len(self.up[i].attn_blocks) > 0:
                    h = self.up[i].attn_blocks[j](h)
                # h_list.append(h)
            if i > 0:
                h = self.up[i].scale(h)
                # h_list.append(h)

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
        attn_resolutions=(
            16,
            32,
        ),
        input_img_resolution=128,
        channels_multipliers=(1, 2, 2, 2),
    ).to("cpu")
    total_params = sum([p.numel() for p in model.parameters()])
    # print("Total parameters = ", total_params)
    # for name, param in model.state_dict().items():
    #     #print(name, param.size())

    test_data = np.random.randn(8, 3, 128, 128).astype(np.float32)
    test_data = torch.from_numpy(test_data).to("cpu")

    t = torch.from_numpy(np.array(np.arange(8))).to("cpu")

    torch.onnx.export(model, (test_data, t), f="unet.onnx")

    # model = ResidualBlock(32, 32, 512).to("cpu")
    # total_params = sum([p.numel() for p in model.parameters()])
    # #print("Total parameters = ", total_params)
    # for name, param in model.state_dict().items():
    #     #print(name, param.size())

    # model = SelfAttentionBlock(128, 2).to("cpu")
    # total_params = sum([p.numel() for p in model.parameters()])
    # #print("Total parameters = ", total_params)
    # for name, param in model.state_dict().items():
    #     #print(name, param.size())

    # test = torch.randn(size=(8, 32, 256, 256))
    # downsample = DownSampleBlock(channels=32, use_conv=True)
    # test = downsample(test)
    # #print(test.shape)

    # test = torch.randn(size=(8, 32, 128, 128))
    # upsample = UpSampleBlock(channels=32, use_conv=True)
    # test = upsample(test)
    # #print(test.shape)
