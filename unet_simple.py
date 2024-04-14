import numpy as np
import torch
from torch import nn
from unet import SelfAttentionBlock, ResidualBlock, UpSampleBlock, DownSampleBlock, SinusoidalPositionalEmbedding, Swiss

# this is a simpler network construction
# this more resembles the original UNet paper
# the network only performs skip connection in
# the same resolution between the last and the first
# ResdiualBlock

class UNetSimple(nn.Module):
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
        super(UNetSimple, self).__init__()
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
            # nn.ReLU(),
            Swiss()
        )

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
        # # print(block_dim_list)
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
        t_emb = self.t_emb(t)

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

        # STAGE 2: Middle
        h = self.middle.res_block_1(h_list[-1], t_emb)
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
            if i > 0:
                h = self.up[i].scale(h)

        # Final stage
        h = self.out_norm(h)
        h = h * torch.sigmoid(h)
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
    # print("Total parameters = ", total_params)
    # for name, param in model.state_dict().items():
    #     ## print(name, param.size())

    test_data = np.random.randn(8, 3, 32, 32).astype(np.float32)
    test_data = torch.from_numpy(test_data).to("cpu")

    t = torch.from_numpy(np.array(np.arange(8))).to("cpu")

    torch.onnx.export(model, (test_data, t), f="unet.onnx")