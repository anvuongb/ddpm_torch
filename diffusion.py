from unet import UNet
import torch.nn as nn
import torch.functional as F
import numpy as np
import torch
import torchvision

def get_noise_schedule(num_steps, start=0.0001, end=0.01):
    return torch.linspace(start, end, num_steps)

def forward_diffusion(x_0, t, alphas_cum, device="cpu"):
    noise = torch.rand_like(x_0)*255
    scale_t = alphas_cum[t]
    x_t = torch.sqrt(scale_t)*x_0 + (1-scale_t)*noise
    return x_t.to(device)

if __name__ == "__main__":
    T = 1000
    betas = get_noise_schedule(T)
    alphas = 1 - betas
    alphas_cum = torch.cumprod(alphas, dim=0)
    img = torchvision.io.read_image("/home/anvuong/Desktop/datasets/CIFAR-10-images/train/cat/2000.jpg").type(torch.float32)
    
    t = torch.tensor([200]).to('cpu')
    x = forward_diffusion(img, t, alphas_cum) # input
    x_label = forward_diffusion(img, t-1, alphas_cum) # output
    # training pair will be (img_diff, img_diff_prev)
    
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
    
    torch.autograd.set_detect_anomaly(True)

    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    opt.zero_grad()
    x = x[None,:,:,:]
    # t = t[None,:]
    print("input_shape=", x.shape)
    x_denoised = model.forward(x, t)
    loss = torch.sum((x_denoised - x)**2)
    loss.backward()

    opt.step()
    
    img_diff = x[0].type(torch.uint8)
    torchvision.io.write_jpeg(img_diff, "images/test_diffused.jpeg")