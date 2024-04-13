from unet import UNet
import torch.functional as F
from torch.utils.data import DataLoader
import torch
import numpy as np
from datasets import CifarDataset, cifar_data_transform, show_images_batch
import tqdm


def get_noise_schedule(num_steps, start=0.0001, end=0.01):
    return torch.linspace(start, end, num_steps)


def forward_diffusion(x_0, t, alphas_cum, device="cpu"):
    noise = torch.randn_like(x_0).to(device)
    scale_t = torch.gather(alphas_cum, 1, t)
    scale_t = scale_t[:, :, None, None]
    x_t = torch.sqrt(scale_t) * x_0 + (1 - scale_t) * noise
    return x_t.to(device), noise

# TODO: finish this sample function
@torch.no_grad()
def sample_from_noise(
    x_0: torch.Tensor, model: UNet, timesteps: int, device: str = "cpu"
):
    x = torch.randn_like(x_0).to(device)
    for t in reversed(range(timesteps)):
        noise_pred = model.forward(x, t)


if __name__ == "__main__":
    for i in range(torch.cuda.device_count()):
        print(
            "found GPU",
            torch.cuda.get_device_name(i),
            "capability",
            torch.cuda.get_device_capability(i),
        )

    # Init dataset
    batch_size = 64
    data_transform = cifar_data_transform()
    data = CifarDataset(
        img_dir="/home/anvuong/Desktop/datasets/CIFAR-10-images/train",
        classes=["cat"],
        transform=data_transform,
    )
    loader = DataLoader(data, batch_size=batch_size, drop_last=True)

    # Init model
    device = "cuda:1"
    model = UNet(
        init_channels=32,
        in_channels=3,
        out_channels=3,
        num_res_blocks=3,
        attn_resolutions=(16,32,),
        input_img_resolution=32,
        channels_multipliers=(1, 2, 2, 2),
    ).to(device)
    total_params = sum([p.numel() for p in model.parameters()])

    # Init diffusion params
    T = 1000
    betas = get_noise_schedule(T)
    alphas = 1 - betas
    alphas_cum = torch.cumprod(alphas, dim=0)
    alphas_cum = alphas_cum.unsqueeze(0).repeat(batch_size, 1).to(device)

    # torch.autograd.set_detect_anomaly(True)

    # Init optimizer
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    opt.zero_grad()

    # # Get a sample batch
    # x = next(iter(loader))
    # t = torch.randint(low=1, high=T-200, size=(batch_size,1))

    # x_in = forward_diffusion(x, t+200, alphas_cum, device=device)
    # x_out = forward_diffusion(x, t, alphas_cum, device=device)

    # show_images_batch("images/x_in.png", x_in)
    # show_images_batch("images/x_out.png", x_out)

    # train params
    epochs = 50

    # TODO: add sample codes
    # generate samples every epoch
    # train loop
    # TODO: loss starts to become nan as epoch 2, need to fix
    total = len(loader)
    print("num batches = ", total)
    for e in range(epochs):
        for idx, x in tqdm.tqdm(enumerate(loader), total=total):
            x = x.to(device)
            t = torch.randint(low=0, high=T, size=(batch_size, 1)).to(device)

            x_in, noise_in = forward_diffusion(x, t, alphas_cum, device=device)
            # x_out = forward_diffusion(x, t, alphas_cum, device=device)

            noise_pred = model.forward(x_in, t.squeeze())
            loss = torch.nn.MSELoss()
            L = loss(noise_in, noise_pred)
            # print(f"epoch {e} iter {idx} loss={L.item()}")
            # if L.item() == torch.nan:
            #     show_images_batch("images/x_in.png", x_in)
            #     show_images_batch("images/x_out.png", x_out)
            #     show_images_batch("images/x_denoised.png", x_denoised)
            #     print("Found nan, break")
            #     break

            L.backward()
            opt.step()
        print(f"epoch {e} loss={L.item()}")
