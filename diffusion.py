from unet import UNet
from unet_simple import UNetSimple
import torch.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import writer
import torch
import numpy as np
from datasets import CifarDataset, cifar_data_transform, show_images_batch
import tqdm
import time
from model_helpers import save_model, load_model


def get_noise_schedule(num_steps, start=0.0001, end=0.02):
    return torch.linspace(start, end, num_steps)


def forward_diffusion(x_0, t, alphas_cum, device="cpu"):
    noise = torch.randn_like(x_0).to(device)
    scale_t = torch.gather(alphas_cum, 1, t)
    scale_t = scale_t[:, :, None, None]
    x_t = torch.sqrt(scale_t) * x_0 + (1 - scale_t) * noise
    return x_t.to(device), noise


# TODO: finish this sample function
@torch.no_grad()
def one_step_denoising(model, x, t, alphas_cum, alphas, betas, device="cpu"):

    alpha_t = torch.gather(alphas, 1, t)
    alpha_t = alpha_t[:, :, None, None]

    beta_t = torch.gather(betas, 1, t)
    beta_t = beta_t[:, :, None, None]

    scale_t = torch.gather(alphas_cum, 1, t)
    scale_t = scale_t[:, :, None, None]

    z = torch.randn_like(x)

    if t.squeeze()[0] == 0:
        z = z * 0

    noise_pred = model(x, t.squeeze())

    x_out = (1 / torch.sqrt(alpha_t)) * (
        x - beta_t / torch.sqrt(1 - scale_t) * noise_pred
    ) + torch.sqrt(beta_t) * z

    x_out = torch.clamp(x_out, -1, 1)

    return x_out.to(device)


@torch.no_grad()
def sample_from_noise(
    x_0: torch.Tensor,
    model: UNet,
    timesteps: int,
    alphas_cum: torch.Tensor,
    alphas: torch.Tensor,
    betas: torch.Tensor,
    device: str = "cpu",
):
    x = torch.randn_like(x_0).to(device)
    for t_ in tqdm.tqdm(reversed(range(timesteps)), total=timesteps):
        t = torch.ones(size=(x.shape[0], 1), dtype=int) * t_
        t = t.to(device)
        x = one_step_denoising(model, x, t, alphas_cum, alphas, betas, device=device)
    return x


if __name__ == "__main__":
    for i in range(torch.cuda.device_count()):
        print(
            "found GPU",
            torch.cuda.get_device_name(i),
            "capability",
            torch.cuda.get_device_capability(i),
        )

    # Init dataset
    batch_size = 512
    data_transform = cifar_data_transform()
    data = CifarDataset(
        img_dir="/home/anvuong/Desktop/datasets/CIFAR-10-images/train",
        classes=["cat"],
        transform=data_transform,
    )
    loader = DataLoader(data, batch_size=batch_size, drop_last=True)

    # Init model
    device = "cuda:0"
    model = UNetSimple(
        init_channels=32,
        in_channels=3,
        out_channels=3,
        num_res_blocks=2,
        attn_resolutions=(16,),
        input_img_resolution=32,
        channels_multipliers=(1, 2, 2, 2),
    ).to(device)

    # cifar10_cfg = {
    #     "resolution": 32,
    #     "in_channels": 3,
    #     "out_ch": 3,
    #     "ch": 128,
    #     "ch_mult": (1, 2, 2, 2),
    #     "num_res_blocks": 2,
    #     "attn_resolutions": (16,),
    #     "dropout": 0.1,
    # }
    # model = unet_ref.UNet(
    #     **cifar10_cfg
    # ).to(device)
    total_params = sum([p.numel() for p in model.parameters()])
    print("Model initialized, total params = ", total_params)

    # Init diffusion params
    T = 500
    betas = get_noise_schedule(T)
    alphas = 1 - betas
    alphas_cum = torch.cumprod(alphas, dim=0)

    # repeat into batch_size for easier indexing by t
    betas = betas.unsqueeze(0).repeat(batch_size, 1).to(device)
    alphas = alphas.unsqueeze(0).repeat(batch_size, 1).to(device)
    alphas_cum = alphas_cum.unsqueeze(0).repeat(batch_size, 1).to(device)

    # torch.autograd.set_detect_anomaly(True)

    # # Get a sample batch
    x = next(iter(loader))
    t = torch.randint(low=1, high=T - 200, size=(batch_size, 1))

    # Exp name
    exp_name = "Diffusion-Cifar10-cat"

    # init tensorboard writer
    current_time = time.time()
    tb_writer = writer.SummaryWriter(f"logdir/{exp_name}_{current_time}")
    tb_writer.add_graph(
        model=model, input_to_model=[x.to(device), t.squeeze().to(device)]
    )

    # train params
    epochs = 10000
    start_epoch = 0

    # # load from save
    # model = load_model(
    #     "/home/anvuong/Desktop/codes/ddpm_torch/models/Diffusion-Cifar10-cat/model.pkl",
    #     model,
    # )
    model.load_state_dict(torch.load("/home/anvuong/Desktop/codes/ddpm_torch/models/Diffusion-Cifar10-cat/model.pkl",map_location=device))
    # start_epoch = 0

    # re-init dataloader
    loader = DataLoader(data, batch_size=batch_size, drop_last=True)

    # Init optimizer
    opt = torch.optim.Adam(model.parameters(), lr=1e-6)
    opt.zero_grad()

    # TODO: currently loss keeps increasing with epochs, shit is happening
    # TODO: need to fix this
    total = len(loader)
    print("num batches = ", total)
    for e in np.arange(start_epoch, start_epoch + epochs):
        for idx, x in tqdm.tqdm(enumerate(loader), total=total):
            x = x.to(device)
            t = torch.randint(low=0, high=T, size=(batch_size, 1)).to(device)

            x_in, noise_in = forward_diffusion(x, t, alphas_cum, device=device)
            # x_out = forward_diffusion(x, t, alphas_cum, device=device)

            noise_pred = model(x_in, t.squeeze())
            loss = torch.nn.functional.mse_loss(noise_in, noise_pred, reduction="mean")
            loss.backward()
            opt.step()
            tb_writer.add_scalar("loss", loss.item(), e * total + idx)

            if loss.item() == torch.nan:
                raise Exception("loss becomes nan")

        print(f"epoch {e} loss={loss.item()}")
        # show image and save model every 100 epochs
        if e % 100 == 0:
            print("Generating sample images")
            x_in = torch.ones(16, *x.shape[1:])
            x_denoised = sample_from_noise(
                x_in, model, T, alphas_cum, alphas, betas, device=device
            )
            x_denoised = x_denoised.to("cpu")
            show_images_batch(f"sampling_images/sample_epoch_{e}.png", x_denoised)
            save_model(f"models/{exp_name}/model.pkl", model)
