from unet import UNet
from torchvision import transforms
import os
from unet_simple import UNetSimple
import torch.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import writer
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import torch
import numpy as np
from datasets import MnistDataset, mnist_data_transform, show_images_batch
import tqdm
import time
from model_helpers import save_model, load_model
import wandb


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
def one_step_denoising(
    model,
    x,
    t,
    alphas_cum,
    alphas_cum_prev,
    alphas,
    betas,
    noise_sched="from_gaussian",
    device="cpu",
):

    alpha_t = torch.gather(alphas, 1, t)
    alpha_t = alpha_t[:, :, None, None]

    beta_t = torch.gather(betas, 1, t)
    beta_t = beta_t[:, :, None, None]

    scale_t = torch.gather(alphas_cum, 1, t)
    scale_t = scale_t[:, :, None, None]

    scale_t_prev = torch.gather(alphas_cum_prev, 1, t)
    scale_t_prev = scale_t_prev[:, :, None, None]

    z = torch.randn_like(x)

    if t.squeeze()[0] == 0:
        z = z * 0

    noise_pred = model(x, t.squeeze())
    noise_pred = torch.clamp(noise_pred, -1, 1)

    # noise = beta
    noise_scale = torch.sqrt(beta_t)
    # if noise_sched == "from_gaussian":
    #     # noise = scaling
    #     noise_scale = torch.sqrt((1 - scale_t_prev) / (1 - scale_t) * beta_t)

    x_out = (1 / torch.sqrt(alpha_t)) * (
        x - beta_t / torch.sqrt(1 - scale_t) * noise_pred
    ) + noise_scale * z

    # noise = beta scale

    return x_out.to(device)


@torch.no_grad()
def sample_from_noise(
    x_0: torch.Tensor,
    model: UNet,
    timesteps: int,
    alphas_cum: torch.Tensor,
    alphas_cum_prev: torch.Tensor,
    alphas: torch.Tensor,
    betas: torch.Tensor,
    device: str = "cpu",
):
    x = torch.randn_like(x_0).to(device)
    for t_ in tqdm.tqdm(reversed(range(timesteps)), total=timesteps):
        t = torch.ones(size=(x.shape[0], 1), dtype=int) * t_
        t = t.to(device)
        x = one_step_denoising(
            model, x, t, alphas_cum, alphas_cum_prev, alphas, betas, device=device
        )
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
    batch_size = 256

    tf = transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor()]
    )  # mnist is already normalised 0 to 1

    data = MNIST("./data", train=True, download=True, transform=tf)
    loader = DataLoader(
        data, batch_size=batch_size, shuffle=True, num_workers=5, drop_last=True
    )

    # loader = DataLoader(data, batch_size=batch_size, drop_last=True)

    # Init model
    device = "cuda:0"
    unet_conf = {
        "init_channels": 64,
        "in_channels": 1,
        "out_channels": 1,
        "num_res_blocks": 2,
        "attn_resolutions": (1,),
        "input_img_resolution": 32,
        "channels_multipliers": (2, 2),
    }
    model = UNet(**unet_conf).to(device)

    total_params = sum([p.numel() for p in model.parameters()])
    print("Model initialized, total params = ", total_params)

    # Init diffusion params
    T = 500
    betas = get_noise_schedule(T)
    alphas = 1 - betas
    alphas_cum = torch.cumprod(alphas, dim=0)
    alphas_cum_prev = torch.cat([torch.Tensor([1.0]), alphas_cum[:-1]])

    # repeat into batch_size for easier indexing by t
    betas = betas.unsqueeze(0).repeat(batch_size, 1).to(device)
    alphas = alphas.unsqueeze(0).repeat(batch_size, 1).to(device)
    alphas_cum = alphas_cum.unsqueeze(0).repeat(batch_size, 1).to(device)
    alphas_cum_prev = alphas_cum_prev.unsqueeze(0).repeat(batch_size, 1).to(device)

    # torch.autograd.set_detect_anomaly(True)

    # # Get a sample batch
    x = next(iter(loader))
    x = x[0]
    # print(x.shape)
    t = torch.randint(low=1, high=T, size=(batch_size, 1))
    print(x.shape)
    # Exp name
    exp_name = "Mnist-all-3"

    # init tensorboard writer
    current_time = time.time()
    # tb_writer = writer.SummaryWriter(f"logdir/{exp_name}_{current_time}")
    tb_writer = writer.SummaryWriter(f"logdir/{exp_name}")
    tb_writer.add_graph(
        model=model, input_to_model=[x.to(device), t.squeeze().to(device)]
    )

    # train params
    epochs = 100
    start_epoch = 25
    load_model = True

    # load from save
    if load_model:
        model.load_state_dict(
            torch.load(
                "/home/anvuong/Desktop/codes/ddpm_torch/models/Mnist-all-3/model.pkl",
                map_location=device,
            )
        )
    # start_epoch = 0

    # re-init dataloader
    # loader = DataLoader(data, batch_size=batch_size, drop_last=True)
    loader = DataLoader(
        data, batch_size=batch_size, shuffle=True, num_workers=5, drop_last=True
    )

    # Init optimizer
    lr = 5e-7
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    opt.zero_grad()

    # TODO: currently loss keeps increasing with epochs, shit is happening
    # TODO: need to fix this
    total = len(loader)
    print("num batches = ", total)
    wandb.init(
        project="mnist-all-3",
        config={
            "load_model": load_model,
            "learning_rate": lr,
            "epochs": epochs,
            "start_epoch": start_epoch,
            "unet_config": unet_conf,
        },
    )
    for e in np.arange(start_epoch, start_epoch + epochs):
        loss_ema = None
        pbar = tqdm.tqdm(enumerate(loader), total=total)
        for idx, x in pbar:
            x = x[0]
            x = x.to(device)
            t = torch.randint(low=0, high=T, size=(batch_size, 1)).to(device)

            x_in, noise_in = forward_diffusion(x, t, alphas_cum, device=device)
            # x_out = forward_diffusion(x, t, alphas_cum, device=device)

            noise_pred = model(x_in, t.squeeze())
            loss = torch.nn.functional.mse_loss(noise_in, noise_pred, reduction="mean")
            loss.backward()
            tb_writer.add_scalar("loss", loss.item(), e * total + idx)

            if loss.item() == torch.nan:
                raise Exception("loss becomes nan")

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")

            opt.step()
            
            # log to wandb
            wandb.log({"loss": loss.item()})

        # linear lrate decay
        opt.param_groups[0]["lr"] = lr * (1 - e / epochs)

        print(f"epoch {e} loss={loss.item()}")
        # show image and save model every 100 epochs
        if e % 1 == 0:
            print("Generating sample images")
            x_in = torch.ones(16, *x.shape[1:])
            x_denoised = sample_from_noise(
                x_in,
                model,
                T,
                alphas_cum,
                alphas_cum_prev,
                alphas,
                betas,
                device=device,
            )
            x_denoised = x_denoised.to("cpu")
            if not os.path.exists(f"./sampling_images/{exp_name}"):
                os.makedirs(f"./sampling_images/{exp_name}")
            # show_images_batch(
            #     f"sampling_images/{exp_name}/sample_epoch_{e}.png", x_denoised
            # )
            # show_images_batch(f"sampling_images/{exp_name}/latest.png", x_denoised)
            save_image(x_denoised, f"sampling_images/{exp_name}/sample_epoch_{e}.png")
            save_image(x_denoised, f"sampling_images/{exp_name}/latest.png")
            wandb.save(f"sampling_images/{exp_name}/*.png")
            save_model(f"models/{exp_name}/model.pkl", model)
