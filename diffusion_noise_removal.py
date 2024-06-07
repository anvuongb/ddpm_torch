# from unet import UNet
from unet_ref import UNet
from torchvision import transforms
import os
from unet_simple import UNetSimple
import torch.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import torch
import numpy as np
from datasets import (
    MnistDataset,
    mnist_data_transform,
    show_images_batch,
    cifar_data_transform,
    CifarDataset,
    celeba_data_transform,
    CelebADataset,
)
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
import tqdm
import time
from model_helpers import save_model, load_model
import wandb
from PIL import Image


@torch.no_grad()
def save_image_2(
    tensor,
    fp,
    format=None,
    **kwargs,
) -> None:
    """
    Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """

    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    ndarr = (
        grid.sub_(1)
        .mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)


def get_noise_schedule(num_steps, start=0.0001, end=0.02):
    return torch.linspace(start, end, num_steps)


def forward_diffusion_gbm(log_x0, t, sigmas, device="cpu"):
    noise = torch.randn_like(log_x0).to(device)

    scale_t = torch.gather(sigmas, 1, t)
    scale_t = scale_t[:, :, None, None]
    scale_0 = torch.gather(sigmas, 1, torch.zeros_like(t))
    scale_0 = scale_0[:, :, None, None]

    var = scale_t - scale_0
    mean = log_x0 - 0.5 * var
    # mean = log_x0
    log_xt = mean + torch.sqrt(var) * noise
    # log_xt = torch.clamp(log_xt, 0, np.log(2)) # clip to valid range
    return log_xt.to(device), noise, var


# TODO: finish this sample function
@torch.no_grad()
def one_step_denoising(
    model,
    log_x,
    t,
    sigmas,
    device="cpu",
):
    scale_t = torch.gather(sigmas, 1, t + 1)
    scale_t = scale_t[:, :, None, None]

    scale_0 = torch.gather(sigmas, 1, torch.zeros_like(t))
    scale_0 = scale_0[:, :, None, None]

    scale_t_prev = torch.gather(sigmas, 1, t)
    scale_t_prev = scale_t_prev[:, :, None, None]

    z = torch.randn_like(log_x)

    noise_pred = model(log_x, t[0])
    noise_pred = torch.clamp(noise_pred, -1, 1)

    var = scale_t - scale_0
    var_t = scale_t - scale_t_prev
    mean = log_x + 0.5 * var_t
    # mean = log_x

    log_x_out = mean + var_t/torch.sqrt(var) * noise_pred + torch.sqrt(var_t) * z
    # log_x_out = torch.clamp(log_x_out, 0, np.log(2))  # clip to valid range

    return log_x_out.to(device)


@torch.no_grad()
def sample_noise_removal(
    x_0: torch.Tensor,
    model: UNet,
    timesteps: int,
    sigmas: torch.Tensor,
    device: str = "cpu",
):
    x = x_0
    for t_ in tqdm.tqdm(reversed(range(timesteps - 1)), total=timesteps - 1):
        t = torch.ones(size=(x.shape[0], 1), dtype=int) * t_
        t = t.to(device)
        x = one_step_denoising(model, x, t, sigmas, device=device)
    # x = torch.clamp(x, 0, np.log(2))
    return x


if __name__ == "__main__":
    for i in range(torch.cuda.device_count()):
        print(
            "found GPU",
            torch.cuda.get_device_name(i),
            "capability",
            torch.cuda.get_device_capability(i),
        )

    # # Init dataset Mnist
    # batch_size = 256

    # tf = transforms.Compose(
    #     [transforms.Resize((32, 32)), transforms.ToTensor(),
    #      transforms.Lambda(lambda t: t + 0.5)]  # to be able to take log
    # )  # mnist is already normalised 0 to 1

    # data = MNIST("./data", train=True, download=True, transform=tf)
    # loader = DataLoader(
    #     data, batch_size=batch_size, shuffle=True, num_workers=5, drop_last=True
    # )

    # # Init dataset Cifar10
    # batch_size = 256
    # data_transform = cifar_data_transform()
    # data = CifarDataset(
    #     img_dir="/home/anvuong/Desktop/datasets/CIFAR-10-images/train",
    #     classes="all",
    #     transform=data_transform,
    # )
    # loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=5, drop_last=True)

    # # Init dataset celeba
    img_size = 128
    batch_size = 28
    data_transform = celeba_data_transform(img_size)
    data = CelebADataset(
        # img_dir="/nfs/stak/users/vuonga2/datasets/img_celeba",
        # img_dir="/nfs/stak/users/vuonga2/datasets/img_celeba_tiny",
        img_dir="/root/datasets/img_celeba_small",
        transform=data_transform,
    )
    loader = DataLoader(
        data, batch_size=batch_size, shuffle=True, num_workers=10, drop_last=True
    )

    # Init model
    device = "cuda:0"
    # unet_conf = {
    #     "init_channels": 64,
    #     "in_channels": 1,
    #     "out_channels": 1,
    #     "num_res_blocks": 2,
    #     "attn_resolutions": (
    #         64,
    #         16,
    #     ),
    #     "input_img_resolution": 128,
    #     "channels_multipliers": (1, 2, 4, 8),
    # }
    unet_conf = {
        "init_channels": 128,
        "in_channels": 1,
        "out_channels": 1,
        "num_res_blocks": 2,
        "attn_resolutions": (16,),
        "input_img_resolution": img_size,
        "channels_multipliers": (1, 1, 2, 2, 4, 4),
    }
    model = UNet(**unet_conf).to(device)

    total_params = sum([p.numel() for p in model.parameters()])
    print("Model initialized, total params = ", total_params)

    # Init diffusion params
    T = 1000
    sigmas = get_noise_schedule(T)
    # repeat into batch_size for easier indexing by t
    sigmas = sigmas.unsqueeze(0).repeat(batch_size, 1).to(device)

    # torch.autograd.set_detect_anomaly(True)

    # # Get a sample batch
    x = next(iter(loader))
    x = x[0]
    # print(x.shape)
    t = torch.randint(low=1, high=T, size=(batch_size, 1))
    print(x.shape)
    # Exp name
    exp_name = f"celeba-small-noise-removal-2-{img_size}"
    if not os.path.exists(os.path.join("models", exp_name)):
        os.makedirs(os.path.join("models", exp_name))
    if not os.path.exists(f"./sampling_images/{exp_name}"):
        os.makedirs(f"./sampling_images/{exp_name}")
    # init tensorboard writer
    current_time = time.time()

    # train params
    epochs = 500
    save_every = 10
    start_epoch = 0
    # load_epoch = start_epoch - 1
    load_model = False
    # load_model_path = f"models/{exp_name}/model_{load_epoch}.pkl"
    load_model_path = "models/celeba-tiny-noise-removal-128/second-run/model_99.pkl"

    # load from save
    if load_model:
        print(f"loading model from epoch {load_model_path}")
        model.load_state_dict(
            torch.load(
                load_model_path,
                map_location=device,
            )
        )

    # re-init dataloader
    # loader = DataLoader(data, batch_size=batch_size, drop_last=True)
    loader = DataLoader(
        data, batch_size=batch_size, shuffle=True, num_workers=5, drop_last=True
    )

    # Init optimizer
    lr = 1e-6
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # lr_schedule = LinearLR(opt, total_iters=epochs)
    lr_schedule = ReduceLROnPlateau(opt, mode="min", patience=5, factor=0.75)
    opt.zero_grad()

    # TODO: currently loss keeps increasing with epochs, shit is happening
    # TODO: need to fix this
    total = len(loader)
    print("num batches = ", total)
    wandb.init(
        project=exp_name,
        config={
            "load_model": load_model,
            "learning_rate": lr,
            "epochs": epochs,
            "start_epoch": start_epoch,
            "unet_config": unet_conf,
            "num_steps": T,
        },
    )

    # mixed precision
    for e in np.arange(start_epoch, start_epoch + epochs):
        loss_ema = None
        pbar = tqdm.tqdm(enumerate(loader), total=total)
        for idx, log_x in pbar:
            log_x = log_x[0]
            log_x = log_x.to(device)
            t = torch.randint(low=0, high=T, size=(batch_size, 1)).to(device)

            log_x_in, noise_in, var_in = forward_diffusion_gbm(
                log_x, t, sigmas, device=device
            )

            noise_pred = model(log_x_in, t.squeeze())
            loss = torch.nn.functional.mse_loss(noise_in, noise_pred, reduction="mean")
            loss.backward()

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
            # if idx == 5:
            #     break

        # linear lrate decay
        # opt.param_groups[0]["lr"] = lr * (1 - e / epochs)
        wandb.log({"learning_rate":opt.param_groups[-1]["lr"]})
        lr_schedule.step(loss)

        print(f"epoch {e} loss={loss.item()}")

        # show image and save model every 100 epochs
        if e % save_every == 0 or e == epochs-1:
            save_model(f"models/{exp_name}/model_{e}.pkl", model)
            print("Generating sample images")
            x_org = torch.exp((log_x+1)*np.log(2)/2)
            save_image_2(torch.exp(x_org), f"sampling_images/{exp_name}/original_epoch_{e}.png")
            x_noised = torch.exp((log_x_in+1)*np.log(2)/2)
            save_image_2(x_noised, f"sampling_images/{exp_name}/noised_epoch_{e}.png")
            xs = []
            for j in range(4): # use 4 images only
                log_x0 = log_x_in[j]
                log_x0 = log_x0[None, :, :, :]
                K = t[j][0].to("cpu").item()
                log_x_denoised = sample_noise_removal(
                    log_x0,
                    model,
                    K,
                    sigmas,
                    device=device,
                )
                x_denoised = torch.exp((log_x_denoised+1)*np.log(2)/2)
                x_denoised = x_denoised.to("cpu")
                xs.append(x_denoised)
            xsout = torch.cat(xs)
            save_image_2(xsout, f"sampling_images/{exp_name}/sample_epoch_{e}.png")
            save_image_2(xsout, f"sampling_images/{exp_name}/latest.png")
