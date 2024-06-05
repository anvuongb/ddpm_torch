from unet import UNet
from torchvision import transforms
import os
from unet_simple import UNetSimple
import torch.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import writer
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import torch
import numpy as np
from datasets import MnistDataset, mnist_data_transform, show_images_batch, cifar_data_transform, CifarDataset, celeba_data_transform, CelebADataset
from torch.optim.lr_scheduler import LinearLR
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
    ndarr = grid.sub_(1).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)

def get_noise_schedule(num_steps, start=0.0001, end=0.02):
    return torch.linspace(start, end, num_steps)

def forward_diffusion_gbm(x_0, t, sigmas, device="cpu"):
    noise = torch.randn_like(x_0).to(device)

    log_x0 = torch.log(x_0)
    
    scale_t = torch.gather(sigmas, 1, t)
    scale_t = scale_t[:, :, None, None]
    scale_0 = torch.gather(sigmas, 1, torch.zeros_like(t))
    scale_0 = scale_0[:, :, None, None]

    mean = log_x0 -0.5*(scale_t - scale_0)
    var = torch.sqrt(scale_t-scale_0)
    log_xt = mean + var*noise
    return log_xt.to(device), noise, var


# TODO: finish this sample function
@torch.no_grad()
def one_step_denoising(
    model,
    x,
    t,
    sigmas,
    device="cpu",
):
    scale_t = torch.gather(sigmas, 1, t)
    scale_t = scale_t[:, :, None, None]

    scale_0 = torch.gather(sigmas, 1, torch.zeros_like(t))
    scale_0 = scale_0[:, :, None, None]

    scale_t_prev = torch.gather(sigmas, 1, t-1)
    scale_t_prev = scale_t_prev[:, :, None, None]

    z = torch.randn_like(x)

    # print(t)
    var = torch.sqrt(scale_t-scale_0)
    noise_pred = model(x, t[0])
    noise_pred = torch.clamp(noise_pred, -1, 1)
    noise_pred = noise_pred/var

    var_t = scale_t-scale_t_prev
    x_out = x + 0.5*var_t*(1+2*noise_pred)+torch.sqrt(var_t)*z

    # noise = beta scale

    return x_out.to(device)


@torch.no_grad()
def sample_noise_removal(
    x_0: torch.Tensor,
    model: UNet,
    timesteps: int,
    sigmas: torch.Tensor,
    device: str = "cpu",
):
    # x = torch.randn_like(x_0).to(device)
    x = x_0
    for t_ in tqdm.tqdm(reversed(range(1, timesteps)), total=timesteps-1):
        t = torch.ones(size=(x.shape[0], 1), dtype=int) * t_
        t = t.to(device)
        x = one_step_denoising(
            model, x, t, sigmas, device=device
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
    batch_size = 40
    data_transform = celeba_data_transform(128)
    data = CelebADataset(
        img_dir="/nfs/stak/users/vuonga2/datasets/img_celeba",
        transform=data_transform,
    )
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=5, drop_last=True)


    # Init model
    device = "cuda:0"
    unet_conf = {
        "init_channels": 64,
        "in_channels": 1,
        "out_channels": 1,
        "num_res_blocks": 2,
        "attn_resolutions": (32, 16,),
        "input_img_resolution": 128,
        "channels_multipliers": (2, 2, 2),
    }
    model = UNet(**unet_conf).to(device)

    total_params = sum([p.numel() for p in model.parameters()])
    print("Model initialized, total params = ", total_params)

    # Init diffusion params
    T = 500
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
    exp_name = "celeba-noise-removal"
    if not os.path.exists(os.path.join("models", exp_name)):
        os.makedirs(os.path.join("models", exp_name))
    if not os.path.exists(f"./sampling_images/{exp_name}"):
        os.makedirs(f"./sampling_images/{exp_name}")
    # init tensorboard writer
    current_time = time.time()
    # tb_writer = writer.SummaryWriter(f"logdir/{exp_name}_{current_time}")
    tb_writer = writer.SummaryWriter(f"logdir/{exp_name}")
    tb_writer.add_graph(
        model=model, input_to_model=[x.to(device), t.squeeze().to(device)]
    )

    # train params
    epochs = 100
    start_epoch = 10
    load_epoch = start_epoch - 1
    load_model = True

    # load from save
    if load_model:
        print(f"loading model from epoch {load_epoch}")
        model.load_state_dict(
            torch.load(
                f"models/celeba-noise-removal/model_{load_epoch}.pkl",
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
    lr = 1e-7
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # lr_schedule = LinearLR(opt, total_iters=epochs)
    opt.zero_grad()

    # TODO: currently loss keeps increasing with epochs, shit is happening
    # TODO: need to fix this
    total = len(loader)
    print("num batches = ", total)
    wandb.init(
        project="noise-removal",
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

            log_x_in, noise_in, var_in = forward_diffusion_gbm(x, t, sigmas, device=device)
            # x_in, noise_in = forward_diffusion(x, t, alphas_cum, device=device)
            # x_out = forward_diffusion(x, t, alphas_cum, device=device)

            noise_pred = model(log_x_in, t.squeeze())
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
            # if idx == 5:
            #     break

        # linear lrate decay
        opt.param_groups[0]["lr"] = lr * (1 - e / epochs)
        # lr_schedule.step()

        print(f"epoch {e} loss={loss.item()}")
        if e % 1 == 0:
            save_model(f"models/{exp_name}/model_{e}.pkl", model)

        # show image and save model every 100 epochs
        if e % 1 == 0:
            print("Generating sample images")
            save_image_2(x, f"sampling_images/{exp_name}/original_epoch_{e}.png")
            x_noised = torch.exp(log_x_in)
            save_image_2(x_noised, f"sampling_images/{exp_name}/noised_epoch_{e}.png")
            xs = []
            for j in range(batch_size):
                # x_in = torch.ones(16, *x.shape[1:])
                x_in = log_x_in[j]
                x_in = x_in[None, :, :, :]
                # print(x_in.shape)
                K = t[j][0].to('cpu').item()
                x_denoised = sample_noise_removal(
                    x_in,
                    model,
                    K,
                    sigmas,
                    device=device,
                )
                x_denoised = x_denoised.to("cpu")
                x_denoised = torch.exp(x_denoised)
                xs.append(x_denoised)
            xs = torch.cat(xs)
            # show_images_batch(
            #     f"sampling_images/{exp_name}/sample_epoch_{e}.png", x_denoised
            # )
            # show_images_batch(f"sampling_images/{exp_name}/latest.png", x_denoised)
            save_image(xs, f"sampling_images/{exp_name}/sample_epoch_{e}.png")
            save_image(xs, f"sampling_images/{exp_name}/latest.png")
            wandb.save(f"sampling_images/{exp_name}/*.png")
            save_model(f"models/{exp_name}/model.pkl", model)
