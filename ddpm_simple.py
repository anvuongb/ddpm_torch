import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from unet_simple_2 import Unet
from torchvision import models, transforms
from torchvision.utils import save_image, make_grid
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import os


def noise_schedule(beta1, beta2, T):
    assert beta1 < beta2 < 1.0
    # linear schedule
    beta_t = torch.linspace(beta1, beta2, T + 1)

    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,
        "oneover_sqrta": oneover_sqrta,
        "sqrt_beta_t": sqrt_beta_t,
        "alphabar_t": alphabar_t,
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,
    }


class DDPM(nn.Module):
    def __init__(self, nn_model, beta1, beta2, T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        for k, v in noise_schedule(beta1, beta2, T).items():
            self.register_buffer(k, v)

        self.T = T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x):
        # Uniformly sample time t and sample gaussian noise
        ts = torch.randint(1, self.T + 1, (x.shape[0],)).to(self.device)
        noise = torch.randn_like(x)

        x_t = (
            self.sqrtab[ts, None, None, None] * x
            + self.sqrtmab[ts, None, None, None] * noise
        )

        noise_pred = self.nn_model(x_t, ts / self.T)
        return self.loss_mse(noise, noise_pred)

    def sample(self, n_sample, size, device, return_step=False):
        # start from noise
        xi = torch.randn(n_sample, *size).to(device)

        xi_s = []
        for i in range(self.T, 0, -1):
            ti = torch.tensor([i / self.T]).to(device)
            ti = ti.repeat(n_sample, 1, 1, 1)

            noise_pred = self.nn_model(xi, ti)
            z = torch.randn(n_sample, *size).to(device)
            if i <= 1:
                z = 0

            # ancestral sampling
            xi = (
                self.oneover_sqrta[i] * (xi - noise_pred * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )

            if return_step and (i % 20 == 0 or i == self.T or i < 8):
                xi_s.append(xi.detach().cpu().numpy())
        xi_s = np.array(xi_s)
        return xi, xi_s


def train_mnist():
    # hardcoding these here
    n_epoch = 100
    batch_size = 256
    n_T = 400  # 500
    device = "cuda:0"
    n_feat = 128  # 128 ok, 256 better (but slower)
    lrate = 1e-4
    save_model = True
    save_dir = "./data/diffusion_outputs10_2/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    embed_method = "cosine"

    ddpm = DDPM(
        nn_model=Unet(in_channels=1, n_features=n_feat, embed_method=embed_method),
        beta1=1e-4,
        beta2=0.02,
        T=n_T,
        device=device,
        drop_prob=0.1,
    )
    total_params = sum([p.numel() for p in ddpm.parameters()])
    print("Model initialized, total params = ", total_params)
    ddpm.to(device)

    # optionally load a model
    # ddpm.load_state_dict(torch.load("./data/diffusion_outputs/dd10pm_unet01_mnist_9.pth"))

    tf = transforms.Compose(
        [transforms.ToTensor()]
    )  # mnist is already normalised 0 to 1

    dataset = MNIST("./data", train=True, download=True, transform=tf)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    wandb.init(
        project="mnist-all-ddpm-simple",
        config={
            # "load_model": load_model,
            "learning_rate": lrate,
            "epochs": n_epoch,
            # "start_epoch": start_epoch,
            "batch_size": batch_size,
            "num_features": n_feat,
            "embed_method":embed_method
        },
    )

    for ep in range(n_epoch):
        print(f"epoch {ep}")
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]["lr"] = lrate * (1 - ep / n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(device)
            loss = ddpm(x)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

            wandb.log({"loss": loss.item()})

        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        ddpm.eval()
        n_rows = 10
        with torch.no_grad():
            n_sample = 4 * n_rows
            x_gen, x_gen_store = ddpm.sample(
                n_sample, (1, 28, 28), device, return_step=True
            )

            # append some real images at bottom, order by class also
            x_real = torch.Tensor(x_gen.shape).to(device)
            for k in range(n_rows):
                for j in range(int(n_sample / n_rows)):
                    x_real[k + (j * n_rows)] = x[k + (j * n_rows)]

            x_all = torch.cat([x_gen, x_real])
            grid = make_grid(x_all * -1 + 1, nrow=10)
            save_image(grid, save_dir + f"image_ep{ep}.png")
            print("saved image at " + save_dir + f"image_ep{ep}.png")
            wandb.save(save_dir + f"image_ep{ep}.png")

            if ep % 5 == 0 or ep == int(n_epoch - 1):
                # create gif of images evolving over time, based on x_gen_store
                fig, axs = plt.subplots(
                    nrows=int(n_sample / n_rows),
                    ncols=n_rows,
                    sharex=True,
                    sharey=True,
                    figsize=(8, 3),
                )

                def animate_diff(i, x_gen_store):
                    print(
                        f"gif animating frame {i} of {x_gen_store.shape[0]}",
                        end="\r",
                    )
                    plots = []
                    for row in range(int(n_sample / n_rows)):
                        for col in range(n_rows):
                            axs[row, col].clear()
                            axs[row, col].set_xticks([])
                            axs[row, col].set_yticks([])
                            # plots.append(axs[row, col].imshow(x_gen_store[i,(row*n_rows)+col,0],cmap='gray'))
                            plots.append(
                                axs[row, col].imshow(
                                    -x_gen_store[i, (row * n_rows) + col, 0],
                                    cmap="gray",
                                    vmin=(-x_gen_store[i]).min(),
                                    vmax=(-x_gen_store[i]).max(),
                                )
                            )
                    return plots

                ani = FuncAnimation(
                    fig,
                    animate_diff,
                    fargs=[x_gen_store],
                    interval=200,
                    blit=False,
                    repeat=True,
                    frames=x_gen_store.shape[0],
                )
                ani.save(
                    save_dir + f"gif_ep{ep}.gif",
                    dpi=100,
                    writer=PillowWriter(fps=5),
                )
                print("saved image at " + save_dir + f"gif_ep{ep}.gif")
                wandb.save(save_dir + f"gif_ep{ep}.gif")

        # optionally save model
        if save_model and ep > 0 and (ep % 10 == 0 or ep == int(n_epoch - 1)):
            torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")
            print("saved model at " + save_dir + f"model_{ep}.pth")


if __name__ == "__main__":
    train_mnist()
