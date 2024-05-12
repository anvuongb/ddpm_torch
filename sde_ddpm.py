import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import scipy
from multiprocessing import Pool
import statsmodels.api as sm
import wandb
import pickle


# build an extremely simple NN to predict 1D noise
class SimpleModel(nn.Module):
    def __init__(self, hidden_units=32):
        super(SimpleModel, self).__init__()
        self.t_emb = nn.Sequential(
            nn.Linear(1, hidden_units), nn.GELU(), nn.Linear(hidden_units, hidden_units)
        )

        self.input_emb = nn.Sequential(
            nn.Linear(2, hidden_units),
            nn.BatchNorm1d(hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, hidden_units),
            nn.Tanh(),
        )

        self.out1 = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.BatchNorm1d(hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, hidden_units),
            nn.Tanh(),
        )

        self.out2 = nn.Sequential(
            nn.Linear(hidden_units, hidden_units), nn.Tanh(), nn.Linear(hidden_units, 2)
        )

    def forward(self, x, t):
        t = t.view(-1, 1)
        t = self.t_emb(t)

        x = x.view(-1, 2)
        x = self.input_emb(x)

        # print(x.shape, t.shape)
        o1 = self.out1(x + t)
        o2 = self.out2(o1 + x + t)
        return o2


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
    def __init__(self, nn_model, beta1, beta2, T, device):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        for k, v in noise_schedule(beta1, beta2, T).items():
            self.register_buffer(k, v)

        self.T = T
        self.device = device
        self.loss_mse = nn.MSELoss()

    def forward(self, x):
        # Uniformly sample time t and sample gaussian noise
        ts = torch.randint(1, self.T + 1, (x.shape[0],)).to(self.device)
        noise = torch.randn_like(x)

        x_t = self.sqrtab[ts, None] * x + self.sqrtmab[ts, None] * noise

        # print(x.shape, ts.shape)
        noise_pred = self.nn_model(x_t, ts / self.T)
        # print(noise_pred.shape, noise.shape)
        return self.loss_mse(noise, noise_pred)

    def sample(self, n_sample, size, device, return_step=False):
        # start from noise
        xi = torch.randn(n_sample, *size).to(device)

        xi_s = []
        for i in range(self.T, 0, -1):
            ti = torch.tensor([i / self.T]).to(device)
            ti = ti.repeat(n_sample, 1)

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


if __name__ == "__main__":

    with open("./data/2d_gaussians/train.pickle", "rb") as f:
        s = pickle.load(f)

    device = "cuda:0"
    T = 500
    lrate = 1e-4
    n_epoch = 250

    n_feat = 256
    batch_size = 8192
    beta1 = 0.0001
    beta2 = 0.02

    wandb.init(
        project="2D Gaussian",
        config={
            "learning_rate": lrate,
            "epochs": n_epoch,
            # "start_epoch": start_epoch,
            "batch_size": batch_size,
            "num_features": n_feat,
            "beta1": beta1,
            "beta2": beta2,
            "T": T,
        },
    )

    ddpm = DDPM(
        nn_model=SimpleModel(n_feat),
        beta1=beta1,
        beta2=beta2,
        T=T,
        device=device,
    )
    # ddpm.load_state_dict(torch.load("./data/2d_gaussians/model_150.pth"))

    total_params = sum([p.numel() for p in ddpm.parameters()])
    print("Model initialized, total params = ", total_params)
    ddpm.to(device)

    data = s
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=5)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    for ep in range(n_epoch):
        print(f"epoch {ep}")
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]["lr"] = lrate * (1 - ep / n_epoch)

        pbar = tqdm.tqdm(dataloader)
        loss_ema = None
        for x in pbar:
            optim.zero_grad()
            x = x.type(torch.FloatTensor)
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

        # if ep
    torch.save(ddpm.state_dict(), f"data/2d_gaussians/model_{n_epoch}.pth")
