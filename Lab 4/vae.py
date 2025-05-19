import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from typing import Tuple

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_folder_path: str = os.path.join(os.path.dirname(__file__), 'imgs_')
os.makedirs(img_folder_path, exist_ok=True)


class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 2 * latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return h[:, :self.latent_dim], h[:, self.latent_dim:]

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def loss(self, x_hat: torch.Tensor, x: torch.Tensor,
             mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        kl_div = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
        kl_div = kl_div.sum(dim=-1)
        reconstruction = F.binary_cross_entropy(
            x_hat, x.view(-1, self.input_dim), reduction='none'
        ).sum(dim=-1)
        total_loss = kl_div + reconstruction
        return total_loss.mean()

    def sample(self, n: int) -> torch.Tensor:
        z = torch.randn(n, self.latent_dim).to(device)
        return self.decode(z)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self(x)[0]


def train(model: VAE, train_loader: DataLoader,
          optimizer: optim.Optimizer, num_epochs: int) -> VAE:
    model.to(device)
    model.train()

    rec = next(iter(train_loader))[0][:10].to(device)

    for epoch in range(num_epochs):
        for x, _ in train_loader:
            x = x.to(device)
            x = x.view(-1, model.input_dim)
            x_hat, mu, log_var = model(x)
            loss = model.loss(x_hat, x, mu, log_var)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}/{num_epochs}, Loss {loss.item():.4f}', end='\r')

        # plot samples
        samples = model.sample(64).view(-1, 1, 28, 28).cpu()
        grid = make_grid(samples, nrow=8, normalize=True)
        plt.imshow(grid.permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.savefig(os.path.join(img_folder_path, f'vae_samples_{epoch}.png'))
        plt.close()

        # plot reconstructions
        rec_hat = model(rec.view(-1, model.input_dim))[0]
        rec_hat = rec_hat.view(-1, 1, 28, 28).cpu()
        imgs = torch.cat([rec.cpu(), rec_hat], dim=0)
        grid = make_grid(imgs, nrow=10, normalize=True)
        plt.imshow(grid.permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.savefig(os.path.join(img_folder_path, f'vae_reconstructions_{epoch}.png'))
        plt.close()

    return model


def train_loader(batch_size: int) -> DataLoader:
    return DataLoader(
        datasets.MNIST(
            root='~/scratch/datasets/', # Change this to your dataset path
            train=True,
            download=True,
            transform=transforms.ToTensor()
        ),
        batch_size=batch_size,
        shuffle=True
    )


if __name__ == '__main__':
    input_dim: int = 28 * 28
    latent_dim: int = 10
    model = VAE(input_dim, latent_dim)
    loader = train_loader(batch_size=128)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs: int = 10
    trained_model = train(model, loader, optimizer, num_epochs)
