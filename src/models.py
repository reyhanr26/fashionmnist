import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. VANILLA AUTOENCODER (Latent Dim = 64)
# =====================================================================
class VanillaAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder(x.view(-1, 784))

    def decode(self, z):
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


# =====================================================================
# 2. SPARSE AUTOENCODER (Sparse Hidden Dim = 256, L1 Penalty)
# =====================================================================
class SparseAE(nn.Module):
    def __init__(self, sparse_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, sparse_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(sparse_dim, 784),
            nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder(x.view(-1, 784))

    def decode(self, h):
        return self.decoder(h).view(-1, 1, 28, 28)

    def forward(self, x):
        h = self.encode(x)
        recon = self.decode(h)
        return recon, h


# =====================================================================
# 3. VARIATIONAL AUTOENCODER / VAE (Latent Dim = 64)
# =====================================================================
class VAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = F.relu(self.fc1(x.view(-1, 784)))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# =====================================================================
# 4. VANILLA GAN (MLP-Based, Prior Noise = 100)
# =====================================================================
class VanillaGenerator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)


class VanillaDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x.view(-1, 784))


# =====================================================================
# 5. DEEP CONVOLUTIONAL GAN / DCGAN (Prior Noise = 100, 28x28)
# =====================================================================
class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(True)
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 128, 7, 7)
        return self.conv(h)


class DCGANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1)
        )

    def forward(self, x):
        return self.net(x)


def weights_init_dcgan(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# =====================================================================
# 6. denoising diffusion
# =====================================================================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class SimpleUNet(nn.Module):
    def __init__(self, time_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU()
        )
        # encodeing
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.t_proj1 = nn.Linear(time_dim, 32)
        self.down1 = nn.Conv2d(32, 64, 4, stride=2, padding=1)  # 14x14

        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.t_proj2 = nn.Linear(time_dim, 64)
        self.down2 = nn.Conv2d(64, 128, 4, stride=2, padding=1) # 7x7

        # bottlenck
        self.bot = nn.Conv2d(128, 128, 3, padding=1)
        self.t_proj_bot = nn.Linear(time_dim, 128)

        # decoding
        self.up1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1) # 14x14
        self.dec1 = nn.Conv2d(128, 64, 3, padding=1)

        self.up2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1) # 28x28
        self.dec2 = nn.Conv2d(64, 32, 3, padding=1)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        h1 = F.gelu(self.conv1(x) + self.t_proj1(t_emb)[:, :, None, None])
        d1 = self.down1(h1)

        h2 = F.gelu(self.conv2(d1) + self.t_proj2(t_emb)[:, :, None, None])
        d2 = self.down2(h2)

        b = F.gelu(self.bot(d2) + self.t_proj_bot(t_emb)[:, :, None, None])

        u1 = self.up1(b)
        u1 = F.gelu(self.dec1(torch.cat([u1, h2], dim=1)))

        u2 = self.up2(u1)
        u2 = F.gelu(self.dec2(torch.cat([u2, h1], dim=1)))
        return self.out(u2)


class DDPM(nn.Module):
    def __init__(self, timesteps=300, beta_start=1e-4, beta_end=0.02, device='cuda'):
        super().__init__()
        self.timesteps = timesteps
        self.device = device
        self.model = SimpleUNet().to(device)

        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

    def forward_diffusion(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise, noise

    def compute_loss(self, x_0):
        b = x_0.size(0)
        t = torch.randint(0, self.timesteps, (b,), device=x_0.device).long()
        x_t, noise = self.forward_diffusion(x_0, t)
        pred_noise = self.model(x_t, t)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, num_samples, return_trajectory=False):
        self.model.eval()
        x = torch.randn(num_samples, 1, 28, 28, device=self.device)
        trajectory = [x.cpu()] if return_trajectory else None
        capture_steps = set(torch.linspace(self.timesteps - 1, 0, 8).long().tolist())

        for i in reversed(range(self.timesteps)):
            t = torch.full((num_samples,), i, device=self.device, dtype=torch.long)
            pred_noise = self.model(x, t)

            alpha = self.alphas[i]
            alpha_cumprod = self.alphas_cumprod[i]
            beta = self.betas[i]

            noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
            x = (1 / torch.sqrt(alpha)) * (
                x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * pred_noise
            ) + torch.sqrt(beta) * noise

            if return_trajectory and i in capture_steps:
                trajectory.append(x.cpu())

        x = (x.clamp(-1, 1) + 1) / 2.0
        if return_trajectory:
            trajectory = [(img.clamp(-1, 1) + 1) / 2.0 for img in trajectory]
            return x, trajectory
        return x
