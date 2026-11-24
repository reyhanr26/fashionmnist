import time
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim_fn
from torchmetrics.image.fid import FrechetInceptionDistance

@torch.no_grad()
def evaluate_reconstruction(model, test_loader, device, is_mlp_frozen=False, mlp_model=None):
    """Menghitung rata-rata Test MSE, Test PSNR (dB), dan Test SSIM pada seluruh Test Set."""
    model.eval()
    if mlp_model is not None:
        mlp_model.eval()

    mse_list, psnr_list, ssim_list = [], [], []

    for x, _ in test_loader:
        x = x.to(device)
        if is_mlp_frozen:
            _, z = mlp_model(x)
            recon = model(z).view(-1, 1, 28, 28)
        elif hasattr(model, 'reparameterize'):  # VAE
            mu, _ = model.encode(x)
            recon = model.decode(mu)
        else:  # Vanilla AE / Sparse AE
            recon, _ = model(x)

        x_np = x.cpu().numpy()
        recon_np = recon.clamp(0, 1).cpu().numpy()

        for i in range(x_np.shape[0]):
            orig_img = x_np[i, 0]
            rec_img = recon_np[i, 0]
            mse_val = float(np.mean((orig_img - rec_img) ** 2))
            psnr_val = float(10.0 * np.log10(1.0 / (mse_val + 1e-10)))
            ssim_val = float(ssim_fn(orig_img, rec_img, data_range=1.0))

            mse_list.append(mse_val)
            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)

    return {
        'MSE': np.mean(mse_list),
        'PSNR': np.mean(psnr_list),
        'SSIM': np.mean(ssim_list)
    }


@torch.no_grad()
def measure_latency_batch64(model, model_type, device, latent_dim=64, runs=30):
    """Mengukur rata-rata Inference Latency (ms) untuk menghasilkan 1 batch berisi 64 citra."""
    model.eval()
    # Warm-up GPU
    for _ in range(3):
        if model_type in ['ae', 'vae', 'gan']:
            z = torch.randn(64, latent_dim, device=device)
            _ = model.decode(z) if hasattr(model, 'decode') else model(z)
        elif model_type == 'ddpm':
            _ = model.sample(64)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    actual_runs = 5 if model_type == 'ddpm' else runs
    latencies = []

    for _ in range(actual_runs):
        start_t = time.perf_counter()
        if model_type in ['ae', 'vae', 'gan']:
            z = torch.randn(64, latent_dim, device=device)
            _ = model.decode(z) if hasattr(model, 'decode') else model(z)
        elif model_type == 'ddpm':
            _ = model.sample(64)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start_t) * 1000.0)

    return float(np.mean(latencies))


@torch.no_grad()
def compute_fid_score(model, model_type, test_loader, device, latent_dim=64, num_samples=1024):
    """Menghitung Fréchet Inception Distance (FID) terhadap citra asli Test Set."""
    model.eval()
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    collected = 0
    for x, _ in test_loader:
        b = min(x.size(0), num_samples - collected)
        real_rgb = x[:b].to(device).repeat(1, 3, 1, 1).clamp(0, 1)
        fid.update(real_rgb, real=True)
        collected += b
        if collected >= num_samples:
            break

    generated = 0
    batch_size = 64
    while generated < num_samples:
        b = min(batch_size, num_samples - generated)
        if model_type == 'vae':
            z = torch.randn(b, latent_dim, device=device)
            fake = model.decode(z)
        elif model_type == 'gan':
            z = torch.randn(b, latent_dim, device=device)
            fake = (model(z) + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        elif model_type == 'ddpm':
            fake = model.sample(b)

        fake_rgb = fake.repeat(1, 3, 1, 1).clamp(0, 1)
        fid.update(fake_rgb, real=False)
        generated += b

    return float(fid.compute().item())
