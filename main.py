import torch
from src.dataset import set_seed, get_fashion_mnist_loaders
from src.mlp_proof import run_mlp_experiment
from src.models import (
    VanillaAE, SparseAE, VAE,
    VanillaGenerator, VanillaDiscriminator,
    DCGANGenerator, DCGANDiscriminator, weights_init_dcgan,
    DDPM
)
from src.train import train_ae, train_vae, train_gan, train_ddpm
from src.evaluate import evaluate_reconstruction
from src.visualize import generate_all_figures

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_fashion_mnist_loaders(batch_size=128, seed=42)

    # 1. Pembuktian Batasan MLP Classifier
    mlp_model, frozen_decoder, mlp_results = run_mlp_experiment(train_loader, test_loader, device, epochs=15)

    # 2. Pelatihan 6 Model Generatif
    vanilla_ae = VanillaAE(latent_dim=64)
    hist_vae_ae = train_ae(vanilla_ae, train_loader, device, epochs=20)

    sparse_ae = SparseAE(sparse_dim=256)
    hist_sparse_ae = train_ae(sparse_ae, train_loader, device, epochs=20, is_sparse=True, l1_lambda=1e-4)

    vae = VAE(latent_dim=64)
    hist_vae = train_vae(vae, train_loader, device, epochs=20)

    vgan_G, vgan_D = VanillaGenerator(100), VanillaDiscriminator()
    hist_vgan = train_gan(vgan_G, vgan_D, train_loader, device, epochs=25, latent_dim=100)

    dcgan_G, dcgan_D = DCGANGenerator(100), DCGANDiscriminator()
    dcgan_G.apply(weights_init_dcgan)
    dcgan_D.apply(weights_init_dcgan)
    hist_dcgan = train_gan(dcgan_G, dcgan_D, train_loader, device, epochs=25, latent_dim=100)

    ddpm = DDPM(timesteps=300, device=device)
    hist_ddpm = train_ddpm(ddpm, train_loader, device, epochs=20)

    histories = {
        'Vanilla AE': hist_vae_ae, 'Sparse AE': hist_sparse_ae, 'VAE': hist_vae,
        'Vanilla GAN': hist_vgan, 'DCGAN': hist_dcgan, 'DDPM': hist_ddpm
    }

    # 3. Evaluasi Metrik Test Set
    print("Vanilla AE:", evaluate_reconstruction(vanilla_ae, test_loader, device))
    print("Sparse AE :", evaluate_reconstruction(sparse_ae, test_loader, device))
    print("VAE       :", evaluate_reconstruction(vae, test_loader, device))

    # 4. Generate 5 Figur Komparatif
    generate_all_figures(
        test_loader, mlp_results,
        vanilla_ae, sparse_ae, vae, vgan_G, dcgan_G, ddpm,
        histories, device
    )

if __name__ == "__main__":
    main()
