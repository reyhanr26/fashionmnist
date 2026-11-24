import numpy as np
import torch
import matplotlib.pyplot as plt

def generate_all_figures(
    test_loader, mlp_results,
    vanilla_ae, sparse_ae, vae, vgan_G, dcgan_G, ddpm,
    histories, device
):
    vanilla_ae.eval()
    sparse_ae.eval()
    vae.eval()
    vgan_G.eval()
    dcgan_G.eval()
    ddpm.eval()

    # Ambil 8 sampel yang sama dengan eksperimen MLP untuk konsistensi visual
    sample_x = mlp_results['orig'].to(device)

    # =================================================================
    # FIGUR 1: Grid Rekonstruksi Side-by-Side (8 sampel x 4 baris)
    # =================================================================
    with torch.no_grad():
        rec_vae_ae, _ = vanilla_ae(sample_x)
        rec_sparse, _ = sparse_ae(sample_x)
        mu, _ = vae.encode(sample_x)
        rec_vae = vae.decode(mu)

    rows_fig1 = [
        ("Citra Asli (Ground Truth)", sample_x.cpu()),
        ("Vanilla AE (z=64)", rec_vae_ae.cpu()),
        ("Sparse AE (h=256, L1)", rec_sparse.cpu()),
        ("Variational AE (VAE, z=64)", rec_vae.cpu())
    ]

    fig, axes = plt.subplots(4, 8, figsize=(14, 7.2))
    for r_idx, (title, imgs) in enumerate(rows_fig1):
        for c_idx in range(8):
            ax = axes[r_idx, c_idx]
            ax.imshow(imgs[c_idx, 0], cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            if c_idx == 0:
                ax.text(-0.15, 0.5, title, transform=ax.transAxes,
                        fontsize=10, fontweight='bold', va='center', ha='right')
    plt.suptitle("Figur 1: Perbandingan Rekonstruksi Test Set (Vanilla AE, Sparse AE, dan VAE)",
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig("outputs/Figur_1_Rekonstruksi.png", dpi=300, bbox_inches='tight')
    plt.close()

    # =================================================================
    # FIGUR 2: Grid Random Generative Sampling (4 Panel x Kisi 4x4)
    # =================================================================
    torch.manual_seed(42)
    with torch.no_grad():
        samp_vae = vae.decode(torch.randn(16, 64, device=device)).cpu()
        samp_vgan = ((vgan_G(torch.randn(16, 100, device=device)) + 1) / 2).cpu()
        samp_dcgan = ((dcgan_G(torch.randn(16, 100, device=device)) + 1) / 2).cpu()
        samp_ddpm = ddpm.sample(16).cpu()

    panels_fig2 = [
        ("Panel A: VAE (z ~ N(0, I))", samp_vae),
        ("Panel B: Vanilla GAN (MLP)", samp_vgan),
        ("Panel C: DCGAN (Convolutional)", samp_dcgan),
        ("Panel D: DDPM (Reverse Denoising)", samp_ddpm),
    ]

    fig = plt.figure(figsize=(12, 12))
    outer = fig.add_gridspec(2, 2, wspace=0.15, hspace=0.22)
    for p_idx, (panel_title, imgs) in enumerate(panels_fig2):
        inner = outer[p_idx].subgridspec(4, 4, wspace=0.05, hspace=0.05)
        ax_title = fig.add_subplot(outer[p_idx])
        ax_title.set_title(panel_title, fontsize=11, fontweight='bold', pad=10)
        ax_title.axis('off')
        for k in range(16):
            ax = fig.add_subplot(inner[k // 4, k % 4])
            ax.imshow(imgs[k, 0], cmap='gray', vmin=0, vmax=1)
            ax.axis('off')

    plt.suptitle("Figur 2: Sintesis Citra Acak (4x4 Grid) dari Prior N(0, I) Antarmodel Generatif",
                 fontsize=13, fontweight='bold', y=0.96)
    plt.savefig("outputs/Figur_2_Generative_Sampling.png", dpi=300, bbox_inches='tight')
    plt.close()

    # =================================================================
    # FIGUR 3: Bukti Empiris Batasan Bobot & Latent Features MLP Classifier
    # =================================================================
    rows_fig3 = [
        ("Panel 1: Citra Asli (Ground Truth)", mlp_results['orig']),
        ("Panel 2a: Proyeksi W^T dari Kelas y_c", mlp_results['w_trans_class']),
        ("Panel 2b: Proyeksi W^T dari Laten Z_MLP", mlp_results['w_trans_z']),
        ("Panel 3a: Frozen Decoder (Z_MLP)", mlp_results['frozen_recon']),
        ("Panel 3b: Pembanding Vanilla AE (Z_AE)", rec_vae_ae.cpu()),
        ("Panel 4: Prior Sampling Ruang Laten MLP", mlp_results['prior_sample']),
    ]

    fig, axes = plt.subplots(6, 8, figsize=(14, 10.5))
    for r_idx, (title, imgs) in enumerate(rows_fig3):
        for c_idx in range(8):
            ax = axes[r_idx, c_idx]
            ax.imshow(imgs[c_idx, 0], cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            if c_idx == 0:
                ax.text(-0.15, 0.5, title, transform=ax.transAxes,
                        fontsize=9.5, fontweight='bold', va='center', ha='right')
    plt.suptitle("Figur 3: Pembuktian Empiris Batasan Bobot (W^T) dan Ruang Laten MLP Classifier",
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig("outputs/Figur_3_MLP_Proof.png", dpi=300, bbox_inches='tight')
    plt.close()

    # =================================================================
    # FIGUR 4: Linear Latent Space Interpolation (alpha = 0.0 -> 1.0)
    # =================================================================
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # Cari 2 sampel test dari kelas kontras (mis. kelas 0 T-shirt & kelas 9 Ankle Boot)
    all_x, all_y = next(iter(test_loader))
    idx_a = (all_y == 0).nonzero(as_tuple=True)[0][0].item()
    idx_b = (all_y == 9).nonzero(as_tuple=True)[0][0].item()
    img_a = all_x[idx_a:idx_a+1].to(device)
    img_b = all_x[idx_b:idx_b+1].to(device)

    torch.manual_seed(7)
    z_gan_a = torch.randn(1, 100, device=device)
    z_gan_b = torch.randn(1, 100, device=device)

    interp_rows = []
    with torch.no_grad():
        # 1. Vanilla AE
        za, zb = vanilla_ae.encode(img_a), vanilla_ae.encode(img_b)
        interp_rows.append(("Vanilla AE", [vanilla_ae.decode((1-a)*za + a*zb).cpu() for a in alphas]))
        # 2. Sparse AE
        ha, hb = sparse_ae.encode(img_a), sparse_ae.encode(img_b)
        interp_rows.append(("Sparse AE", [sparse_ae.decode((1-a)*ha + a*hb).cpu() for a in alphas]))
        # 3. VAE
        mu_a, _ = vae.encode(img_a)
        mu_b, _ = vae.encode(img_b)
        interp_rows.append(("VAE", [vae.decode((1-a)*mu_a + a*mu_b).cpu() for a in alphas]))
        # 4. Vanilla GAN
        interp_rows.append(("Vanilla GAN", [((vgan_G((1-a)*z_gan_a + a*z_gan_b)+1)/2).cpu() for a in alphas]))
        # 5. DCGAN
        interp_rows.append(("DCGAN", [((dcgan_G((1-a)*z_gan_a + a*z_gan_b)+1)/2).cpu() for a in alphas]))

    fig, axes = plt.subplots(5, 6, figsize=(11, 9))
    for r_idx, (m_name, imgs) in enumerate(interp_rows):
        for c_idx, a in enumerate(alphas):
            ax = axes[r_idx, c_idx]
            ax.imshow(imgs[c_idx][0, 0], cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            if r_idx == 0:
                ax.set_title(f"α = {a:.1f}", fontsize=10, fontweight='bold')
            if c_idx == 0:
                ax.text(-0.15, 0.5, m_name, transform=ax.transAxes,
                        fontsize=10, fontweight='bold', va='center', ha='right')
    plt.suptitle("Figur 4: Linear Latent Space Interpolation z(α) = (1 - α)z_A + αz_B",
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig("outputs/Figur_4_Latent_Interpolation.png", dpi=300, bbox_inches='tight')
    plt.close()

    # =================================================================
    # FIGUR 5: Training Loss Curves & DDPM Denoising Trajectory
    # =================================================================
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.38, wspace=0.28)

    # 6 subplot kurva loss
    plot_specs = [
        ("Vanilla AE (MSE Loss)", ['loss'], ['Vanilla AE']),
        ("Sparse AE (MSE + L1 Loss)", ['loss'], ['Sparse AE']),
        ("VAE (ELBO Loss)", ['loss'], ['VAE']),
        ("Vanilla GAN (Adversarial Loss)", ['g_loss', 'd_loss'], ['Generator', 'Discriminator']),
        ("DCGAN (Adversarial Loss)", ['g_loss', 'd_loss'], ['Generator', 'Discriminator']),
        ("DDPM (Noise Prediction MSE)", ['loss'], ['DDPM']),
    ]

    for idx, (title, keys, labels) in enumerate(plot_specs):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        hist_key = title.split(" (")[0]
        h = histories[hist_key]
        for k, lbl in zip(keys, labels):
            ax.plot(range(1, len(h[k]) + 1), h[k], marker='o', markersize=3, label=lbl)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel("Epoch", fontsize=8.5)
        ax.set_ylabel("Loss", fontsize=8.5)
        ax.grid(True, alpha=0.3)
        if len(keys) > 1:
            ax.legend(fontsize=8)

    # Panel bawah: DDPM Reverse Denoising Trajectory (x_T -> x_0)
    torch.manual_seed(42)
    _, traj = ddpm.sample(3, return_trajectory=True)
    step_indices = np.linspace(0, len(traj) - 1, 8, dtype=int)
    t_labels = np.linspace(ddpm.timesteps, 0, 8, dtype=int)

    sub_gs = gs[2, :].subgridspec(3, 8, wspace=0.06, hspace=0.06)
    ax_traj_title = fig.add_subplot(gs[2, :])
    ax_traj_title.set_title("Sekuens Reverse Denoising DDPM (x_T → x_0)", fontsize=11, fontweight='bold', pad=14)
    ax_traj_title.axis('off')

    for r in range(3):
        for c, s_idx in enumerate(step_indices):
            ax = fig.add_subplot(sub_gs[r, c])
            ax.imshow(traj[s_idx][r, 0], cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            if r == 0:
                ax.set_title(f"t = {t_labels[c]}", fontsize=8.5)

    plt.suptitle("Figur 5: Dinamika Kurva Training Loss Keenam Model dan Trajektori Reverse Diffusion DDPM",
                 fontsize=13, fontweight='bold', y=0.98)
    plt.savefig("outputs/Figur_5_Loss_and_DDPM_Trajectory.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Kelima figur berhasil disimpan di folder outputs/!")
