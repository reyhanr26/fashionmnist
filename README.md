# Studi Komparatif Empiris Model Generatif (AE, Sparse AE, VAE, GAN, DCGAN, DDPM)

Repositori ini berisi implementasi PyTorch untuk tugas Take-Home **Deep Learning (D6421) Sesi 07** pada dataset **Fashion-MNIST** (split 80% Train, 10% Validation, 10% Test, `seed=42`).

## Struktur Direktori
- `src/dataset.py` : Pengunduhan otomatis Fashion-MNIST dan pembagian dataset 80/10/10.
- `src/mlp_proof.py` : Eksperimen pembuktian batasan bobot (W^T) dan ruang laten (Z_MLP) pada MLP Classifier.
- `src/models.py` : Implementasi arsitektur Vanilla AE, Sparse AE, VAE, Vanilla GAN, DCGAN, dan DDPM.
- `src/train.py` : Pipeline pelatihan dan pencatatan waktu per epoch serta kurva loss.
- `src/evaluate.py` : Pengukuran metrik Test MSE, PSNR, SSIM, FID Score, dan Inference Latency (batch 64).
- `src/visualize.py` : Pembuatan 5 visualisasi komparatif utama (300 DPI) di folder `outputs/`.
- `main.py` : Skrip utama untuk menjalankan seluruh alur eksperimen secara end-to-end.

## Cara Menjalankan
    pip install -r requirements.txt
    python main.py
