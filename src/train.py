import time
import torch
import torch.nn as nn
import torch.nn.functional as F

def train_ae(model, train_loader, device, epochs=20, lr=1e-3, is_sparse=False, l1_lambda=1e-4):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    history = {'loss': [], 'epoch_time': []}

    for epoch in range(epochs):
        start_t = time.time()
        model.train()
        total_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            recon, h = model(x)
            loss = criterion(recon, x)
            if is_sparse:
                loss = loss + l1_lambda * torch.mean(torch.abs(h))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        history['loss'].append(total_loss / len(train_loader.dataset))
        history['epoch_time'].append(time.time() - start_t)
    return history


def train_vae(model, train_loader, device, epochs=20, lr=1e-3, beta=1.0):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {'loss': [], 'epoch_time': []}

    for epoch in range(epochs):
        start_t = time.time()
        model.train()
        total_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            recon_loss = F.binary_cross_entropy(recon, x, reduction='sum') / x.size(0)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
            loss = recon_loss + beta * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        history['loss'].append(total_loss / len(train_loader.dataset))
        history['epoch_time'].append(time.time() - start_t)
    return history


def train_gan(G, D, train_loader, device, epochs=25, lr=2e-4, latent_dim=100):
    G.to(device)
    D.to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()
    history = {'g_loss': [], 'd_loss': [], 'epoch_time': []}

    for epoch in range(epochs):
        start_t = time.time()
        G.train()
        D.train()
        g_sum, d_sum = 0.0, 0.0

        for x, _ in train_loader:
            b = x.size(0)
            real_x = x.to(device) * 2.0 - 1.0
            real_labels = torch.ones(b, 1, device=device)
            fake_labels = torch.zeros(b, 1, device=device)

            # Update Discriminator
            z = torch.randn(b, latent_dim, device=device)
            fake_x = G(z).detach()
            d_loss = criterion(D(real_x), real_labels) + criterion(D(fake_x), fake_labels)

            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()

            # Update Generator
            z = torch.randn(b, latent_dim, device=device)
            gen_x = G(z)
            g_loss = criterion(D(gen_x), real_labels)

            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

            g_sum += g_loss.item() * b
            d_sum += d_loss.item() * b

        n = len(train_loader.dataset)
        history['g_loss'].append(g_sum / n)
        history['d_loss'].append(d_sum / n)
        history['epoch_time'].append(time.time() - start_t)
    return history


def train_ddpm(ddpm, train_loader, device, epochs=20, lr=1e-3):
    ddpm.to(device)
    optimizer = torch.optim.Adam(ddpm.parameters(), lr=lr)
    history = {'loss': [], 'epoch_time': []}

    for epoch in range(epochs):
        start_t = time.time()
        ddpm.train()
        total_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device) * 2.0 - 1.0
            loss = ddpm.compute_loss(x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        history['loss'].append(total_loss / len(train_loader.dataset))
        history['epoch_time'].append(time.time() - start_t)
    return history
