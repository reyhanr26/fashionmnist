#tahap 1
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = x.view(-1, 784)
        h1 = F.relu(self.fc1(x))
        z = F.relu(self.fc2(h1)) # Latent feature z_MLP \in R^64
        logits = self.fc3(z)
        return logits, z

class FrozenLatentDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )
    def forward(self, z):
        return self.net(z)

def run_mlp_experiment(train_loader, test_loader, device, epochs=15):
    # tahap 1 : mlp classifying
    mlp = MLPClassifier().to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        mlp.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = mlp(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
    mlp.eval()
    correct, total = 0, 0
    all_z = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits, z = mlp(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_z.append(z)
            
    acc = 100.0 * correct / total
    print(f"[Tahap 1] Akurasi Test Set MLP Classifier: {acc:.2f}% (Syarat min: 88%)")
    all_z = torch.cat(all_z, dim=0)
    
    sample_x, sample_y = next(iter(test_loader))
    sample_x, sample_y = sample_x[:8].to(device), sample_y[:8].to(device)
    
    W1 = mlp.fc1.weight.detach()
    W2 = mlp.fc2.weight.detach()
    W3 = mlp.fc3.weight.detach()
    
    with torch.no_grad():
        # tahap 2 : dari One-Hot Class y_c
        y_onehot = F.one_hot(sample_y, num_classes=10).float()
        z_hat = F.relu(y_onehot @ W3)
        h1_hat = F.relu(z_hat @ W2)
        x_from_class = torch.sigmoid(h1_hat @ W1).view(-1, 1, 28, 28)
        
        # tahap 3 : Latent Features z_MLP
        _, z_sample = mlp(sample_x)
        h1_from_z = F.relu(z_sample @ W2)
        x_from_z = torch.sigmoid(h1_from_z @ W1).view(-1, 1, 28, 28)
        
    # tahap 4 : frozen latent decoding
    for param in mlp.parameters():
        param.requires_grad = False
        
    decoder = FrozenLatentDecoder().to(device)
    dec_opt = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()
    
    for epoch in range(epochs):
        decoder.train()
        for x, _ in train_loader:
            x = x.to(device)
            with torch.no_grad():
                _, z = mlp(x)
            recon = decoder(z)
            loss = mse_loss(recon, x.view(-1, 784))
            dec_opt.zero_grad()
            loss.backward()
            dec_opt.step()
            
    decoder.eval()
    with torch.no_grad():
        x_frozen_recon = decoder(z_sample).view(-1, 1, 28, 28)
        
        # tahap 5 uji sampleing
        mu_z = all_z.mean(dim=0)
        std_z = all_z.std(dim=0)
        z_rand = torch.randn(8, 64, device=device) * std_z + mu_z
        x_prior_sample = decoder(z_rand).view(-1, 1, 28, 28)
        
    results = {
        'accuracy': acc,
        'orig': sample_x.cpu(),
        'w_trans_class': x_from_class.cpu(),
        'w_trans_z': x_from_z.cpu(),
        'frozen_recon': x_frozen_recon.cpu(),
        'prior_sample': x_prior_sample.cpu()
    }
    torch.save(results, "outputs/mlp_proof_results.pt")
    return mlp, decoder, results
