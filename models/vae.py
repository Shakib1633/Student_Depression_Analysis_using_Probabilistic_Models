# models/vae.py  —  Variational Autoencoder Classifier

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import *

# ── Model definition ──────────────────────────────────────────────────────────

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=8, num_classes=2):
        super().__init__()
        self.encoder    = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU())
        self.fc_mu      = nn.Linear(32, latent_dim)
        self.fc_log_var = nn.Linear(32, latent_dim)
        self.decoder    = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, input_dim), nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        h       = self.encoder(x)
        mu      = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        z       = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var, self.classifier(z)


def vae_loss(recon, x, mu, log_var, logits, labels, beta=1.0, gamma=5.0):
    recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')
    kl_loss    = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    cls_loss   = nn.functional.cross_entropy(logits, labels, reduction='sum')
    return recon_loss + beta * kl_loss + gamma * cls_loss


def vae_loss_components(recon, x, mu, log_var, logits, labels, beta=1.0, gamma=5.0):
    """Returns individual loss components"""
    recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')
    kl_loss    = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    cls_loss   = nn.functional.cross_entropy(logits, labels, reduction='sum')
    total_loss = recon_loss + beta * kl_loss + gamma * cls_loss
    return total_loss, recon_loss, kl_loss, cls_loss


def run(df, train_df, test_df, label_encoders, X):
    print("\n" + "="*60)
    print("  VARIATIONAL AUTOENCODER")
    print("="*60)

    # ── Prepare tensors ───────────────────────────────────────────────────────
    X_np      = df.drop(columns=[TARGET]).values.astype(np.float32)
    X_np      = (X_np - X_np.min(axis=0)) / (X_np.max(axis=0) - X_np.min(axis=0) + 1e-8)

    # Use same indices as train/test split by rebuilding from the split dfs
    X_train_np = train_df.drop(columns=[TARGET]).values.astype(np.float32)
    X_train_np = (X_train_np - X_np.min(axis=0)) / (X_np.max(axis=0) - X_np.min(axis=0) + 1e-8)
    y_train_np = train_df[TARGET].values.astype(np.float32)

    X_test_np  = test_df.drop(columns=[TARGET]).values.astype(np.float32)
    X_test_np  = (X_test_np - X_np.min(axis=0)) / (X_np.max(axis=0) - X_np.min(axis=0) + 1e-8)
    y_test_np  = test_df[TARGET].values.astype(np.float32)

    X_train_t = torch.tensor(X_train_np)
    y_train_t = torch.tensor(y_train_np).long()
    X_test_t  = torch.tensor(X_test_np)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=64, shuffle=True)

    INPUT_DIM = X_np.shape[1]

    # ── Train ─────────────────────────────────────────────────────────────────
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model     = VAE(INPUT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    EPOCHS    = 50
    
    # Track all loss components for both train and test
    train_losses = []
    train_recon_losses = []
    train_kl_losses = []
    train_cls_losses = []
    test_losses = []
    test_recon_losses = []
    test_kl_losses = []
    test_cls_losses = []

    model.train()
    for epoch in range(1, EPOCHS + 1):
        # ── Training ──
        epoch_train_loss = 0.0
        epoch_train_recon = 0.0
        epoch_train_kl = 0.0
        epoch_train_cls = 0.0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            recon, mu, log_var, logits = model(xb)
            loss, recon_loss, kl_loss, cls_loss = vae_loss_components(recon, xb, mu, log_var, logits, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            
            epoch_train_loss += loss.item()
            epoch_train_recon += recon_loss.item()
            epoch_train_kl += kl_loss.item()
            epoch_train_cls += cls_loss.item()
        
        # Average over batches
        avg_train_loss = epoch_train_loss / len(X_train_np)
        avg_train_recon = epoch_train_recon / len(X_train_np)
        avg_train_kl = epoch_train_kl / len(X_train_np)
        avg_train_cls = epoch_train_cls / len(X_train_np)
        
        train_losses.append(avg_train_loss)
        train_recon_losses.append(avg_train_recon)
        train_kl_losses.append(avg_train_kl)
        train_cls_losses.append(avg_train_cls)
        
        # ── Test Evaluation ──
        model.eval()
        with torch.no_grad():
            recon_test, mu_test, log_var_test, logits_test = model(X_test_t.to(device))
            test_loss, test_recon, test_kl, test_cls = vae_loss_components(
                recon_test, X_test_t.to(device), mu_test, log_var_test, logits_test, 
                torch.tensor(y_test_np).long().to(device)
            )
            
            avg_test_loss = test_loss.item() / len(X_test_np)
            avg_test_recon = test_recon.item() / len(X_test_np)
            avg_test_kl = test_kl.item() / len(X_test_np)
            avg_test_cls = test_cls.item() / len(X_test_np)
            
            test_losses.append(avg_test_loss)
            test_recon_losses.append(avg_test_recon)
            test_kl_losses.append(avg_test_kl)
            test_cls_losses.append(avg_test_cls)
        
        model.train()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{EPOCHS}  |  Train Loss: {avg_train_loss:.4f}  |  Test Loss: {avg_test_loss:.4f}")

    # ── Plot All Loss Components ──────────────────────────────────────────────
    epochs_range = range(1, EPOCHS + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('VAE Training and Test Losses', fontsize=16, fontweight='bold')
    
    # Plot 1: Total Loss
    axes[0, 0].plot(epochs_range, train_losses, label='Train', linewidth=2, marker='o', markersize=3)
    axes[0, 0].plot(epochs_range, test_losses, label='Test', linewidth=2, marker='s', markersize=3)
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Reconstruction Loss
    axes[0, 1].plot(epochs_range, train_recon_losses, label='Train', linewidth=2, marker='o', markersize=3)
    axes[0, 1].plot(epochs_range, test_recon_losses, label='Test', linewidth=2, marker='s', markersize=3)
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Reconstruction Loss'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: KL Divergence
    axes[1, 0].plot(epochs_range, train_kl_losses, label='Train', linewidth=2, marker='o', markersize=3)
    axes[1, 0].plot(epochs_range, test_kl_losses, label='Test', linewidth=2, marker='s', markersize=3)
    axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('KL Loss')
    axes[1, 0].set_title('KL Divergence'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Classification Loss
    axes[1, 1].plot(epochs_range, train_cls_losses, label='Train', linewidth=2, marker='o', markersize=3)
    axes[1, 1].plot(epochs_range, test_cls_losses, label='Test', linewidth=2, marker='s', markersize=3)
    axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Classification Loss'); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout(); plt.show()
    
    # ── Combined Plot: All components on one graph ────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(epochs_range, train_recon_losses, label='Train Reconstruction', linewidth=2, marker='o', markersize=3)
    ax.plot(epochs_range, test_recon_losses, label='Test Reconstruction', linewidth=2, marker='s', markersize=3, linestyle='--')
    ax.plot(epochs_range, train_kl_losses, label='Train KL Divergence', linewidth=2, marker='^', markersize=3)
    ax.plot(epochs_range, test_kl_losses, label='Test KL Divergence', linewidth=2, marker='v', markersize=3, linestyle='--')
    ax.plot(epochs_range, train_losses, label='Train Total Loss', linewidth=2.5, marker='D', markersize=3, alpha=0.7)
    ax.plot(epochs_range, test_losses, label='Test Total Loss', linewidth=2.5, marker='*', markersize=6, linestyle='--', alpha=0.7)
    
    ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('VAE: All Loss Components (Train vs Test)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

    # ── Evaluate ──────────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        _, _, _, logits = model(X_test_t.to(device))
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        y_pred = np.argmax(probs, axis=1)
        y_prob = probs[:, 1]

    print(f"\nAccuracy : {accuracy_score(y_test_np, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test_np, y_pred, average='weighted'):.4f}")
    print(f"F1 Score : {f1_score(y_test_np, y_pred, average='weighted'):.4f}")
    print(f"ROC AUC  : {roc_auc_score(y_test_np, y_prob):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_np, y_pred))

    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test_np, y_pred), annot=True, fmt='d', cmap='Purples')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.title('Confusion Matrix: Depression (VAE)')
    plt.tight_layout(); plt.show()

    # ── Latent space visualisation ────────────────────────────────────────────
    with torch.no_grad():
        _, mu_all, _, _ = model(X_test_t.to(device))
        mu_all = mu_all.cpu().numpy()

    z_2d = PCA(n_components=2).fit_transform(mu_all)
    plt.figure(figsize=(7, 5))
    sc = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=y_test_np, cmap='coolwarm', alpha=0.6, s=15)
    plt.colorbar(sc, label='Depression (0=No, 1=Yes)')
    plt.xlabel('PC 1'); plt.ylabel('PC 2')
    plt.title('VAE Latent Space (PCA projection)')
    plt.tight_layout(); plt.show()

    # ── Single input prediction ───────────────────────────────────────────────
    def predict_depression(gender, age, academic_pressure, cgpa,
                           study_satisfaction, sleep_duration, dietary_habits,
                           suicidal_thoughts, work_study_hours, financial_stress,
                           family_history):
        age_bin   = pd.cut([age],   bins=[0,18,25,35,100],   labels=[0,1,2,3], include_lowest=True).astype(int)[0]
        cgpa_bin  = pd.cut([cgpa],  bins=[-0.01,5.0,6.5,8.0,10.01], labels=[0,1,2,3], include_lowest=True).astype(int)[0]
        hours_bin = pd.cut([work_study_hours], bins=[-0.01,4.5,8.5,10.5,12.01], labels=[0,1,2,3], include_lowest=True).astype(int)[0]

        def encode(col, val):
            return int(label_encoders[col].transform([val])[0])

        raw = np.array([[
            encode('Gender', gender), age_bin, academic_pressure, cgpa_bin,
            study_satisfaction, encode('Sleep Duration', sleep_duration),
            encode('Dietary Habits', dietary_habits),
            encode('Have you ever had suicidal thoughts ?', suicidal_thoughts),
            hours_bin, financial_stress,
            encode('Family History of Mental Illness', family_history)
        ]], dtype=np.float32)

        raw_norm = (raw - X_np.min(axis=0)) / (X_np.max(axis=0) - X_np.min(axis=0) + 1e-8)

        model.eval()
        with torch.no_grad():
            _, mu, _, _ = model(torch.tensor(raw_norm).to(device))
            logits_det  = model.classifier(mu)
            probs       = torch.softmax(logits_det, dim=1).cpu().numpy()[0]

        pred = int(np.argmax(probs))
        print(f"\n{'─'*40}")
        print(f"  Prediction       : {'Depressed' if pred == 1 else 'Not Depressed'}")
        print(f"  P(Not Depressed) : {probs[0]:.4f}")
        print(f"  P(Depressed)     : {probs[1]:.4f}")
        print(f"{'─'*40}")
        return pred, probs

    predict_depression(
        gender="Male", age=22, academic_pressure=5, cgpa=6.0,
        study_satisfaction=2, sleep_duration="Less than 5 hours",
        dietary_habits="Unhealthy", suicidal_thoughts="Yes",
        work_study_hours=9, financial_stress=4, family_history="No"
    )

    return model
