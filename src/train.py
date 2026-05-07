import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR


class NKDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def train_model(data: dict,
                device: str = 'cuda',
                d_model: int = 64,
                n_heads: int = 4,
                n_layers: int = 4,
                ff_dim: int = 256,
                dropout: float = 0.1,
                batch_size: int = 128,
                lr: float = 1e-3,
                weight_decay: float = 1e-4,
                epochs: int = 100,
                patience: int = 10,
                checkpoint_dir: str = './results/checkpoints',
                silent: bool = False):
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    if not silent:
        print(f"\t\t- Device: {device}")

    train_ds = NKDataset(data['X_train'], data['Y_train'])
    val_ds = NKDataset(data['X_val'], data['Y_val'])

    n_workers = 0 if len(train_ds) < 1000 else 4
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=n_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=n_workers, pin_memory=True)

    from src.model import NKTransformer
    input_dim = data['X_train'].shape[-1]
    output_dim = data['Y_train'].shape[-1]
    model = NKTransformer(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        ff_dim=ff_dim, dropout=dropout,
        input_dim=input_dim, output_dim=output_dim,
    ).to(device)

    if hasattr(torch, 'compile') and device.type == 'cuda':
        try:
            model = torch.compile(model, mode='reduce-overhead', dynamic=True)
            if not silent:
                print("\t\t- Optimization: torch.compile (dynamic-shape mode)")
        except Exception:
            pass

    total_params = sum(p.numel() for p in model.parameters())
    if not silent:
        print(f"\t\t- Parameters: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        model.train()
        train_loss = 0.0
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast('cuda'):
                    pred = model(X_batch)
                    loss = criterion(pred, Y_batch)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(X_batch)
                loss = criterion(pred, Y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        train_loss /= len(train_ds)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, Y_batch)
                val_loss += loss.item() * X_batch.size(0)

        val_loss /= len(val_ds)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if not silent:
            print(f"\t\t  [Epoch {epoch:3d}] Loss: {train_loss:.6f} | Val: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pt'))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if not silent:
                    print(f"\t\t  [!] Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best_model.pt'),
                                     map_location='cpu', weights_only=True))
    model.eval()
    if not silent:
        print(f"\t\t- Convergence: Epoch {best_epoch} (Val MSE: {best_val_loss:.6f})")

    return model, history
