import pandas as pd, numpy as np, torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from model import FlameletMLP 

# 1. Load dataset 
df = pd.read_csv('model_data.csv')
# Check if the file is loaded correctly
print(f"DataFrame shape: {df.shape}")
print(f"DataFrame columns: {df.columns.tolist()}")
print(f"First few rows:\n{df.head()}")
# Check for missing values
print(f"Missing values in DataFrame:\n{df.isnull().sum()}")
# 2. Split into features and targets
X = df[['chai', 'C', 'Zvar', 'Zmean']].values
y = df.drop(columns=['chai', 'C', 'Zvar', 'Zmean']).values

# 3. Train/validation/test split (70/15/15)
X_train_temp, X_test, y_train_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_temp, y_train_temp, test_size=0.1765, random_state=42)

# 4. Standardize features
X_scaler = StandardScaler().fit(X_train)
X_train_s, X_val_s, X_test_s = map(X_scaler.transform,
                                   (X_train, X_val, X_test))

y_scaler = StandardScaler().fit(y_train)
# eps = 1e-8                                     # clamp tiny std‑devs
# tiny = y_scaler.scale_ < eps
# if tiny.any():
#     print("Clamped these target columns:", np.where(tiny)[0])
#     y_scaler.scale_[tiny] = eps
y_train_s, y_val_s, y_test_s = map(y_scaler.transform,
                                   (y_train, y_val, y_test))

# 5. Convert to torch tensors
def to_torch(a): return torch.tensor(a, dtype=torch.float32)
X_train_t, X_val_t, X_test_t = map(to_torch, (X_train_s, X_val_s, X_test_s))
y_train_t, y_val_t, y_test_t = map(to_torch, (y_train_s, y_val_s, y_test_s))

# 6. Prepare DataLoaders
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                          batch_size=256, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t,   y_val_t),   batch_size=256)
test_loader  = DataLoader(TensorDataset(X_test_t,  y_test_t),  batch_size=256)


# 7. Select device: MPS if available, else CPU
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# 8. Define MLP model and move to device
model = FlameletMLP(input_dim=4, output_dim=25).to(device)

# 9. Optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2, verbose=True)

# 10. Training loop with loss tracking + early stopping
n_epochs, patience = 40, 7
best_val, stall = float('inf'), 0
train_losses, val_losses = [], []


for epoch in tqdm(range(1, n_epochs + 1), desc="Training"):
    # —— train —— #
    model.train(); running = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = criterion(model(xb), yb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        running += loss.item() * xb.size(0)
    train_losses.append(running / len(train_loader.dataset))

    # —— validate —— #
    model.eval(); running = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)      
            running += loss.item() * xb.size(0)
            
    val_loss = running / len(val_loader.dataset)
    val_losses.append(val_loss)
    # scheduler.step(val_loss)

    tqdm.write(f"Epoch {epoch:02d}  train {train_losses[-1]:.7f}  val {val_loss:.7f}")

    if val_loss < best_val:
        best_val, stall = val_loss, 0
        torch.save(model.state_dict(), "model_best.pth")
    else:
        stall += 1
        if stall >= patience:
            tqdm.write("Early stopping ↩")
            break

# 10b. Save the final model (optional—best is already saved above)
torch.save(model.state_dict(), "model_final.pth")

# 11. Load best model before final evaluation
model.load_state_dict(torch.load("model_best.pth"))
model.to(device).eval()

# 12. Evaluate on test set (batched)
with torch.no_grad():
    test_mse = sum(criterion(model(xb.to(device)), yb.to(device)).item()
                   * xb.size(0)    # weight by batch size
                   for xb, yb in test_loader) \
               / len(test_loader.dataset)
print(f"Test MSE: {test_mse:.4f}")


# 13. Inverse transform and save predictions
with torch.no_grad():
    preds_test = model(X_test_t.to(device)).cpu().numpy()
preds      = y_scaler.inverse_transform(preds_test)
y_test_inv = y_scaler.inverse_transform(y_test_s)

output_cols = df.columns.difference(['chai','C','Zvar','Zmean']).tolist()
pd.DataFrame(preds, columns=output_cols).to_csv("predictions.csv", index=False)
pd.DataFrame(y_test_inv, columns=output_cols).to_csv("y_test.csv", index=False)
# 14. Plot training & validation loss curve
plt.figure()
plt.plot(train_losses, label='train'); plt.plot(val_losses, label='val')
plt.legend(); plt.xlabel('epoch'); plt.ylabel('MSE')
plt.title('Training vs Validation MSE'); plt.tight_layout(); plt.savefig('loss_curve.png', dpi=300)

#15. Compute per‐species errors and save 
sel = ['co2','co','o2','o','oh','h','h2o','production_rate']
errs = []
for col in sel:
    if col not in output_cols: continue
    i = output_cols.index(col)
    act, prd = y_test_inv[:, i], preds[:, i]
    errs.append(
        dict(species=col,
             MAE=mean_absolute_error(act, prd),
             RMSE=np.sqrt(mean_squared_error(act, prd))))
pd.DataFrame(errs).to_csv("species_errors.csv", index=False)

# 16. Plot Predicted vs Actual for selected species
for col in sel:
    if col not in output_cols:
        continue
    i = output_cols.index(col)
    act, prd = y_test_inv[:, i], preds[:, i]        # ← full test arrays

    mae  = mean_absolute_error(act, prd)
    rmse = np.sqrt(mean_squared_error(act, prd))

    plt.figure(figsize=(4, 4))
    plt.scatter(act, prd, alpha=0.3, s=8)           # alpha ↓ for density
    lo, hi = act.min(), act.max()
    if lo != hi:
        plt.plot([lo, hi], [lo, hi], 'r--', label='Ideal')

    # error annotation
    plt.text(0.05, 0.95,
             f"MAE  = {mae:.3g}\nRMSE = {rmse:.3g}",
             transform=plt.gca().transAxes,
             ha='left', va='top',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.xlabel(f'Actual value')
    plt.ylabel(f'Predicted value')
    plt.title(f'Predicted vs Actual: {col}')
    plt.tight_layout()
    plt.savefig(f'pred_vs_actual_{col}.png', dpi=300)
    plt.close()