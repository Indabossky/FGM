import torch
import torch.nn as nn

# 1. Define the MLP architecture with BatchNorm and Dropout
class FlameletMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super(FlameletMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),

            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),

            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# 2. Instantiate the model
input_dim = 4    # ['chai', 'C', 'Zvar', 'Zmean']
output_dim = 25  # number of species + production_rate
model = FlameletMLP(input_dim, output_dim)

# 3. Move to device (MPS if available, else CPU)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model = model.to(device)

# 4. Print model summary
print(model)
