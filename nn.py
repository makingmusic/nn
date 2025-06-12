import torch
import random
import math
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# -------------- 1. Hyper-parameters ----------------
SEED          = 0
TRAIN_UPTO    = 1000        # integers 1…1000 for training
BATCH_SIZE    = 32
LR            = 1e-2  # Starting LR, will decay to 1e-6
EPOCHS        = 5000         # converges in < 1 s on CPU

torch.manual_seed(SEED)

# Visualization related params
NUM_PLOTS     = 10 # number of plots/steps to show for viz

# -------------- 2. Dataset -------------------------
# Generate column vectors [[1], [2], …] so they match nn.Linear's (N, 1) expectation
x_train = torch.arange(1, TRAIN_UPTO+1, dtype=torch.float32).unsqueeze(1)
y_train = 10 * x_train

ds  = TensorDataset(x_train, y_train)
dl  = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

# -------------- 3. Model & training loop -----------
model = nn.Sequential(nn.Linear(1, 1, bias=False))   # weight will learn ≈10
loss_fn   = nn.MSELoss()
optimiser = torch.optim.SGD(model.parameters(), lr=LR)

# LR scheduler: decay from 1e-2 to 1e-6 over EPOCHS
# Calculate gamma for exponential decay: final_lr = initial_lr * gamma^epochs
# 1e-6 = 1e-2 * gamma^50000 => gamma = (1e-6/1e-2)^(1/50000) = (1e-4)^(1/50000)
gamma = math.pow(1e-6 / LR, 1.0 / EPOCHS)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=gamma)

# Debug: Check initial state
print(f"Initial weight: {model[0].weight.item():.6f}")
print(f"Data range - x: [{x_train.min().item():.1f}, {x_train.max().item():.1f}], y: [{y_train.min().item():.1f}, {y_train.max().item():.1f}]")

for epoch in range(EPOCHS):
    for batch_idx, (xb, yb) in enumerate(dl):
        pred = model(xb)
        loss = loss_fn(pred, yb)
        
        # Debug: Check for NaN early
        if torch.isnan(loss) or torch.isnan(model[0].weight).any():
            print(f"NaN detected at epoch {epoch+1}, batch {batch_idx+1}")
            print(f"  Input batch: {xb[:3].flatten()}")
            print(f"  Target batch: {yb[:3].flatten()}")
            print(f"  Predictions: {pred[:3].flatten()}")
            print(f"  Loss: {loss.item()}")
            print(f"  Weight: {model[0].weight.item()}")
            break
            
        loss.backward()
        
        # Debug: Check gradients
        grad = model[0].weight.grad
        if torch.isnan(grad).any():
            print(f"NaN gradient at epoch {epoch+1}, batch {batch_idx+1}")
            print(f"  Gradient: {grad.item()}")
            break
        
        # Clip gradients to prevent explosion (nan problems)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
        optimiser.step()
        optimiser.zero_grad()
    
    # Break if NaN detected
    if torch.isnan(loss) or torch.isnan(model[0].weight).any():
        break
    
    # Step the scheduler after each epoch
    scheduler.step()
        
    # Optional tiny log
    if (epoch+1) % (EPOCHS//NUM_PLOTS) == 0:
        w = model[0].weight.item()
        current_lr = optimiser.param_groups[0]['lr']
        print(f"epoch {epoch+1:3d}  loss={loss.item():.6f}  weight≈{w:.10f}  lr={current_lr:.2e}")
# -------------- 4. Demo on unseen data -------------
model.eval()
test_nums = torch.tensor([[float(torch.randint(1, 10000, (1,))[0])] for _ in range(3)])
with torch.no_grad():
    print("\nInference:")
    for n, out in zip(test_nums.squeeze(1), model(test_nums).squeeze(1)):
        print(f"{int(n.item()):>5d}  ->  {out.item():.2f}  (ratio: {out.item()/n.item():.5f})")