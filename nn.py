import torch
import random
import math
from torch import nn
from dataset import create_dataset, verify_predictions

# -------------- Config ----------------
SEED          = 0
LR_BEGIN      = 1e-2  # Starting LR, will decay to LR_END
LR_END        = 1e-3
EPOCHS        = 400  

torch.manual_seed(SEED) # for reproducibility

# -------------- Visualization Config ----------------
NUM_PLOTS     = 10 # number of plots/steps to show/record throughout the training loop

# -------------- Dataset -------------------------
ds, dl, x_train, y_train = create_dataset()

# -------------- Training setup -----------
model = nn.Sequential(nn.Linear(1, 1, bias=False))   # weight will learn ≈10
loss_fn   = nn.MSELoss()
optimiser = torch.optim.SGD(model.parameters(), lr=LR_BEGIN)

# LR scheduler: decay from LR_BEGIN to LR_END over EPOCHS
# Calculate gamma for exponential decay: final_lr = initial_lr * gamma^epochs
# LR_END = LR_BEGIN * gamma^EPOCHS => gamma = (LR_END/LR_BEGIN)^(1/EPOCHS)
gamma = math.pow(LR_END / LR_BEGIN, 1.0 / EPOCHS)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=gamma)

# -------------- 3. Training loop -----------

# Debug: Check initial state
print(f"Init NN weight: {model[0].weight.item():.6f}")
print(f"Data range - x: [{x_train.min().item():.1f}, {x_train.max().item():.1f}], y: [{y_train.min().item():.1f}, {y_train.max().item():.1f}]")

for epoch in range(EPOCHS):
    if (epoch+1) % (EPOCHS//NUM_PLOTS) == 0:
        w = model[0].weight.item()
        current_lr = optimiser.param_groups[0]['lr']
        print(f"epoch {epoch+1:3d}  loss={loss.item():.6f}  weight≈{w:.10f}  lr={current_lr:.2e}")

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
        # end batch for loop
    
    # Break if NaN detected
    if torch.isnan(loss) or torch.isnan(model[0].weight).any():
        print("NaN detected, stopping training")
        break
    
    # Step the scheduler after each epoch
    scheduler.step()
# end epoch for loop
print ("training complete") 
print(f"epoch {epoch+1:3d}  loss={loss.item():.6f}  weight≈{w:.10f}")

# Evaluate the model
verify_predictions(model, x_train, y_train)