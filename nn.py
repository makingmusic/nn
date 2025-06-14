import torch
import random
import math
from torch import nn
from dataset import create_dataset, verify_predictions

# -------------- Config ----------------
SEED          = 0
LR_BEGIN      = 1e-3  # Reduced from 1e-1
LR_END        = 1e-5  # Reduced from 1e-2
EPOCHS        = 10000  

torch.manual_seed(SEED) # for reproducibility

# -------------- Visualization Config ----------------
NUM_PLOTS     = 10 # number of plots/steps to show/record throughout the training loop

# -------------- Dataset -------------------------
ds, dl, x_train, y_train = create_dataset()

# -------------- Training setup -----------
patience = 500  # Early stopping patience. The model will continue training for up to 500 epochs after it stops improving

model = nn.Sequential(
    nn.Linear(1, 16, bias=True),    # First layer: 1 input -> 32 neurons
    nn.ReLU(),                      # Activation function
    nn.Linear(16, 1, bias=True)     # Second layer: 32 -> 1 output
)

# Choose one of these loss functions:
# 1. MSE (original)
# loss_fn = nn.MSELoss()

# 2. MAE/L1 Loss
# loss_fn = nn.L1Loss()

# 3. Huber Loss
# loss_fn = nn.HuberLoss()

# 4. Relative Error Loss
# def relative_error_loss(pred, target):
#     return torch.mean(torch.abs((pred - target) / target))
# loss_fn = relative_error_loss

# 5. Log-Cosh Loss
# def log_cosh_loss(pred, target):
#     return torch.mean(torch.log(torch.cosh(pred - target)))
# loss_fn = log_cosh_loss

# 6. Percentage Error Loss
def percentage_error_loss(pred, target):
    return torch.mean(torch.abs((pred - target) / target) * 100)
loss_fn = percentage_error_loss

optimiser = torch.optim.SGD(model.parameters(), lr=LR_BEGIN)

# Using ReduceLROnPlateau scheduler instead of exponential decay
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, 
    mode='min', 
    factor=0.5, 
    patience=100
)

# -------------- Training loop -----------
print(f"Data range - x: [{x_train.min().item():.1f}, {x_train.max().item():.1f}], y: [{y_train.min().item():.1f}, {y_train.max().item():.1f}]")

best_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()  # Set model to training mode
    epoch_loss = 0.0
    num_batches = 0
    
    for batch_idx, (xb, yb) in enumerate(dl):
        pred = model(xb)
        loss = loss_fn(pred, yb)
        
        # Debug: Check for NaN early
        if torch.isnan(loss) or torch.isnan(model[0].weight).any():
            print(f"NaN detected at epoch {epoch+1}, batch {batch_idx+1}")
            break
            
        loss.backward()
        
        # Debug: Check gradients
        grad = model[0].weight.grad
        if torch.isnan(grad).any():
            print(f"NaN gradient at epoch {epoch+1}, batch {batch_idx+1}")
            print(f"  Gradient: {grad.item()}")
            break
        
        # Reduced max_norm for gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
        optimiser.step()
        optimiser.zero_grad()
        
        epoch_loss += loss.item()
        num_batches += 1
    
    # Calculate average epoch loss
    avg_epoch_loss = epoch_loss / num_batches
    
    # Update learning rate based on loss
    scheduler.step(avg_epoch_loss)
    
    # Early stopping check
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        patience_counter = 0
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break
    
    # Print progress
    if (epoch+1) % (EPOCHS//NUM_PLOTS) == 0:
        current_lr = optimiser.param_groups[0]['lr']
        print(f"epoch {epoch+1:3d}  loss={avg_epoch_loss:.4f}  lr={current_lr:.2e}")
    
    # Break if NaN detected
    if torch.isnan(loss) or torch.isnan(model[0].weight).any():
        print("NaN detected, stopping training")
        break

print("training complete") 
print(f"Final loss: {avg_epoch_loss:.4f}")

# Evaluate the model
model.eval()  # Set model to evaluation mode
verify_predictions(model, x_train, y_train)