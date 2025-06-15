import torch
import random
import math
from torch import nn
from dataset import create_dataset, verify_predictions, create_model

# -------------- Config ----------------
SEED          = 0
LR_BEGIN      = 1e-2  # Reduced from 1e-1
LR_END        = 1e-5  # Reduced from 1e-2
EPOCHS        = 50000  
TRAINING_TYPE = 'quadratic' # # 'linear', 'quadratic', 'sine', 'exponential', 'step'
MODEL_TYPE    = '3layer' # '1layer', '2layer', '3layer'


torch.manual_seed(SEED) # for reproducibility

# -------------- Visualization Config ----------------
NUM_PLOTS     = 10 # number of plots/steps to show/record throughout the training loop

# -------------- Dataset -------------------------
print("creating dataset ..", end="", flush=True)
dl, x_train, y_train = create_dataset(TRAINING_TYPE) 
print(" done")

# -------------- Training setup -----------
# Early stopping patience. 
# The model will continue training for up to 500 epochs after it stops improving
patience = 500  
model = create_model(MODEL_TYPE)

def percentage_error_loss(pred, target):
    return torch.mean(torch.abs((pred - target) / target) * 100)
# Percentage Error Loss
loss_fn = percentage_error_loss

# Choose one of these loss functions:
#loss_fn = nn.MSELoss()

optimiser = torch.optim.SGD(model.parameters(), lr=LR_BEGIN)

# Using ReduceLROnPlateau scheduler instead of exponential decay
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, 
    mode='min', 
    factor=0.5, 
    patience=100
)

# -------------- Training loop -----------

best_loss = float('inf')
patience_counter = 0

print ("starting model training")
for epoch in range(EPOCHS):
    model.train()  # Set model to training mode
    epoch_loss = 0.0
    num_batches = 0
    
    for batch_idx, (xb, yb) in enumerate(dl):
        pred = model(xb)
        loss = loss_fn(pred, yb)
        
        # Debug: Check for NaN early
        if torch.isnan(loss):
            print(f"NaN loss detected at epoch {epoch+1}, batch {batch_idx+1}")
            break
            
        # Check for NaN in model parameters
        has_nan = False
        for param in model.parameters():
            if torch.isnan(param).any():
                has_nan = True
                break
        if has_nan:
            print(f"NaN weights detected at epoch {epoch+1}, batch {batch_idx+1}")
            break
            
        loss.backward()
        
        # Debug: Check gradients
        has_nan_grad = False
        for param in model.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                has_nan_grad = True
                break
        if has_nan_grad:
            print(f"NaN gradient detected at epoch {epoch+1}, batch {batch_idx+1}")
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
        print(f"Loss did not improve for {patience} epochs. Stopping early at epoch {epoch+1}.")
        break
    
    # Print progress
    if (epoch+1) % (EPOCHS//NUM_PLOTS) == 0:
        current_lr = optimiser.param_groups[0]['lr']
        print(f"epoch {epoch+1:3d}  loss={avg_epoch_loss:.4f}  lr={current_lr:.2e}")
    
    # Break if NaN detected
    if torch.isnan(loss):
        print("NaN loss detected, stopping training")
        break
        
    has_nan = False
    for param in model.parameters():
        if torch.isnan(param).any():
            has_nan = True
            break
    if has_nan:
        print("NaN weights detected, stopping training")
        break

print("training complete") 
print(f"Final loss: {avg_epoch_loss:.4f}")

# Evaluate the model
model.eval()  # Set model to evaluation mode
verify_predictions(model, x_train, y_train)