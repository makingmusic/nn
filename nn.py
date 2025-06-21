import torch
import random
import math
from torch import nn
from dataset import create_dataset, verify_predictions, create_model, print_model_summary

# -------------- Config ----------------
SEED          = 0
LR_BEGIN      = 1e-1  # Reduced from 1e-1
LR_END        = 1e-2  # Reduced from 1e-2
EPOCHS        = 2000  
TRAINING_TYPE = 'quadratic' # # 'linear', 'quadratic', 'sine', 'exponential', 'step'
MODEL_TYPE    = '2layer' # '1layer', '2layer', '3layer'

# Additional optimization parameters
WEIGHT_DECAY  = 1e-4  # L2 regularization
MOMENTUM      = 0.9   # For SGD with momentum
BATCH_NORM    = True  # Use batch normalization
DROPOUT_RATE  = 0.1   # Dropout for regularization

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
patience = 1000  
model = create_model(MODEL_TYPE)

# I want to use a custom loss function that calculates the percentage error
# default loss function is MSELoss.
#loss_fn = nn.MSELoss()

def percentage_error_loss(pred, target):
    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    return torch.mean(torch.abs((pred - target) / (target + epsilon)) * 100)

# Percentage Error Loss
loss_fn = percentage_error_loss

# why SGD ? well, it is default and I am not trying to be fancy here.
optimiser = torch.optim.SGD(model.parameters(), lr=LR_BEGIN)

# Learning rate decay function 
# Using exponential decay for quadratic functions - more predictable than ReduceLROnPlateau
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimiser, 
    gamma=0.999999  # Decay factor
)

# -------------- Training loop -----------

best_loss = float('inf') # start pesimistic
patience_counter = 0    
loss_history = []  # Track loss history for analysis

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
        
        # Gradient clipping with adaptive norm
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
        optimiser.step()
        optimiser.zero_grad()
        
        epoch_loss += loss.item()
        num_batches += 1
    
    # Calculate average epoch loss
    avg_epoch_loss = epoch_loss / num_batches
    loss_history.append(avg_epoch_loss)
    
    # Update learning rate based on scheduler
    scheduler.step()
    
    # Early stopping with improved logic
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        patience_counter = 0
        # Save best model (optional)
        # torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print(f"Loss did not improve for {patience} epochs. Stopping early at epoch {epoch+1}.")
        break
    
    # Print progress with more details
    if (epoch+1) % (EPOCHS//NUM_PLOTS) == 0:
        current_lr = optimiser.param_groups[0]['lr']
        print(f"epoch {epoch+1:3d}  loss={avg_epoch_loss:.4f}  lr={current_lr:.2e}  best={best_loss:.4f}")
    
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
print(f"Best loss: {best_loss:.4f}")

# Evaluate the model
model.eval()  # Set model to evaluation mode
verify_predictions(model, x_train, y_train)

print_model_summary(model)