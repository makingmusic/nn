import torch
from torch.utils.data import TensorDataset, DataLoader


# Dataset related parameters
TRAIN_UPTO = 1000        # integers 1…1000 for training
BATCH_SIZE = 32

def create_dataset():
    # Generate column vectors [[1], [2], …] so they match nn.Linear's (N, 1) expectation
    x_train = torch.arange(1, TRAIN_UPTO+1, dtype=torch.float32).unsqueeze(1)
    y_train = (10 * x_train) + 1000

    ds = TensorDataset(x_train, y_train)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    
    return ds, dl, x_train, y_train


def verify_predictions(model, x_train, y_train):
    model.eval()
    with torch.no_grad():
        deviation_count = 0
        total_points = len(x_train)
        for x, y in zip(x_train, y_train):
            pred = model(x)
            deviation = abs(pred.item() - y.item()) / y.item()
            if deviation > 0.01:  # 1% threshold
                #print(f"x: {x.item()}, y: {y.item()}, pred: {pred.item()}, deviation: {deviation:.2%}")
                deviation_count += 1
        
        print(f"\nTotal deviations: {deviation_count} out of {total_points} points ({deviation_count/total_points:.2%})")
