import torch
from torch.utils.data import TensorDataset, DataLoader
import math
import torch.nn as nn


# Dataset related parameters
TRAIN_UPTO = 1000        # integers 1…1000 for training
BATCH_SIZE = 32

def create_linear_dataset():
    x_train = torch.arange(1, TRAIN_UPTO+1, dtype=torch.float32).unsqueeze(1)
    y_train = (100 * x_train) + 1000
    return x_train, y_train

def create_quadratic_dataset():
    x_train = torch.arange(1, TRAIN_UPTO+1, dtype=torch.float32).unsqueeze(1)
    y_train = 10 * (x_train ** 2) + 100
    return x_train, y_train

def create_sine_dataset():
    x_train = torch.arange(1, TRAIN_UPTO+1, dtype=torch.float32).unsqueeze(1)
    y_train = 1000 * torch.sin(x_train / 50) + 2000
    return x_train, y_train

def create_exponential_dataset():
    x_train = torch.arange(1, TRAIN_UPTO+1, dtype=torch.float32).unsqueeze(1)
    y_train = 1000 * torch.exp(x_train / 200)
    return x_train, y_train

def create_step_dataset():
    x_train = torch.arange(1, TRAIN_UPTO+1, dtype=torch.float32).unsqueeze(1)
    y_train = 1000 * torch.floor(x_train / 100) + 1000
    return x_train, y_train

def create_dataset(dataset_type='1layer'):
    dataset_functions = {
        'linear': create_linear_dataset,
        'quadratic': create_quadratic_dataset,
        'sine': create_sine_dataset,
        'exponential': create_exponential_dataset,
        'step': create_step_dataset
    }
    
    if dataset_type not in dataset_functions:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Choose from {list(dataset_functions.keys())}")
    
    x_train, y_train = dataset_functions[dataset_type]() # super clever, a function pointer. i like it.
    
    ds = TensorDataset(x_train, y_train)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    
    return dl, x_train, y_train


def create_model(model_type='1layer'):
    class OneLayerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(1, 1)
            )
        
        def forward(self, x):
            return self.network(x)
    
    class TwoLayerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(1, 4),
                nn.Tanh(),  # Tanh activation works better for quadratic functions
                nn.Linear(4, 1)
            )
        
        def forward(self, x):
            return self.network(x)
    
    class ThreeLayerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(1, 8),
                nn.ReLU(),
                nn.Linear(8, 8),
                nn.ReLU(),
                nn.Linear(8, 1)
            )
        
        def forward(self, x):
            return self.network(x)
     
    model_classes = {
        '1layer': OneLayerModel,
        '2layer': TwoLayerModel,
        '3layer': ThreeLayerModel,
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(model_classes.keys())}")
    
    return model_classes[model_type]()

# This part should become a separate file perhaps. It is an entire eval script when it grows up.
# TODO: Write a better eval than the dumb "1% deviation" one that i have written today.
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
