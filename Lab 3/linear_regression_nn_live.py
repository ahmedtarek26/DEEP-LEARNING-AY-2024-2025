from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
from typing import Tuple

# ------------------------------------------------------------------------------ 
"""
How to build your own custom dataset.

Mandatory methods to implement:
- __init__(self, data: torch.Tensor)
- __getitem__(self, index)

When your CustomDataset is used in a DataLoader, the DataLoader will call __len__ method to know the size of the dataset (thi is used to compute the number of batches!).
So if you want to use a DataLoader, you need to implement also 
- __len__(self)

In practice, your CustomDataset will be always used with a DataLoader, so you need to implement:
- __init__(self, data: torch.Tensor)
- __getitem__(self, index)
- __len__(self)
"""
class CustomDataset(Dataset):

    def __init__(self, data: torch.Tensor):
        self.data = data

    def __getitem__(self, index) -> torch.Tensor:
        return self.data[index]
    
    def __len__(self) -> int:
        return self.data.size(0)
    
if __name__ == '__main__':
    real_params = torch.tensor([1.0, 3.0])
    xaxis = torch.arange(-10, 10, 0.1) # shape: (200,)
    xaxis = xaxis.unsqueeze(1) # shape: (200, 1)
    yaxis = real_params[0] + real_params[1]*xaxis # shape: (200, 1) (broadcasting!!!)
    dataset = torch.cat([xaxis, yaxis], dim=-1) # shape: (200, 2)

    dataset = CustomDataset(dataset)
    print(len(dataset))
    print(dataset[0])
    print(dataset[0].shape)
    print(dataset[0][0], dataset[0][1])
    print(dataset[0][0].shape, dataset[0][1].shape)
    

# ------------------------------------------------------------------------------ 

class RegressionDataset(Dataset):

    def __init__(self, data: torch.Tensor):
        self.data = data
        self.x = None
        self.y = None
        
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        pass
   
    def __len__(self):
        pass
       
class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.first_layer = nn.Linear(in_features=None,
                                           out_features=None,
                                           bias=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
def loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    pass

def train(n_epochs: int, model: nn.Module, data_loader: DataLoader, opt: torch.optim.Optimizer):
    
    for _ in range(n_epochs):
        for x, y in data_loader:
            pass
        

"""
if __name__ == '__main__':
    # Define the dataset
    real_params = torch.tensor([1.0, 3.0])
    xaxis = torch.arange(-10, 10, 0.1) # shape: (200,)
    xaxis = xaxis.unsqueeze(1) # shape: (200, 1)
    yaxis = real_params[0] + real_params[1]*xaxis # shape: (200, 1) (broadcasting!!!)
    dataset = torch.cat([xaxis, yaxis], dim=-1) # shape: (200, 2)

    batch_size = 5
    
    # Create a DataLoader
    dataset = RegressionDataset(dataset)
    dataloader = None

    # Define the model and optimizer
    model = None
    optimizer = None

    # Train the model
    train(n_epochs=10, model=model, data_loader=dataloader, opt=optimizer)
"""   
