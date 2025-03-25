import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from typing import Callable
# (1, 28, 28) --->  28*28 = 784

"""
def forward(input, par):
    combine input and par
    get output

forward(input)

"""

class LinearClassifier(nn.Module):

    def __init__(self, in_features: int = 784, out_features: int = 10):
        super().__init__()
        
        self.layer1 = nn.Linear(in_features=in_features, out_features=300, bias=True)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(in_features=300, out_features=out_features, bias=True)

    def forward(self, x) -> torch.Tensor:
        # x.shape = (B, 1, 28, 28)
        ### PYTORCH OPERATION: VIEW
        x = x.view(-1, 784) # B, 784
        h = self.relu(self.layer1(x))
        out = self.layer2(h)
        return out # B, out_features=10
    

def train(max_epochs: int, model: nn.Module, dataloader: DataLoader, loss: nn.Module | Callable, optimizer: torch.optim.Optimizer):
    model.train()
    for epoch in range(max_epochs):
        
        for x, y in dataloader:

            out = model(x)
            l = loss(out, y)

            optimizer.zero_grad()
            l.backward() # dl / d model.parametrs()
            optimizer.step()

        print(f'Epoch {epoch}, loss: {l.item()}')

    return model

torch.no_grad()
def test(model: nn.Module, test_loader: DataLoader) -> float:
    model.eval()
    

    correct = 0
    total = 0
    

    for x, y in test_loader:
        output = model(x)
        predicted = torch.argmax(output,1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

       
    accuracy = correct / total * 100
    print(f'Overall Accuracy: {accuracy:.2f}%')

    

    return accuracy
    

if __name__ == '__main__':

   

    from torchvision import transforms, datasets
    from torch.utils.data import DataLoader, Dataset
    train_dataset = datasets.MNIST("~/datasets/", train=True, download=True,
                        transform=transforms.ToTensor()
                        )
    
    test_loader = DataLoader(
        datasets.MNIST('~/datasets/', train=False, download=True, transform=transforms.ToTensor()),
        batch_size=128, shuffle=False
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    
    model = LinearClassifier()

    optimizer = torch.optim.Adam(model.parameters() , lr = 0.001)
    
    model = train(max_epochs=10, model=model, dataloader=train_loader, loss = nn.CrossEntropyLoss(), optimizer=optimizer)

    test(model=model, test_loader=test_loader)
    