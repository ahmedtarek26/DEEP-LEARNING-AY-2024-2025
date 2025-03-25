import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from typing import Callable

"""
- device
- nn.Sequental
- seed of NN
"""

DEVICE = torch.device('mps' if torch.mps.is_available() else 'cpu')

class VanillaLinearClassifier(nn.Module):
    def __init__(self, in_features: int = 784, out_features: int = 10):
        super().__init__()
        torch.manual_seed(0)
        self.layer1 = nn.Linear(in_features, 300, bias=True)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(300, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), 784)
        h = self.relu(self.layer1(x))
        logits = self.layer2(h)

        return logits

class LinearClassifier(nn.Module):
    def __init__(self, 
                n_input: int = 784, 
                n_classes: int = 10):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_input, 300),
            nn.ReLU(),
            nn.Linear(300, n_classes),
        )   

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1) # (B, n_input)
        logits = self.net(x) # (B, n_classes)
        return logits
    

def my_criterion(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    The loss implementation.
    
    logits.shape: (B, n_classes)
    labels.shape: (B,)
    """
    pass
    


def train(max_epochs: int, 
        model: nn.Module,
        train_loader: DataLoader, 
        optimizer: torch.optim.Optimizer, 
        criterion: nn.Module) -> nn.Module:
    model.train()
    model.to(DEVICE)

    for epoch in range(max_epochs):
        total_loss = 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            print(f'Epoch {epoch}, Batch {i}, Loss {loss.item()}', end='\r')
        print(f'Epoch {epoch}, Batch {i}, Loss {total_loss/len(train_loader)}')

    return model

@torch.no_grad()
def test(model: nn.Module, test_loader: DataLoader) -> float:
    model.eval()
    model.to(DEVICE)

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        output = model(x)
        predicted = torch.argmax(output,1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    accuracy = correct / total * 100
    print(f'Overall Accuracy: {accuracy:.2f}%')

    print("\nClassification Report:\n", classification_report(all_labels, all_preds, digits=2))

    return accuracy



if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    # os.path.expanduser('~/datasets/')
    

    train_loader = DataLoader(
        datasets.MNIST('~/datasets/', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=128, shuffle=True
    )

    test_loader = DataLoader(
        datasets.MNIST('~/datasets/', train=False, download=True, transform=transforms.ToTensor()),
        batch_size=128, shuffle=False
    )

    print('MLP experiment')
    mlp_model = LinearClassifier(784, 10)
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.001)
    mlp_model = train(max_epochs=10, model=mlp_model, train_loader=train_loader, optimizer=optimizer, criterion=nn.CrossEntropyLoss())

    test(mlp_model, test_loader)