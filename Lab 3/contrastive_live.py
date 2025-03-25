import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.transforms import functional as TF
from typing import Tuple

# https://lilianweng.github.io/posts/2021-05-31-contrastive/

DEVICE = 'mps' if torch.mps.is_available() else 'cpu'
#DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ------ Custom Dataset ------
class ContrastiveDataset(Dataset):
    def __init__(self, train=True):
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.augment = transforms.Compose([
            transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        self.dataset = datasets.MNIST(
            root='~/datasets/',
            train=train,
            download=True,
            transform=self.base_transform
        )

    def __getitem__(self, index):
        img, label = self.dataset[index]
       
        img_pil = TF.to_pil_image(img)
        ##TODO: Implement the contrastive learning data augmentation
        x1 = self.augment(img_pil)
        x2 = self.augment(img_pil)
        return (x1, x2), label  

    def __len__(self):
        return len(self.dataset)
    
class TripletDataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        pass
    
    def __len__(self):
        pass
    
# ------ Custom Loss Function ------
class CustomContrastiveLoss(nn.Module):
    
    # NT-Xent (InfoNCE) loss used in SimCLR.
    # https://lilianweng.github.io/posts/2021-05-31-contrastive/#simclr
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Useful torch functions:
        - torch.matmul
        - torch.cat
        - torch.arange

        Useful torch.nn.functional functions:
        - F.normalize
        - F.cross_entropy (internally applies softmax!!!!!!!!!!!!!!)
        """
        # Normalize the batch embeddings
        z = None
        
        # Compute the cosine similarity matrix
        sim = None
        # Scale the similarity matrix by the temperature
        sim = None
        
        
        """
        For the first N samples, the "positive" is i+B
        For the last N samples, the "positive" is i-B

        If B=8, targets looks like:
        targets = [4, 5, 6, 7, 0, 1, 2, 3]

        This means:
        - for the first data in the batch (z[0]), its positive is z[4] (the 5th data in the batch, ie the augmented version of the same datapoint)
        - for the second data in the batch (z[1]), its positive is z[5] (the 6th data in the batch)
        - for the third data in the batch (z[2]), its positive is z[6] (the 7th data in the batch)
        - for the fourth data in the batch (z[3]), its positive is z[7] (the 8th data in the batch)

        - for the fifth data in the batch (z[4]), its positive is z[0] (symmetric to the first case)
        - for the sixth data in the batch (z[5]), its positive is z[1] (symmetric to the second case)
        - ...

        """
        B = z.shape[0] // 2
        targets = None # shape: (2B,)

        
        # Mask out self-similarities so they don’t contribute to denominator (the i != k in the formula)
        sim_mask = torch.ones_like(sim, dtype=torch.bool)
        sim_mask.fill_diagonal_(False)
        # Setting the diagonal to -inf to exclude self-sim from softmax
        sim = sim.masked_fill(~sim_mask, float('-inf'))

        loss = None
        return 

class CustomTripletLoss(nn.Module):

    # https://lilianweng.github.io/posts/2021-05-31-contrastive/#triplet-loss

    def __init__(self):
       pass

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        pass

# ------ Feature Extractor ------
# Here we define two different feature extractors: one based on a ConvNet and another based on a simple Linear Encoder.

class ConvFeatureExtractor(nn.Module):
    def __init__(self, in_channels: int = 1, out_features: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64*7*7, out_features=out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64*7*7)
        x = F.relu(self.fc1(x))
        return x # (B, 128)


class LinearEncoder(nn.Module):
    def __init__(self, in_features: int = 784, out_features: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 784)
        return self.encoder(x)


# ------ Siamese Network ------
class SiameseNetwork(nn.Module):
    def __init__(self, feature_extractor: nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.feature_extractor(x)
        return z

    
    def train_triplet_loss(self, n_epochs: int, criterion: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer):
        pass 

    def train_contrastive_simclr(self, n_epochs: int, criterion: CustomContrastiveLoss, dataloader: ContrastiveDataset, optimizer: torch.optim.Optimizer):
        if not isinstance(dataloader.dataset, ContrastiveDataset):
            raise ValueError(f'Dataloader for this type of training must be instantiated using a ContrastiveDataset class.')
        if not isinstance(criterion, CustomContrastiveLoss):
            raise ValueError('The loss implementation must be CustomContrastiveLoss.')
        
        # daaloader = DataLoader(dataloader, batch_size=256, shuffle=True)
        ##TODO 
        self.train()
        for epoch in range(n_epochs):
            total_loss = 0
            for x, _ in dataloader:
                x = [xi.to(DEVICE) for xi in x] # remember that x = (x1, x2)
                # get the embeddings
                z1, z2 = None
                # concatenate the embeddings to get the augmented batch
                z = torch.cat([z1, z2], dim=0) # (2B, d)
                # the first positive pair is z[0], z[B]. Others are the negatives
                # the second positive pair is z[1], z[B+1]. Others are the negatives, etc.
                # the B-th positive pair is z[B-1], z[2B-1]. Others are the negatives

                # compute the loss
                loss = None
                total_loss += loss.item()
                print(f'Loss {loss.item()}', end='\r')
            print(f'Epoch {epoch}, Loss {loss/len(dataloader)}')


def test_visualization(model: nn.Module, test_loader: DataLoader):
    from matplotlib import pyplot as plt
    import numpy as np
    Z = []
    Y = []
    for batch, y in test_loader:
        x, *_ = batch
        z = model.feature_extractor(x.to(DEVICE))
        Z.append(z.detach().cpu().numpy())
        Y.append(y.detach().cpu().numpy())
    Z = np.concatenate(Z, axis=0)
    Y = np.concatenate(Y, axis=0)

    from umap import UMAP
    reducer = UMAP()
    Z = reducer.fit_transform(Z)

    plt.figure(figsize=(8, 6))
    plt.scatter(Z[:, 0], Z[:, 1], c=Y, cmap='tab10', s=5)
    if isinstance(test_loader.dataset, TripletDataset):
        loss_type = 'TripletLoss'
    elif isinstance(test_loader.dataset, ContrastiveDataset):
        loss_type = 'Contrastive SimCLR'
    plt.title(f'UMAP of the Embeddings - Model: {model.feature_extractor.__class__.__name__} - Loss: {loss_type}')
    plt.colorbar()
    plt.grid()
    plt.savefig(f'umap_{model.feature_extractor.__class__.__name__}_{loss_type}.png')



if __name__ == '__main__':
    feature_extractor = ConvFeatureExtractor() # LinearEncoder()
    n_epochs = 5
    dataset = ContrastiveDataset()
    criterion = CustomContrastiveLoss()

    model = SiameseNetwork(feature_extractor=feature_extractor)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train_contrastive_simclr(n_epochs=n_epochs, criterion=criterion, dataloader=DataLoader(dataset, batch_size=256, shuffle=True), optimizer=optimizer)

    test_loader = DataLoader(dataset, batch_size=128, shuffle=False)
    test_visualization(model=model, test_loader=test_loader)

    

