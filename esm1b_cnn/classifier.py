import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

class insertion_classifier (nn.Module):
    def __init__(self):
        super(insertion_classifier, self).__init__()
        self.flatten = nn.Flatten()
        self.model= nn.Sequential(
            nn.Conv1d (129,50,3),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d (50,20,2),
            nn.ReLU(),
            nn.MaxPool1d(3),
            nn.Flatten (1),
            nn.Linear (2880,1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 50),
            nn.ReLU(),
            nn.Linear(50, 1)

        )
    def forward (self,x):
        output = self.model(x)
        return output

if __name__ == '__main__':
    model = insertion_classifier()
    print (model)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print (params)