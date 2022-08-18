import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class insertion_classifier (nn.Module):
    def __init__(self):
        super(insertion_classifier, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1305,1000),
            nn.ReLU(),
            nn.Linear(1000,500),
            nn.ReLU(),
            nn.Linear (500,50),
            nn.ReLU(),
            nn.Linear (50,1)

        )
    def forward (self,x):
        output = self.linear_relu_stack(x)
        return output

if __name__ == '__main__':
    model = insertion_classifier()
    print (model)