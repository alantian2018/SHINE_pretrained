import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np


class insertion_classifier (nn.Module):
    def __init__(self):
        super(insertion_classifier, self).__init__()
        self.LSTM1 = nn.LSTM (1305,100,num_layers= 1,batch_first=True)
        self.LSTM2 = nn.LSTM (100,20,num_layers=1,batch_first = True)
        self.relu1 = nn.ReLU()

        self.flatten = nn.Flatten (1)
        self.model= nn.Sequential(

            nn.Linear (2580,1000),
            nn.ReLU(),
            nn.Linear (1000,500),
            nn.ReLU(),
            nn.Linear(500, 50),
            nn.ReLU(),
            nn.Linear(50, 1)

        )
    def forward (self,x):

        output,hn = self.LSTM1(x)
        output = self.relu1(output)
        output,hn =self.LSTM2(output)
        output = self.relu1(output)
        output = self.flatten (output)
     #   print (output.shape)
        output = self.model(output)
        return output

if __name__ == '__main__':
    model = insertion_classifier()
    print (model)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print (params)


