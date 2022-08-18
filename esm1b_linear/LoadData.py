import os
import torch
import sklearn.metrics as metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class load_data(Dataset):
    def __init__(self,embs_dir):
        self.embs_dir = embs_dir
        self.files = os.listdir( embs_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        esm1b_path = os.path.join (self.embs_dir , self.files[item])
        esm1b = torch.load (esm1b_path)
        emb = esm1b['emb']
        label = esm1b['label']

        return emb, label

def plot_roc(_preds, _labels,plot,fold):
    # plot ROC curve
    _preds = torch.reshape(_preds, (-1,))
    fpr, tpr, threshold = metrics.roc_curve(_labels.cpu().numpy(), _preds.cpu().numpy())
    roc_auc = metrics.auc(fpr, tpr)
   # print ('ROC_AUC {:.2f}'.format (roc_auc))

    return roc_auc,fpr,tpr