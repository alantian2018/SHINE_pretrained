import numpy
import pathlib
from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import random_split as random_split, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from sklearn.model_selection import KFold

def plot_roc(_preds, _labels,plot,fold):
    # plot ROC curve

    fpr, tpr, threshold = metrics.roc_curve(_labels, _preds)
    roc_auc = metrics.auc(fpr, tpr)
   # print ('ROC_AUC {:.2f}'.format (roc_auc))

    return roc_auc,fpr,tpr
def train ():
    plt.title(f'ROC')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')




    print ('plot other models')

    df = pd.read_csv('all.csv', sep='\t', header=0, keep_default_na=False)



    models=['PrimateAI_score' ,'REVEL_score',	'MPC_score',	'CADD_raw',	'MVP_score', 'M-CAP_score', 'VEST4_score',	'Polyphen2_HVAR_score'	,'UNEECON_score']

    for m in models:
        _labels=[]
        _preds= []
        for idx,r in df.iterrows():
            if (r[m]!=''):
                _preds.append (float (r[m]))
                _labels.append (float (r['target']))


        ROC, fpr, tpr = plot_roc(_preds, _labels, 0,0)


        print(f'{m}: {round(ROC, 2)}')

        plt.plot(fpr, tpr, label=f' {m} = %0.2f' % ROC)
    plt.legend(loc='lower right')
    plt.savefig('test_auc_all.png')

train()