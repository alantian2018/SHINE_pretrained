import numpy
import pathlib
import pandas as pd
from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import random_split as random_split, DataLoader
from LoadData import load_data, plot_roc
import matplotlib.pyplot as plt
import argparse
from classifier import insertion_classifier
from sklearn.model_selection import KFold
import os


def create_parser():
    parser = argparse.ArgumentParser(description='Pretrain classifier on esm1b and esm-msa embeddings')

    parser.add_argument(
        "esm1b_dir",
        type=pathlib.Path,
        help="directory for extracted representations from ESM-1b",
    )

    parser.add_argument(
        "out_dir",
        type=pathlib.Path,
        help="directory for model that should be tested",
    )
    parser.add_argument(
        'csv_file',
        type=pathlib.Path,
        help="Path to csv file"
    )
    return parser


def reset_model(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def train(args):
    PATH = args.out_dir
    path = args.esm1b_dir

    dataset = load_data(path)

    plt.title(f'ROC for CNN')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    test_loader = DataLoader(dataset, batch_size=32)
    print('Dataset generated')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = nn.BCEWithLogitsLoss()

    best_ROC, best_tpr, best_fpr = 0, 0, 0
    for idx, path in enumerate(os.listdir(PATH)):
        model = torch.load(os.path.join(PATH, path))
        model.to(device)
        print('Begin testing')

        test_loss = 0.0
        _preds = torch.tensor([]).to(device)
        _labels = torch.tensor([]).to(device)
        model.eval()

        with torch.no_grad():
            for (data, targets) in test_loader:
                data, targets = data.to(device), targets.to(device)
                targets = targets.unsqueeze(1)
                targets = targets.float()
                data = data.float()
                test_outputs = model(data)
                loss = criterion(test_outputs, targets)
                test_loss += loss.item() * data.size(0)
                _preds = torch.cat((_preds, test_outputs[:, -1:]))
                _labels = torch.cat((_labels, targets))

            ROC, fpr, tpr = plot_roc(_preds, _labels, 0, 0)

            print(f'AUC: {round(ROC, 2)}')
            if (best_ROC < ROC):
                best_ROC = ROC
                best_fpr = fpr
                best_tpr = tpr

    plt.plot(best_fpr, best_tpr, label=f'RNN = %0.2f' % best_ROC)

    print('plot other models')

    df = pd.read_csv(args.csv_file, sep='\t', header=0, keep_default_na=False)

    models = ['PrimateAI_score', 'REVEL_score', 'MPC_score', 'CADD_raw', 'MVP_score', 'M-CAP_score', 'VEST4_score',
              'Polyphen2_HVAR_score', 'UNEECON_score']

    for m in models:
        _labels = []
        _preds = []
        for idx, r in df.iterrows():
            if (r[m] != ''):
                _preds.append(float(r[m]))
                _labels.append(float(r['target']))

        ROC, fpr, tpr = plot_roc(torch.FloatTensor(_preds), torch.FloatTensor(_labels), 0, 0)

        print(f'{m}: {round(ROC, 2)}')

        plt.plot(fpr, tpr, label=f' {m} = %0.2f' % ROC)
    plt.legend(loc='lower right')
    plt.savefig('test_auc.png')


if __name__ == '__main__':
    args = create_parser().parse_args()
    train(args)