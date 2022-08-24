import os
import numpy
import pathlib
from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import random_split as random_split, DataLoader
from LoadData import load_data,plot_roc
import matplotlib.pyplot as plt
import argparse
from classifier import insertion_classifier
from sklearn.model_selection import KFold

def create_parser ():
    parser=argparse.ArgumentParser(description='Pretrain classifier on esm1b and esm-msa embeddings')

    parser.add_argument(
        "esm1b_dir",
        type=pathlib.Path,
        help="directory for extracted representations from ESM-1b",
    )

    parser.add_argument (
        'k_folds',
        type = int,
        help = 'Number of folds in cross val',
    )

    parser.add_argument(
        "out_dir",
        type=pathlib.Path,
        help="directory for trained model",
    )

    parser.add_argument('--epochs',default=20,type=int,help='Specifies number of training epochs. Default 20')

    return parser








def reset_model (model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def train (args):
    path = args.esm1b_dir
    k_folds = args.k_folds
    kfold  = KFold (n_splits = k_folds,shuffle= True)
    dataset = load_data(path)
    roc_folds  = {}
    auc_fold_progress= {}

    plt.title('ROC')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):

        print (f"FOLD {fold+1}")
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        train_loader = DataLoader(dataset, batch_size=32, sampler = train_subsampler)
        test_loader = DataLoader(dataset, batch_size=32,sampler = test_subsampler)
        print ('Datasets generated')
        print(len(train_subsampler),len(test_subsampler))
        model = insertion_classifier()
        print("model loaded")
        model.apply (reset_model)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        train_loss_progress = []
        test_auc_progress = []
        epochs = args.epochs
        model.to(device)
        print ('Begin Training')
        for epoch in range (epochs):
            model.train()

            train_loss = 0.0
            for (data, targets) in train_loader:
                data, targets = data.to(device), targets.to(device)
                targets=targets.float()
                data=data.float()
                targets = targets.unsqueeze(1)
                optimizer.zero_grad()
                output = model(data)

                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * data.size(0)




            train_loss = train_loss / len(train_subsampler)

            train_loss_progress.append(train_loss)
            print ('\n')
            print (f'Training progress for Fold {fold+1}')
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))

            test_loss =   0.0
            _preds = torch.tensor([]).to(device)
            _labels = torch.tensor([]).to(device)
            with torch.no_grad():
                for (data, targets) in test_loader:
                    data, targets = data.to(device), targets.to(device)
                    targets = targets.unsqueeze(1)
                    targets=targets.float()
                    data=data.float()
                    test_outputs = model(data)
                    loss = criterion(test_outputs, targets)
                    test_loss += loss.item() * data.size(0)
                    _preds = torch.cat((_preds, test_outputs[:, -1:]))
                    _labels = torch.cat((_labels, targets))

            ROC,fpr,tpr = plot_roc(_preds, _labels,epoch==epochs-1,fold+1)
            test_auc_progress.append(ROC)

            print (f'AUC: {round(ROC,2)}')
            if epoch == epochs-1:
                roc_folds [fold+1] = ROC
                plt.plot(fpr, tpr, label=f'Fold {fold+1} = %0.2f' % ROC)

            test_loss = test_loss / len(test_subsampler)
            print('Epoch: {} \tTesting Loss: {:.6f}'.format(epoch + 1, test_loss))
            print('\n')
        model.cpu()
        plt.legend(loc='lower right')
        plt.savefig('roc.png')
        auc_fold_progress[fold+1]=test_auc_progress
        save_path =f'{args.out_dir}-{fold+1}.pt'
        torch.save (model , save_path)

    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    sum = 0.0
    for key, value in roc_folds.items():
        print(f'Fold {key}: {value} ')
        sum += value
    print(f'Average: {sum / len(roc_folds.items())}')

    plt.clf()

    plt.title ("AUC vs. Epoch")
    plt.xlabel ('Epoch')
    plt.ylabel ('AUC')
    plt.ylim([.5, 1])
    plt.xticks ()
    for i in range (k_folds):
        plt.plot ([str(j) for j in range (1,epochs+1)],auc_fold_progress[i+1],label = f'Fold {i+1}')
    plt.legend()
    plt.savefig ('auc_progress.png')








if __name__ == '__main__':
    args = create_parser().parse_args()
    train(args)