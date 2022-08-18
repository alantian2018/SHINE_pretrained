
import argparse
import pathlib
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from os import path

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDRegressor, LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
import os
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score


def create_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate representations for pathogenicity prediction"  # noqa
    )

    parser.add_argument(
        "input_csv",
        type=pathlib.Path,
        help="variant file on which to extract representations",
    )


    parser.add_argument(
        "esm1b_dir",
        type=pathlib.Path,
        help="directory for extracted representations from ESM-1b",
    )
    parser.add_argument(
        'out_root',
        type = pathlib.Path,
        help ='directory for result embeddings'

    )


    return parser


parser = create_parser()
args = parser.parse_args()

# load AAIndex, O and U are not available
# aaindex = pd.read_csv("/share/terra/Users/xf2193/resource/AAIndex/AAIndex_imputed.txt",sep='\t',header=0)
# aaindex = aaindex.to_dict('list')
AA = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
      'X', 'Y']
AA = np.array(AA).reshape(-1, 1)
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(AA)

INPUT_PATH = args.input_csv
esm1b_PATH = args.esm1b_dir  # Path to directory of embeddings for fasta
out_root = args.out_root
EMB_LAYER = 33





def dataset(INPUT_PATH, esm1b_PATH,out_root):
    # Load embeddings (Xs) and target effects (ys)
    df = pd.read_csv(INPUT_PATH, sep='\t', header=0, keep_default_na=False)

    invalid_entries = 0
    index_error = 0
    # the initial token is not saved, so 0-based
    for idx, line in df.iterrows():
        if (idx%2000==0 ):
            print (f"{int (100*idx/len (df.index))}% done")
        ref = line['ref_aa']
        alt = line['alt_aa']
        pos_var = line['aa_pos'] - 1  # 1-based to 0-based
        label  = line ['target']
        # representation from ESM-1b
        transcript = line['transcript_id']
        fn =[filename for filename in os.listdir(esm1b_PATH) if filename.startswith(transcript)]
        if (len(fn)==0) :
            print(transcript)
            invalid_entries +=1
            continue
        esm1b = torch.tensor ([])
        for entry in fn:
            path_to_emb = os.path.join (esm1b_PATH,entry)
            embs = torch.load(path_to_emb)
            esm1b = torch.cat ((esm1b,embs['representations'][EMB_LAYER]))  # 1280 features
        if esm1b.shape[0] <= pos_var:
            print(f'{transcript} embedding does not exist at {pos_var}')
            index_error+=1
            continue
        esm1b = esm1b[pos_var,]

        # AAIndex
        if alt in AA:
            #            b = aaindex[alt]
            ref = np.array(ref).reshape(-1, 1)
            a = enc.transform(ref).toarray()
            alt = np.array(alt).reshape(-1, 1)
            b = enc.transform(alt).toarray()
            b = torch.tensor((a - b).flatten())
        else:
            print(line)
            b = torch.zeros(25)  # 25 features
        # combine features

        embs = {"emb":torch.cat((esm1b, b)), 'label':label}
        if (idx%2000==0 ):
            print (embs,embs['emb'].shape)
        transcript = f'{idx+1}_{transcript}'
        OUT_PATH = path.join (out_root, transcript)+".pt"
        torch.save (embs , OUT_PATH)


    print (f'invalid entries {invalid_entries}')
    print (f'out of bounds {index_error}')



dataset(INPUT_PATH, esm1b_PATH,out_root)
