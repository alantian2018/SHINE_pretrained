import os
import argparse
import pathlib
import pandas as pd
from Bio import SeqIO

def create_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate representations for pathogenicity prediction"  # noqa
    )

    parser.add_argument(
        "input_csv",
        type=pathlib.Path,
        help="variant file on which to extract representations",
    )

    parser.add_argument (
        'esm_dirs',
        type = pathlib.Path,
        help = 'Path to ESM fasta files'
    )

    parser.add_argument(
        "result",
        type=pathlib.Path,
        help="result file on predictions",
    )


    return parser

def main (args):
    INPUT_PATH = args.input_csv
    ESM_dirs = args.esm_dirs
    OUTPUT_PATH= args.result
    print (ESM_dirs)
    df = pd.read_csv(INPUT_PATH, sep='\t', header=0, keep_default_na=False)
    l = set()
    for idx, line in df.iterrows():
        fn = os.path.join (ESM_dirs,line['transcript_id'])+'.fasta'
        if (not os.path.exists (fn)):
            print (f"{line['transcript_id']} doesn't exist")
            continue
        l.add (fn)
    out= []
    for fn in l:
        for record in SeqIO.parse(fn, "fasta"):
            if (len(record.seq)<=1022):
                out.append(record)
            else:
                seqs = len(record.seq)//1022+1
                for i in range (0,seqs):
                    sub_record = record[i*1022:min(i*1022+1022,len(record.seq))]
                    sub_record.id = sub_record.id +f"_{i}"
                    sub_record.name = sub_record.id
                    sub_record.description = sub_record.id
                    out.append (sub_record)
    print (f'Writing with {len(out)} sequences')
    with open (OUTPUT_PATH,'w') as handle:
        SeqIO.write (out,handle,'fasta')


if __name__=='__main__':
    args = create_parser().parse_args()
    main (args)

