# Copyright (C) 2020 NEC Corporation
# See LICENSE.

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

import json

from tqdm import tqdm

from lzd import compress_sequence as compress_sequence_lzd
from repair import compress_sequence as compress_sequence_repair

DATASETS = [
    "H3",
    "H3K9ac",
    "H3K4me2",
    "H3K14ac",
    "H4",
    "H3K4me3",
    "H3K36me3",
    "H4ac",
    "H3K79me3",
    "H3K4me1"
]

def load_datafile(filename):
    labels = []
    seqs = []
    with open(filename) as h:
        for line in h:
            line = line.replace("\n","")
            if len(line) > 0:
                row = line.split("\t")
                labels.append(int(row[0]))
                seqs.append(row[1])

    return labels, seqs

def preprocess_dataset(data_name, alg="repair"):
    train_labels, train_seqs = \
        load_datafile(f"data/dna/{data_name}/train.tsv")
    test_labels, test_seqs = \
        load_datafile(f"data/dna/{data_name}/test.tsv")

    char_table = {"A": 1, "G": 2, "C": 3, "T": 4}

    def _preprocess_seq(seq):
        seq = [char_table[char] for char in seq]

        if alg=="repair":
            comp_seq = compress_sequence_repair(seq)
        elif alg=="lzd":
            comp_seq = compress_sequence_lzd(seq)

        return seq, comp_seq

    for split, seqs, labels in zip(["train", "test"], [train_seqs, test_seqs], [train_labels, test_labels]):
        data_X = []
        data_Xcomp = []

        for seq in tqdm(seqs, position=0, leave=True):
            char_seq, comp_seq = _preprocess_seq(seq)

            data_X.append(char_seq)
            data_Xcomp.append(comp_seq)

        with open(f"data/dna/{data_name}/{split}.label", "w") as h:
            json.dump(labels, h)
        with open(f"data/dna/{data_name}/{split}.uncomp", "w") as h:
            json.dump(data_X, h)
        with open(f"data/dna/{data_name}/{split}.{alg}", "w") as h:
            json.dump(data_Xcomp, h)

if __name__=="__main__":
    for dataset in DATASETS:
        preprocess_dataset(dataset)
    for dataset in DATASETS:
        preprocess_dataset(dataset, alg="lzd")
