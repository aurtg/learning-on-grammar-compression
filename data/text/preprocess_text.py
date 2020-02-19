# Copyright (C) 2020 NEC Corporation
# See LICENSE.

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

import csv

maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

from functools import partial
import json
import multiprocessing as mp

from tqdm import tqdm

import numpy as np

from repair import compress_sequence as compress_sequence_repair

DATASETS = [
    "sogou_news_csv",
    "yelp_review_full_csv",
    "yelp_review_polarity_csv"
]

N_CORE = mp.cpu_count()

def load_datafile(filename):
    labels = []
    texts = []
    with open(filename) as h:
        reader = csv.reader(h)
        for row in reader:
            labels.append(row[0])
            texts.append(row[-1])

    return labels, texts

def _preprocess_text(raw_text, max_length, char2ind, alg):
    if max_length is not None:
        # Limit text length to fixed length.
        text = raw_text[:max_length]
        if len(text) < max_length:
            text = text + " "*(max_length - len(text))
    else:
        text = raw_text

    # Indexing each characters.
    char_seq = [1 if char not in char2ind.keys() else char2ind[char] for char in text]

    # Compressing character sequence.
    if alg=="repair":
        comp_seq = compress_sequence_repair(char_seq)

    return char_seq, comp_seq

def average_length_of_dataset(data_name):
    train_labels, train_texts = \
        load_datafile(f"data/text/{data_name}/train.csv")

    text_len = [len(text) for text in train_texts]

    return np.mean(text_len), len(train_texts)

def preprocess_dataset(data_name, max_length=None, alg="repair"):
    train_labels, train_texts = \
        load_datafile(f"data/text/{data_name}/train.csv")
    test_labels, test_texts = \
        load_datafile(f"data/text/{data_name}/test.csv")

    # Indexing text categories.
    set_labels = set(train_labels) | set(test_labels)
    n_labels = len(set_labels)
    lab2ind = {lab: i_lab for i_lab, lab in enumerate(set_labels)}

    def _indexing(label):
        return lab2ind[label]
    train_labels = list(map(_indexing, train_labels))
    test_labels = list(map(_indexing, test_labels))

    # Load list of characters.
    with open("data/text/character_dictionary") as h:
        first_line = next(h)
        chars = " " + first_line[:69] + "\n"
    char2ind = {char: i_char + 1 for i_char, char in enumerate(chars)}
    # Blank space & unknown character will be indexed by 1.

    # Preprocessing texts.
    for split, texts, labels in zip(["train", "test"], [train_texts, test_texts], [train_labels, test_labels]):
        data_X = []
        data_Xcomp = []

        with mp.Pool(N_CORE) as pool:
            for i_data, (char_seq, comp_seq) in enumerate(
                    pool.imap(
                        partial(_preprocess_text, max_length=max_length, char2ind=char2ind, alg=alg),
                        texts, chunksize=100)
                ):
                if i_data % 100 == 0:
                    sys.stdout.write("{:.2f}% completed\r".format(i_data/len(texts)))
                    sys.stdout.flush()
                data_X.append(char_seq)
                data_Xcomp.append(comp_seq)

        with open(f"data/text/{data_name}/{split}.label", "w") as h:
            json.dump(labels, h)
        with open(f"data/text/{data_name}/{split}.uncomp", "w") as h:
            json.dump(data_X, h)
        with open(f"data/text/{data_name}/{split}.{alg}", "w") as h:
            json.dump(data_Xcomp, h)

if __name__=="__main__":
    for dataset in DATASETS:
        print(dataset)
        preprocess_dataset(dataset)
        preprocess_dataset(dataset, alg="lzd")
        #print(average_length_of_dataset(dataset))
