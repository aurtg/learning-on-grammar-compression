# Copyright (C) 2020 NEC Corporation
# See LICENSE.

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from argparse import ArgumentParser
import itertools
import json

import numpy as np

import torch
import torch.nn as nn

from torch_scatter import scatter_max

from common import train_eval, common_parser

class Model(nn.Module):
    def __init__(self, n_char, h_dim, n_cls, n_lstm_layers, dropout):
        super(Model, self).__init__()

        self.n_char = n_char
        self.h_dim = h_dim
        self.n_lstm_layers = n_lstm_layers

        self.embedding = nn.Embedding(n_char, h_dim, padding_idx=0)
        nn.init.normal_(self.embedding.weight, std=1/h_dim**0.5)

        self.lstm = nn.LSTM(
            input_size = h_dim,
            hidden_size = h_dim,
            num_layers = n_lstm_layers,
            bidirectional = True
        )

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.classifier = nn.Linear(h_dim*2, n_cls)

    def forward(self, seqs):
        device = next(self.parameters()).device

        n_seqs = len(seqs)

        len_seqs = [len(_) for _ in seqs]
        max_length = max(len_seqs)
        seq_order = np.argsort(len_seqs)[::-1]

        padded_seqs = [
            list(seqs[i_seq]) + [0] * (max_length - len(seqs[i_seq]))
            for i_seq in seq_order
        ]
        padded_seqs = torch.LongTensor(padded_seqs).to(device)
        h = self.embedding(padded_seqs)
        torch_seq_lengths = torch.LongTensor(sorted(len_seqs, reverse=True)).to(device)

        h = torch.transpose(h, 0, 1) # (seq_len x batch_size x emb_dim)
        h = torch.nn.utils.rnn.pack_padded_sequence(h, torch_seq_lengths, batch_first=False)

        h, _ = self.lstm(h)

        h, _ = torch.nn.utils.rnn.pad_packed_sequence(h, batch_first=False)
        h = h.view(max_length, len(seqs), 2, self.h_dim)
        h = torch.transpose(h, 0, 1) # (batch_size x seq_len x 2 x dim_lstm)
        h = torch.cat([h[:,:,0,:], h[:,:,1,:]], dim=2) # (batch_size x seq_len x dim_lstm*2)

        # Reordering sequences to original order, and apply max pooling.
        lst_now = []
        lst_to = []
        lst_pos = []
        for i_now, i_orig in enumerate(seq_order):
            len_seq = len_seqs[i_orig]

            lst_now += [i_now] * len_seq
            lst_to += [i_orig] * len_seq
            lst_pos += range(len_seq)
        lst_now = torch.LongTensor(lst_now).to(device)
        lst_to = torch.LongTensor(lst_to).to(device)
        lst_pos = torch.LongTensor(lst_pos).to(device)

        h = h[lst_now, lst_pos]

        min_h = torch.min(h).item()

        # print("="*40)
        # print(h)
        # print(lst_to)
        # scatter_max does not support minus values.
        out, _ = scatter_max(h-min_h, lst_to, dim=0, dim_size=len(seqs))
        out = out + min_h

        if self.dropout is not None:
            out = self.dropout(out)

        #
        out = self.classifier(out)

        return out

def construct_model(args):
    model = Model(
        n_char = args.n_char,
        h_dim = args.h_dim,
        n_cls = args.n_cls,
        n_lstm_layers = args.n_layers,
        dropout = args.dropout
    )

    return model

if __name__=="__main__":
    parser = common_parser()

    ## Model configurations
    parser.add_argument("--h-dim", type=int, default=200,
        help="Dimension size of vector representations. default: 200")
    parser.add_argument("--n-layers", type=int, default=1,
        help="Number of Bi-LSTM layers. default: 1")
    parser.add_argument("--dropout", type=float, default=0.0,
        help="Dropout probability. default: 0.0")

    args = parser.parse_args()

    train_eval(args, construct_model, "_naive_lstm_result.log")
