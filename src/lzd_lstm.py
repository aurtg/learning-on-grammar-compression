# Copyright (C) 2020 NEC Corporation
# See LICENSE.

import sys

from argparse import ArgumentParser
import itertools
import json
import pickle as pic

import numpy as np

import torch
import torch.nn as nn

from torch_scatter import scatter_max

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold
import sklearn.metrics

from model import *
from common import train_eval, common_parser

class LZD_Encoder(Base_LZD_Encoder):
    def __init__(self, n_char, h_dim, model_type, model_args, group_mode):
        super(LZD_Encoder, self).__init__(n_char, h_dim, group_mode)

        self.model_type = model_type
        self.model_args = model_args

        self.build_model()

    def build_composer_layer(self):
        if self.model_type == "fully-connected":
            return Fully_Connected_Composer(self.h_dim,
                self.model_args["hid_dims"], self.h_dim)
        elif self.model_type == "dual-gru":
            return Dual_GRU_Composer(self.h_dim)

class Repair_Encoder(Base_RePair_Encoder):
    def __init__(self, n_char, h_dim, model_type, model_args, group_mode):
        super(Repair_Encoder, self).__init__(n_char, h_dim, group_mode, None)

        self.model_type = model_type
        self.model_args = model_args

        self.build_model()

    def build_composer_layer(self):
        if self.model_type == "fully-connected":
            return Fully_Connected_Composer(self.h_dim,
                self.model_args["hid_dims"], self.h_dim)
        elif self.model_type == "dual-gru":
            return Dual_GRU_Composer(self.h_dim)

class Model(nn.Module):
    def __init__(self, lzd_encoder, h_dim, n_cls, n_lstm_layers, dropout):
        super(Model, self).__init__()

        self.lzd_encoder = lzd_encoder

        self.h_dim = h_dim
        self.n_lstm_layers = n_lstm_layers

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

        self.clf = nn.Linear(h_dim*2, n_cls)

    def forward(self, lst_lzd_sequences):
        device = next(self.parameters()).device

        n_seq = len(lst_lzd_sequences)

        ## Encode symbols of compressed sequences.
        h_frags, non_char_segments = self.lzd_encoder(lst_lzd_sequences)

        len_seqs = [len(_) for _ in non_char_segments]
        torch_seq_lengths = torch.LongTensor(sorted(len_seqs, reverse=True)).to(device)
        max_length = max(len_seqs)
        seq_order = np.flip(np.argsort(len_seqs), axis=0).copy()
        torch_seq_order = torch.LongTensor(seq_order)

        # Extract representations of symbols & sort sequences by their length
        padded_non_char_seg = [_ + [0]*(max_length-len(_)) for _ in non_char_segments]
        padded_non_char_seg = torch.LongTensor(padded_non_char_seg).to(device)
        h_frags = h_frags[torch_seq_order.unsqueeze(1), padded_non_char_seg[torch_seq_order]]

        ## Apply Bi-LSTM
        h = torch.transpose(h_frags, 0, 1)
        h = torch.nn.utils.rnn.pack_padded_sequence(h, torch_seq_lengths,
            batch_first=False)

        h, _ = self.lstm(h)

        h, _ = torch.nn.utils.rnn.pad_packed_sequence(h, batch_first=False)
        h = h.view(max_length, n_seq, 2, self.h_dim)
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
        out, _ = scatter_max(h-min_h, lst_to, dim=0, dim_size=n_seq)
        out = out + min_h

        # Apply classifier.
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.clf(out)

        return out

def construct_model(args):
    if args.composer == "fully-connected":
        model_args = {"hid_dims": [args.h_dim]}
    elif args.composer == "dual-gru":
        model_args = None
    else:
        raise Exception(args.composer)

    if args.alg == "lzd":
        encoder = LZD_Encoder(
            n_char = args.n_char,
            h_dim = args.h_dim,
            model_type = args.composer,
            model_args = model_args,
            group_mode = args.group_mode
        )
    elif args.alg == "repair":
        encoder = Repair_Encoder(
            n_char = args.n_char,
            h_dim = args.h_dim,
            model_type = args.composer,
            model_args = model_args,
            group_mode = args.group_mode
        )
    model = Model(
        lzd_encoder = encoder,
        h_dim = args.h_dim,
        n_cls = args.n_cls,
        n_lstm_layers = args.n_lstm_layers,
        dropout = args.dropout
    )

    return model

if __name__=="__main__":
    parser = common_parser()

    ## Model configurations
    parser.add_argument("--composer",
        choices=["fully-connected", "dual-gru"],
        default="fully-connected",
        help="Type of a composer module.")

    parser.add_argument("--h-dim", type=int, default=200,
        help="Dimension size of vector representations. default: 200")
    parser.add_argument("--n-lstm-layers", type=int, default=1,
        help="Number of Bi-LSTM layers. default: 1")
    parser.add_argument("--dropout", type=float, default=0.0,
        help="Dropout probability. default: 0.0")

    parser.add_argument("--group-mode", type=int, default=0)


    args = parser.parse_args()

    train_eval(args, construct_model, "_lzd_lstm_result.log")
