# Copyright (C) 2020 NEC Corporation
# See LICENSE.

import torch
import torch.nn as nn

import grouping

#####################################
# Encoder for LZD compressed sequence
#####################################

class Base_LZD_Encoder(nn.Module):
    """Encode segments of a LZD compressed sequence."""
    def __init__(self, n_char, h_dim, grouping_mode=0, cell_dim=None):
        super(Base_LZD_Encoder, self).__init__()

        self.n_char = n_char
        self.h_dim = h_dim
        self.cell_dim = cell_dim

        self.grouping_mode = grouping_mode

    def build_model(self):
        ## Build model.
        self.emb_char = nn.Embedding(self.n_char, self.h_dim)
        nn.init.normal_(self.emb_char.weight, std=1/self.h_dim**0.5)

        self.composer_layer = self.build_composer_layer()
            # (inp1_vec, inp2_vec) -> out_vec

    def build_composer_layer(self):
        raise NotImplementedError

    def forward(self, lzd_sequences):
        device = next(self.parameters()).device

        n_seq = len(lzd_sequences)
        max_len = max([len(_) for _ in lzd_sequences])

        # Cluster non-terminals into groups which we compute representations simultaneously.
        groups = grouping.create_computation_groups(lzd_sequences, self.grouping_mode)

        h_frag = torch.zeros(n_seq, max_len, self.h_dim).to(device)

        # Character (terminal symbol) representations
        lst_i_seq = torch.LongTensor(groups.char_group[0]).to(device)
        lst_i_pos = torch.LongTensor(groups.char_group[1]).to(device)
        lst_char = torch.LongTensor(groups.char_group[2]).to(device)

        emb_char = self.emb_char(lst_char)

        # Fix values of cell states.
        if self.cell_dim is not None:
            emb_char[:,-self.cell_dim:] = 0

        h_frag[lst_i_seq, lst_i_pos] += emb_char

        # Non-terminal representations
        for group in groups.groups:
            lst_i_seq = torch.LongTensor(group[0]).to(device)
            lst_i_first = torch.LongTensor(group[1]).to(device)
            lst_i_second = torch.LongTensor(group[2]).to(device)
            lst_i_pos = torch.LongTensor(group[3]).to(device)

            # Apply composer.
            h_frag[lst_i_seq, lst_i_pos] +=\
                self.composer_layer(
                    h_frag[lst_i_seq, lst_i_first],
                    h_frag[lst_i_seq, lst_i_second]
                )

        # Mask first 'null' segment as padding vector.
        h_frag[:,0].fill_(0)

        # Ignore cell states
        if self.cell_dim is not None:
            h_frag = h_frag[:,:,:-self.cell_dim]

        return h_frag, groups.non_char_segments

##########################################
# Encoder for re-pair compressed sequence
##########################################

class Base_RePair_Encoder(Base_LZD_Encoder):
    def __init__(self, n_char, h_dim, grouping_mode=0, cell_dim=None):
        super(Base_RePair_Encoder, self).__init__(n_char, h_dim, grouping_mode, cell_dim)

    def forward(self, repair_sequences):
        device = next(self.parameters()).device

        n_seq = len(repair_sequences)

        # Cluster non-terminals into groups which we compute representations simultaneously.
        groups = grouping.create_computation_groups_repair(
            repair_sequences, self.grouping_mode)

        h_factor = torch.zeros(n_seq, groups.length_dummy_seq, self.h_dim).to(device)

        # Character (terminal symbol) representations
        lst_i_seq = torch.LongTensor(groups.char_group[0]).to(device)
        lst_i_pos = torch.LongTensor(groups.char_group[1]).to(device)
        lst_char = torch.LongTensor(groups.char_group[2]).to(device)

        emb_char = self.emb_char(lst_char)

        # Fix values of cell states.
        if self.cell_dim is not None:
            emb_char[:,-self.cell_dim:] = 0

        h_factor[lst_i_seq, lst_i_pos] += emb_char

        # Non-terminal representations
        for group in groups.groups:
            lst_i_seq = torch.LongTensor(group[0]).to(device)
            lst_i_first = torch.LongTensor(group[1]).to(device)
            lst_i_second = torch.LongTensor(group[2]).to(device)
            lst_i_pos = torch.LongTensor(group[3]).to(device)

            composed_rslt = self.composer_layer(
                h_factor[lst_i_seq, lst_i_first],
                h_factor[lst_i_seq, lst_i_second]
            )
            h_factor[lst_i_seq, lst_i_pos] += composed_rslt

        # Ignore cell states
        if self.cell_dim is not None:
            h_factor = h_factor[:,:,:-self.cell_dim]

        return h_factor, groups.non_char_segments

#
# Composers
#

class Fully_Connected_Composer(nn.Module):
    def __init__(self, in_dim, hid_dims, out_dim):
        super(Fully_Connected_Composer, self).__init__()

        self.layers = nn.ModuleList()

        dim_seq = [in_dim*2, *hid_dims, out_dim]
        for i_layer in range(len(dim_seq)-1):
            dim1, dim2 = dim_seq[i_layer], dim_seq[i_layer+1]

            linear = nn.Linear(dim1, dim2)
            nn.init.kaiming_normal_(linear.weight, nonlinearity="leaky_relu")
            self.layers.append(linear)

    def forward(self, input1, input2):
        x = torch.cat([input1, input2], dim=1)
        for i_layer, layer in enumerate(self.layers):
            x = layer(x)
            if i_layer != len(self.layers) - 1:
                x = nn.functional.relu(x)
            else:
                # Without this kind of clipping, we obsedved value explosion of factor representations.
                x = torch.sigmoid(x)

        return x

class Dual_GRU_Composer(nn.Module):
    def __init__(self, h_dim):
        super(Dual_GRU_Composer, self).__init__()

        self.z_gate = nn.Linear(2*h_dim, 3*h_dim)
        self.r_gate = nn.Linear(2*h_dim, 2*h_dim)
        self.out = nn.Linear(2*h_dim, h_dim)

        self.h_dim = h_dim

    def forward(self, input1, input2):
        cat_input = torch.cat([input1, input2], dim=1)

        r = torch.sigmoid(self.r_gate(cat_input))
        o = torch.sigmoid(self.out(cat_input * r))
        # o = torch.sigmoid(self.out(cat_input))

        z = nn.functional.softmax(self.z_gate(cat_input).view(-1, self.h_dim, 3), dim=2)
        column_cat_input = torch.cat(
            [input1.unsqueeze(2), input2.unsqueeze(2), o.unsqueeze(2)], dim=2
        )

        out = torch.sum(column_cat_input * z, dim=2)

        return out
