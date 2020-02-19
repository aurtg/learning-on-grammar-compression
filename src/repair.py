# Copyright (C) 2020 NEC Corporation
# See LICENSE.

import sys

import datetime
import os
import subprocess
import random

import struct

EXE = "Re-Pair_root_directory/txt2cfg" # <- CHANGE IT!!

# Rule index will starts from this value.
CHAR_SIZE = 256

def load_repair_data(filename):
    binary_seq = open(filename, "rb").read()

    txt_len, num_rules, seq_len = struct.unpack_from("III", binary_seq[:4 * 3])

    binary_seq = binary_seq[4*3:]

    rules = []
    for i_rule in range(num_rules - CHAR_SIZE):
        first, second = struct.unpack_from("II", binary_seq[:4*2])
        binary_seq = binary_seq[4*2:]

        rules.append((first, second))

    comp_seq = []
    for i_factor in range(seq_len):
        code = struct.unpack_from("I", binary_seq[:4])
        binary_seq = binary_seq[4:]

        comp_seq.append(code[0])

    return rules, comp_seq

def compress_sequence(sequence):
    byte_sequence = [_.to_bytes(1, "big") for _ in sequence]

    suffix = str(datetime.datetime.now().time()) + str(random.randint(0, sys.maxsize))

    while True:
        tmp_filename = f"_temporary_binary_file_repair_{suffix}"
        if os.path.exists(tmp_filename):
            continue
        with open(tmp_filename, "wb") as h:
            for _ in byte_sequence:
                h.write(_)
        break

    tmp_out_filename = f"_temporary_output_binary_file_repair_{suffix}"

    subprocess.run([EXE, tmp_filename, tmp_out_filename],
        stdout=subprocess.DEVNULL)

    rules, comp_seq = load_repair_data(tmp_out_filename)

    os.remove(tmp_filename)
    os.remove(tmp_out_filename)

    return rules, comp_seq

if __name__=="__main__":
    # filename = sys.argv[1]
    #
    # print(load_repair_data(filename))

    print(compress_sequence([1,2,3,1,2,1,2,3,1,2,1,2]))
