# Copyright (C) 2020 NEC Corporation
# See LICENSE.

import sys

import datetime
import os
import subprocess
import random

LZD = "lzd_library_root_directory/out/lzd" # <- CHANGE IT!!

def compress_sequence(sequence):
    """
    Arguments:
        sequence -- A sequence of string IDs (1~255) in integer.
    """
    byte_sequence = [_.to_bytes(1, "big") for _ in sequence]

    suffix = str(datetime.datetime.now().time()) + str(random.randint(0, sys.maxsize))

    tmp_filename = f"_temporary_binary_file_{suffix}"
    with open(tmp_filename, "wb") as h:
        for _ in byte_sequence:
            h.write(_)

    tmp_out_filename = f"_temporary_output_binary_file_{suffix}"

    subprocess.run([LZD, "-f", tmp_filename, "-o", tmp_out_filename, "-a", "lzd-seg"])

    compressed_pairs = []
    with open(tmp_out_filename) as h:
        for line in h:
            line = line.replace("\n", "")
            if len(line) > 0:
                compressed_pairs.append(list(map(int, line.split("\t"))))

    os.remove(tmp_filename)
    os.remove(tmp_out_filename)

    return compressed_pairs

if __name__=="__main__":
    pairs = compress_sequence([1,2,3,4,5])
    print(pairs)
