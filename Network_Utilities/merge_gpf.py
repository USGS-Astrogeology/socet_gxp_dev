#!/usr/bin/env python3

import sys
from os import path
import argparse
import pandas as pd
import numpy as np
from plio.io.io_bae import read_gpf, save_gpf


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description = """Merge several GPF files into a single file.""")
    parser.add_argument("output_path",
                        help = """Path to output file.""")
    parser.add_argument("input_list",
                        nargs='+',
                        help = "List of GPFs to merge together.")
    args = parser.parse_args()
    return args

def main(output_path,input_list):
    frames = []
    for gpf in input_list:
        frames.append(read_gpf(gpf))

    out_df = pd.concat(frames)
    save_gpf(out_df,output_path)



if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(**vars(args)))

