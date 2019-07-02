#!/usr/bin/env python3

import sys
from os import path
import argparse
import pandas as pd
import numpy as np
from plio.io.io_bae import read_ipf, save_ipf


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description = """Merge several IPF files into a single file.""")
    parser.add_argument("output_path",
                        help = """Path to output file.""")
    parser.add_argument("input_list",
                        nargs='+',
                        help = "List of IPFs to merge together.")
    args = parser.parse_args()
    return args

def main(output_path,input_list):
    df = read_ipf(input_list)
    df['ipf_file'] = path.splitext(path.split(output_path)[1])[0]
    out_dir = path.split(output_path)[0]
    save_ipf(df,out_dir)



if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(**vars(args)))


