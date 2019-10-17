#!/usr/bin/env python3

import sys
import argparse
import pandas as pd
import numpy as np
from plio.io.io_bae import read_gpf

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description = """Convert a Socet GPF file to CSV.""")
    parser.add_argument("gpf",
                        help = "Input GPF to convert")
    parser.add_argument("outpath",
                        help = """Path to output file.""")
    parser.add_argument("--no-convert",
                        action='store_true',
                        help = "When set, the lat_Y_North and long_X_East fields will be copied as-is into output file. By default, these fields are assumed to be in radians, and are converted to degrees.")
    args = parser.parse_args()
    return args


def main(gpf,outpath,no_convert):

    df = read_gpf(gpf)

    columns = df.columns.values.tolist()

    if not no_convert:
        df['lat_Y_North'] = np.degrees(df['lat_Y_North'])
        df['long_X_East'] = np.degrees(df['long_X_East'])

    df.to_csv(path_or_buf=outpath,header=True,index=False,columns=columns)



if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(**vars(args)))

