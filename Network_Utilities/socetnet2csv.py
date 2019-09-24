#!/usr/bin/env python3

import sys
from os import path
import argparse
import pandas as pd
import numpy as np
from plio.io.io_bae import read_gpf, read_ipf

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description = """Perform an inner join on the point IDs of a GPF and set of IPFs, write the resulting dataframe to a CSV.""")
    # parser.add_argument("--ipf-only",
    #                     action='store_true',
    #                     help = """Only save the updated IPFs. If input_gpf is known a priori to be clean, this option avoids saving a duplicate of it to disk.""")
    parser.add_argument("output_csv",
                        help = """Path to output CSV.""")
    parser.add_argument("input_gpf",
                        help = "The name of the GPF.")
    parser.add_argument("input_ipf",
                        nargs='+',
                        help = "The name of the IPFs associated with the GPF.")
    args = parser.parse_args()
    return args

def main(output_csv,input_gpf,input_ipf):
    gpf_df = read_gpf(input_gpf)

    ipf_df = read_ipf(list(input_ipf))

    # Inner Join the GPF and IPF dataframes based on PointIDs
    merged_df = pd.merge(gpf_df, ipf_df, left_on='point_id', right_on='pt_id')

    columns = merged_df.columns.values.tolist()

    merged_df['lat_Y_North'] = np.degrees(merged_df['lat_Y_North'])
    merged_df['long_X_East'] = np.degrees(merged_df['long_X_East'])

    merged_df.to_csv(path_or_buf=output_csv,header=True,index=False,columns=list(merged_df.columns))

if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(**vars(args)))

