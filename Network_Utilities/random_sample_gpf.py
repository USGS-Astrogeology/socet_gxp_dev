#!/usr/bin/env python3

import sys
import os
import argparse
import pandas as pd
import numpy as np
from plio.io.io_bae import read_gpf, save_gpf


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description = """Select a random sample of points in GPF and save result to disk. Uses pandas.DataFrame.sample, without replacement. Any points with stat == 0 will be filtered out before sampling.""")
    parser.add_argument("--frac", type=float, required=True,
                        help = "The fraction of points to return. Must be floating point value on interval (0,1].")
    parser.add_argument("input_gpf",
                        help = "The name of the GPF to modify")
    parser.add_argument("output_gpf",
                        help = """The desired name of the output GPF""")
    args = parser.parse_args()
    return args

### Main Loop ###
## Parse arguments
args = parse_arguments()

gpf_df = read_gpf(args.input_gpf)
# Copy active points into a new dataframe
active_pts = gpf_df.loc[gpf_df['stat'] == 1]
# Randomly sample the active points
sampled_pts = active_pts.sample(frac=args.frac, replace=False)
# Save result to disk
save_gpf(sampled_pts,args.output_gpf)
