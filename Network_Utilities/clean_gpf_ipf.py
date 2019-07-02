#!/usr/bin/env python3

import sys
from os import path
import argparse
import pandas as pd
import numpy as np
from plio.io.io_bae import read_gpf, save_gpf, read_ipf, save_ipf


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description = """Delete inactive points in a given GPF and 1 or more IPF files, and perform inner join based on Point IDs to create a new set of GPF and IPFs without orphan points.""")
    parser.add_argument("--ipf-only",
                        action='store_true',
                        help = """Only save the updated IPFs. If input_gpf is known a priori to be clean, this option avoids saving a duplicate of it to disk.""")
    parser.add_argument("--suffix",
                        nargs='?',
                        default='clean',
                        const='clean',
                        help = """Suffix to attach to output filenames. Default is 'clean'.""")
    parser.add_argument("output_dir",
                        help = """The directory to write the cleaned GPF and IPFs to.""")
    parser.add_argument("input_gpf",
                        help = "The name of the GPF to modify")
    parser.add_argument("input_ipf",
                        nargs='+',
                        help = "The name of the IPFs to modify")
    args = parser.parse_args()
    return args

def main(ipf_only,suffix,output_dir,input_gpf,input_ipf):
    gpf_df = read_gpf(input_gpf)

    # Copy active points into a new dataframe
    gpf_active_pts = gpf_df.loc[gpf_df['stat'] == 1]

    ipf_df = read_ipf(list(input_ipf))
    ipf_active_pts = ipf_df.loc[ipf_df['val'] == 1] 

    # Inner Join the GPF and IPF dataframes based on PointIDs
    merged_df = pd.merge(gpf_active_pts, ipf_active_pts, left_on='point_id', right_on='pt_id')

    # Even after inner join, it is possible that a Point ID in the GPF only exists in a single IPF
    # We eliminate such cases by returning a view that only contains *duplicate* Point IDs
    merged_clean_df  = merged_df[merged_df.duplicated(['pt_id'],keep=False)]

    # Now we split the merged clean dataframe back into separate dataframes for the GPF and IPFs
    clean_gpf = merged_clean_df.drop_duplicates(subset=['point_id']).iloc[:,0:12]
    clean_ipf = merged_clean_df.iloc[:,12:25]

    # Append '_clean' to the IPF filenames in clean_ipf
    clean_ipf['ipf_file'] = (clean_ipf['ipf_file'] + '_' + suffix)

    gpf_base = path.split(path.splitext(input_gpf)[0])[1]
    output_gpf_path = path.join(output_dir,gpf_base + '_' + suffix + '.gpf')

    # TODO : Check that args.output_dir exists
    # TODO : Check whether file already exists and error out to prevent overwriting

    # Save cleaned GPF + IPFs to disk
    if ipf_only is False:
        save_gpf(clean_gpf,output_gpf_path)

    save_ipf(clean_ipf,output_dir)



if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(**vars(args)))

    
