#!/usr/bin/env python3

import sys
import os
import argparse
import subprocess
import readline
import pandas as pd
import numpy as np
from plio.io.io_bae import read_gpf, save_gpf


## Create an argument parser
def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description = """Transform points in a Socet Set Ground Point File (GPF).
The transformed latitude, longitude, and height values from the Tie Points are then written to a new GPF with their sigmas set equal to 1 and the "known" flag 
changed from "1" (Tie Point) to "3" (XYZ Control). Non-Tie Points from the original GPF are written to the new GPF with their "known" flags changed to "1." 
Tie Points from the original GPF that were not active ("stat" = 0) are copied "as-is" into the new GPF. The output GPF preserves the order of the ground points from the original GPF.

If it is desired to update all active points in the input GPF, use the '--all-points' flag. The modified points will still have their "known" flag set to "3" (XYZ Control) in the 
output GPF.

The script requires the "plio" Python library (https://github.com/USGS-Astrogeology/plio) in order to read/write GPFs. 
The Ames Stereo Pipeline program pc_align must be available in the user's path or somewhere else where Python can find it. 
More information about the Ames Stereo Pipeline is available on the project's Git repository: https://github.com/NeoGeographyToolkit/StereoPipeline""",
                                     epilog = """EXAMPLES:
                                     Align Tie Points from a HiRISE DTM in Socet's ASCII DTM format to a reference DTM in GeoTIFF format. 
The Socet ASCII DTM will be automatically converted to a pc_align-compatible CSV, but the user should specify either the datum or planetary radii that describe the 
surface that the Socet DTM's heights are referenced to. In the following example, the Socet DTM is referenced to the Mars 2000 ellipsoid.
    %(prog)s CTX_reference_dtm.tif raster HiRISE_Gale_low_res.asc ascii_dtm --radii 3396190 3376000 HiRISE_Gale.gpf output_HiRISE_Gale.gpf --max-displacement 50 \n

""")
    parser.add_argument("socet_gpf",
                        help = "The name of the Socet Ground Point File to transform.")
    parser.add_argument("transform_matrix",
                        help = """Name of a pc_align-compatible transformation matrix to apply to the input GPF.""")
    parser.add_argument("tfm_socet_gpf",
                        help = """Name to use for the output (transformed) ground point file. Must include ".gpf" extension.""")
    parser.add_argument("--all-points",
                        action='store_true',
                        help = "This flag will force updating of all active (stat = 1) points in the input GPF, not just tie points.")
    refshape = parser.add_mutually_exclusive_group(required=True)
    refshape.add_argument("--datum",
                        nargs=1,
                        choices=['D_MARS', 'D_MOON', 'MOLA', 'NAD27', 'NAD83', 'WGS72', 'WGS_1984'],
                        help = """Use this datum for heights in the input GPF.""")
    refshape.add_argument("--radii",
                          nargs=2,
                          metavar=('semi-major-axis','semi-minor-axis'),
                          type=float,
                          help="""Semi-major and semi-minor axes, expressed in meters, that define the ellipsoid that heights in the input GPF are referenced to.""")
    args = parser.parse_args()
    return args

### Main Loop ###
## Parse arguments
args = parse_arguments()

print(args)

socet_gpf = args.socet_gpf
socet_gpf_basename = os.path.splitext(socet_gpf)[0]
transform_matrix = args.transform_matrix
tfm_socet_gpf = args.tfm_socet_gpf

if tfm_socet_gpf[-4:] != ".gpf":
    print("""USER ERROR: Output file name must include ".gpf" extension""")
    sys.exit(1)

gpf_align_prefix = os.path.splitext(tfm_socet_gpf)[0]

## Read in the Socet ground point file using plio's read_gpf()
gpf_df = read_gpf(socet_gpf)
# Set the index of the GPF dataframe to be the point_id column
gpf_df.set_index('point_id', drop=False, inplace=True)


## If user passed "--all-points" option, copy all active points to new data frame
##  Otherwise, only copy tie points (known == 0) that are active
if args.all_points is True:
    tp_df =  gpf_df[(gpf_df.stat == 1)].copy()
else:
    tp_df = gpf_df[(gpf_df.known == 0) & (gpf_df.stat == 1)].copy()

# Convert lat/longs from radians to degrees
print("Converting Tie Point lat/long from radians to degrees")
tp_df.lat_Y_North = np.degrees(tp_df.lat_Y_North)
tp_df.long_X_East = ((360 + np.degrees(tp_df.long_X_East)) % 360)
        
## Write the lat/long/height values of the tie points to a CSV (compatible with pc_align)
tp_df.to_csv(path_or_buf=(socet_gpf_basename + '.csv'),
             header=False,
             index=False,
             columns=['lat_Y_North',
                      'long_X_East',
                      'ht'])

## Collect pc_align arguments to apply transform into a list
## Set num-iterations = 0 and turn off max-displacement (-1) because only going to apply existing transform
apply_tfm_args = ["pc_align",
                "--initial-transform",transform_matrix,
                "--num-iterations","0",
                "--max-displacement","-1",
                "--save-transformed-source-points",
                "-o", gpf_align_prefix ]

## Extend the list of arguments for pc_align to include the datum or radii as necessary
if args.datum is not None:
    apply_tfm_args.extend(["--datum", str(args.datum[0])])
elif args.radii is not None:
    apply_tfm_args.extend(["--semi-major-axis", str(args.radii[0]), "--semi-minor-axis", str(args.radii[1])])

## Extend the list to place point clouds at the end of the list of arguments for pc_align
## Note that we're specifying the same file as the reference and source clouds because pc_align requires 2 files as input,
##  even if we're only applying a transform and not iterating
apply_tfm_args.extend([(socet_gpf_basename + '.csv'),(socet_gpf_basename + '.csv')])

## Apply transform to the tie points CSV
print("Calling pc_align with 0 iterations to apply transform to points from GPF")
try:
    run_apply_tfm = subprocess.run(apply_tfm_args,
                                 stdout=subprocess.PIPE,
                                 encoding='utf-8')
    print(run_apply_tfm.stdout)
except subprocess.CalledProcessError as e:
    print(e)
    sys.exit(1)

## mergeTransformedGPFTies
### Ingest the transformed tie points to a pandas data frame
t = np.genfromtxt((gpf_align_prefix + '-trans_source.csv'),
                  delimiter=',',
                  skip_header=3,
                  dtype='unicode')
id_list = tp_df['point_id'].tolist()
tfm_index = pd.Index(id_list)
tfm_tp_df = pd.DataFrame(t, index=tfm_index, columns=['lat_Y_North',
                                                      'long_X_East',
                                                      'ht'])
tfm_tp_df = tfm_tp_df.apply(pd.to_numeric)

## Update the original Tie Point data frame with the transformed lat/long/z values from pc_align
tp_df.update(tfm_tp_df)

# ### Convert long from 0-360 to +/-180 and convert lat/long back to radians
tp_df.lat_Y_North = np.radians(tp_df['lat_Y_North'])
tp_df.long_X_East = np.radians(((tp_df['long_X_East'] + 180) % 360) - 180)

print("Updating GPF data frame  with Transformed lat/long/z values from pc_align")
gpf_df.update(tp_df)

print("Updating ground point types in output GPF")

## Build boolean masks of gpf_df to enable selective updating of point types
## transformed point mask
if args.all_points is True:
    tfm_tp_mask = (gpf_df['stat'] == 1)
else:
    ## non-tie point mask
    non_tp_mask = ((gpf_df['stat'] == 1) & (gpf_df['known'] > 0))
    tfm_tp_mask = ((gpf_df['stat'] == 1) & (gpf_df['known'] == 0))
    ## Change non-Tie Points to Tie Point
    print("Changing active non-Tiepoints to Tiepoints")
    gpf_df.loc[non_tp_mask, 'known'] = 0

## Change transformed points to XYZ Control, sigmas = 1.0, residuals = 0.0
print("Changing transformed points to XYZ Control with sigmas = 1 and residuals = 0")
gpf_df.loc[tfm_tp_mask, 'known'] = 3
gpf_df.loc[tfm_tp_mask, 'sig0':'sig2'] = 1.0
gpf_df.loc[tfm_tp_mask, 'res0':'res2'] = 0.0

## Convert the 'stat' and 'known' columns to unsigned integers
gpf_df.known = pd.to_numeric(gpf_df['known'], downcast = 'unsigned')
gpf_df.stat = pd.to_numeric(gpf_df['stat'], downcast = 'unsigned')

print("Writing transformed GPF to file: " + tfm_socet_gpf)
save_gpf(gpf_df, tfm_socet_gpf)

## Write list of pointIDs of the transformed tiepoints to a file
## Included for legacy compatibility, not actually used for anything
tp_df.to_csv(path_or_buf=(socet_gpf_basename + '.tiePointIds.txt'),
             sep=' ', header=False,
             index=False,
             columns=['point_id'])
