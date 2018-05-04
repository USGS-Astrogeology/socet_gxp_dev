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
                                     description = """This script aligns Tie Points from a Socet Set/Socet GXP Ground Point File (GPF) to a reference elevation data set using the pc_align program from the NASA Ames Stereo Pipeline. Typically, Tie Points are too sparse to be reliably aligned to a reference using the iterative closest points algorithm in pc_align. Therefore, this script allows pc_align to first be applied to a (potentially low-resolution) digital terrain model that corresponds to the GPF. The resulting transformation matrix is applied to the Tie Points during a second call to pc_align.

The transformed latitude, longitude, and height values from the Tie Points are then written to a new GPF with their sigmas set equal to 1 and the "known" flag changed from "1" (Tie Point) to "3" (XYZ Control). Non-Tie Points from the original GPF are written to the new GPF with their "known" flags changed to "1." Tie Points from the original GPF that were not active ("stat" = 0) are copied "as-is" into the new GPF. The output GPF preserves the order of the ground points from the original GPF.

The script requires the "plio" Python library (https://github.com/USGS-Astrogeology/plio) in order to read/write GPFs. 

The Ames Stereo Pipeline program pc_align must be available in the user's path or somewhere else where Python can find it. More information about the Ames Stereo Pipeline is available on the project's Git repository: https://github.com/NeoGeographyToolkit/StereoPipeline""",
                                     epilog = """EXAMPLES:
Align Tie Points from a stereopair to MOLA shot data using a low-resolution version of the CTX DTM is Socet's ASCII format. The MOLA shot data are in the table format produced by "pedr2tab."
    %(prog)s MOLA_reference.tab table CTX_NE_Syrtis_low_res_aate.asc CTX_NE_Syrtis.gpf tfm_CTX_NE_Syrtis.gpf \n

Align Tie Points from a stereopair to a DTM in Socet's ASCII format.
    %(prog)s CTX_reference_dtm.asc ascii_dtm HiRISE_Gale_low_res.asc HiRISE_Gale.gpf output_HiRISE_Gale.gpf
""")
    parser.add_argument("referencePC",
                        help="The name of the file that contains the reference point cloud. File must be in pedr2tab or Socet ASCII DTM format.")
    parser.add_argument("referenceFormat",
                        # TODO: add support for "CSV" and "raster." The latter would refer to any ASP-supported raster format, such as GeoTIFF or ISIS3 cube
                        choices = ["table", "ascii_dtm", "csv"],
                        type = str.lower, # effectively make this case insensitive
                        help = """A flag indicating the format of the reference PC. "table" is MOLA shot data output by the program "pedr2tab," "ascii_dtm" is a Socet ASCII DTM," and "CSV" is any pc_align compatible comma delimited text file.""")
    parser.add_argument("socet_dtm",
                        help = "The name of the file containing the Socet Set or GXP DTM to be aligned. Must be in ASCII format.")
    parser.add_argument("socet_gpf",
                        help = "The name of the Socet Ground Point File that corresponds to socet_dtm.")
    parser.add_argument("tfm_socet_gpf",
                        help = """Name to use for the output (transformed) ground point file. Must include ".gpf" extension.""")
    parser.add_argument("--datum",
                        nargs=1,
                        default="D_MARS",
                        choices=['D_MARS', 'D_MOON', 'MOLA', 'NAD27', 'NAD83', 'WGS72', 'WGS_1984'],
                        help = """Use this datum for CSV files. Defaults to D_MARS for legacy compatibility.""")
    parser.add_argument("--max-displacement",
                        nargs=1,
                        default='300',
                        # The float needs to be converted to a string for subprocess.run()
                        #  but forcing it to be a float in argparse is a lazy way of
                        #  minimizing chances the user passes something invalid
                        type=float,
                        help="""Maximum expected displacement of source points as result of alignment, in meters. Used for removing gross outliers from source point cloud. Defaults to 300 meters for legacy compatibility.""")
    parser.add_argument('pc_align_args',
                        nargs = argparse.REMAINDER,
                        help = """Additional arguments that will be passed directly to pc_align.""")
    args = parser.parse_args()
    return args


def ascii_dtm2csv(ascii_dtm, outname):
    """
    Read an ASCII DTM from Socet Set into a pandas data frame,
    write out to CSV with latitude and longitude columns swapped
    

    Parameters
    ----------
    ascii_dtm : str
                path to the input ASCII DTM from Socet Set

    outname : str
              path to the output CSV

    """
    d = np.genfromtxt(ascii_dtm, skip_header=14, dtype='unicode')
    ref_df = pd.DataFrame(d, columns=["long","lat","z"])
    ref_df = ref_df.apply(pd.to_numeric)

    ## Extract lat/long/z and write out CSV. Note swapped lat/long columns
    ref_df.to_csv(path_or_buf=outname, header=False, index=False, columns=['lat','long','z'])
    return
    

### Main Loop ###
## Parse arguments
args = parse_arguments()

print(args)

referencePC = args.referencePC
ref_basename = os.path.splitext(referencePC)[0]
socet_gpf = args.socet_gpf
socet_gpf_basename = os.path.splitext(socet_gpf)[0]
socet_dtm = args.socet_dtm
socet_dtm_basename = os.path.splitext(socet_dtm)[0]
tfm_socet_gpf = args.tfm_socet_gpf

if tfm_socet_gpf[-4:] != ".gpf":
    print("""USER ERROR: Output file name must include ".gpf" extension""")
    sys.exit(1)

## Logic to figure out what to do with referencePC depending on format

if args.referenceFormat == "table":
    ### "table" is the fixed-length ascii table output by pedr2tab
    ref_dtm = (ref_basename + "_RefPC.csv")
    ## Ingest to pandas dataframe
    d = np.genfromtxt(referencePC, skip_header=2, dtype='unicode')
    ref_df = pd.DataFrame(d, columns=["long_East", "lat_North", "topography",
                                  "MOLArange", "planet_rad", "c",
                                  "A",  "offndr",  "EphemerisTime",
                                  "areod_lat", "areoid_rad", "shot",
                                  "pkt", "orbit", "gm"])
    ref_df = ref_df.apply(pd.to_numeric)
    ### Extract ographic lat, lon, topo columns and write out to CSV
    ### NOTE: pc_align doesn't understand that elevation values are relative to a geoid
    ### This is here for legacy compatibility ONLY
    print("\n\n *** WARNING: Using MOLA heights above geoid ***\n\n")
    ref_df.to_csv(path_or_buf=ref_dtm, header=False, index=False,
                  columns=['areod_lat','long_East','topography'])
elif args.referenceFormat == "ascii_dtm":
    ### "ascii_dtm" is a Socet Set format ASCII DTM
    ### Convert to CSV and swap order of lat/long columns
    ascii_dtm2csv(referencePC, ref_dtm)
elif args.referenceFormat == "csv":
    ref_dtm = referencePC
else:
    ## If argparse has done its job, we should never fall through to this point
    print("PROGRAMMER ERROR: Unable to determine reference elevation format")
    sys.exit(1)
    
## Convert the Socet ASCII DTM to CSV for pc_align
socet_dtm_csv = (socet_dtm_basename + ".csv")
ascii_dtm2csv(socet_dtm,socet_dtm_csv)

    
## Read in the Socet ground point file using plio's read_gpf()
gpf_df = read_gpf(socet_gpf)
# Set the index of the GPF dataframe to be the point_id column
gpf_df.set_index('point_id', drop=False, inplace=True)


## Create copy of tie points (known = 0) that are on (stat = 1)
tp_df = gpf_df[(gpf_df.known == 0) & (gpf_df.stat == 1)].copy()
# print(tp_df.head())

# Convert lat/longs from radians to degrees
print("Converting Tie Point lat/long from radians to degrees")
tp_df.lat_Y_North = np.degrees(tp_df.lat_Y_North)
tp_df.long_X_East = ((360 + np.degrees(tp_df.long_X_East)) % 360)


# print(tp_df.head())

align_prefix = (socet_dtm_basename + '_pcAligned_DTM')
gpf_align_prefix = (socet_dtm_basename + '_pcAligned_gpfTies')

## Collect arguments for pc_align subprocess into a list
align_args = ["pc_align",
                "--max-displacement", str(args.max_displacement[0]),
                "--datum", args.datum,
                "-o", align_prefix,
                "--save-inv-trans"]

## If the user passed additional arguments for pc_align, extend align_args to include them
if args.pc_align_args:
    align_args.extend(args.pc_align_args)

## Source and reference files must come last in call to pc_align
align_args.extend([socet_dtm_csv, ref_dtm])
print(align_args)

try:
    print("Running pc_align on " + socet_dtm_csv + " and " + ref_dtm)
    run_align = subprocess.run( align_args,
                                check=True,
                                stderr=subprocess.STDOUT,
                                encoding='utf-8')
except subprocess.CalledProcessError as e:
    print(e)
    sys.exit(1)
    
## Write the lat/long/height values of the tie points to a CSV (compatible with pc_align)
tp_df.to_csv(path_or_buf=(socet_gpf_basename + '.csv'),
             header=False,
             index=False,
             columns=['lat_Y_North',
                      'long_X_East',
                      'ht'])

## Collect pc_align arguments to apply transform into a list
apply_tfm_args = ["pc_align",
                "--initial-transform",(align_prefix + '-transform.txt'),
                "--num-iterations","0",
                "--max-displacement","-1",
                "--datum", args.datum,
                "--save-inv-trans",
                "-o", gpf_align_prefix ,
                (socet_gpf_basename + '.csv'),
                ref_dtm ]


## Apply transform from previous pc_align run to tie points CSV
print("Calling pc_align with 0 iterations to apply transform from previous run to Tie Points from GPF")
run_apply_tfm = subprocess.run(apply_tfm_args,
                                 stdout=subprocess.PIPE,
                                 encoding='utf-8')
print(run_apply_tfm.stdout)


## mergeTransformedGPFTies
### Ingest the transformed tie points to a pandas data frame
t = np.genfromtxt((gpf_align_prefix + '-trans_reference.csv'),
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


print("Updating GPF DataFrame with Transformed lat/long/z values from pc_align")
gpf_df.update(tp_df)

## Build boolean masks of gpf_df to enable selective updating of point types
## non-tie point mask
non_tp_mask = (gpf_df['known'] > 0)
## transformed tie point mask
tfm_tp_mask = ((gpf_df['stat'] == 1) & (gpf_df['known'] == 0))


print("Updating ground point types in output GPF")

## Change non-Tie Points to Tie Point
print("Changing active non-Tiepoints to Tiepoints")
gpf_df.loc[non_tp_mask, 'known'] = 0

## off tie point mask
## no-op

## Change transformed Tie Points to XYZ Control, sigmas = 1.0, residuals = 0.0
print("Changing transformed Tiepoints to XYZ Control with sigmas = 1 and residuals = 0")
gpf_df.loc[tfm_tp_mask, 'known'] = 3
gpf_df.loc[tfm_tp_mask, 'sig0':'sig2'] = 1.0
gpf_df.loc[tfm_tp_mask, 'res0':'res2'] = 0.0

## Convert the 'stat' and 'known' columns to unsigned integers
gpf_df.known = pd.to_numeric(gpf_df['known'], downcast = 'unsigned')
gpf_df.stat = pd.to_numeric(gpf_df['stat'], downcast = 'unsigned')


print("Writing transformed GPF to file: " + tfm_socet_gpf)
save_gpf(gpf_df, tfm_socet_gpf)

## Write list of pointIDs of the tiepoints to a file
## Included for legacy compatibility, not actually used for anything
tp_df.to_csv(path_or_buf=(socet_gpf_basename + '.tiePointIds.txt'),
             sep=' ', header=False,
             index=False,
             columns=['point_id'])
