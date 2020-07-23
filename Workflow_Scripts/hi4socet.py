#!/usr/bin/env python
import sys
import argparse
import os
from os import path
from pysis import isis
from pysis.exceptions import ProcessError
import hinoproj

def parse_args():
    parser = argparse.ArgumentParser(
                                     description="""Prepare a set of HiRISE balance cubes from a single observation for ingestion to Socet Set. 
                                     The script first runs the ISIS programs spiceinit and spicefit on each cube to update and smooth their SPICE, followed by the ISIS program noproj. The user can specify which cube to use as the \"match cube\" for noproj. By default, the script will attempt to use the cube corresponding to the RED5 CCD. The ISIS program hijitreg is run to gather line/sample translations between adjacent noproj'd CCDs.  The resulting line/sample translations are applied when mosaicking the indivdual CCDs to
                                     reconstruct the image using the ISIS program handmos. During mosaicking, the translations from hijitreg are applied such that the position of the match cube is held fixed in the final mosaic. The 32-bit mosaic is then stretched to 8-bits and converted into a raw binary file. The values used for the 32-bit --> 8-bit stretch are saved to *_STRETCH_PAIRS.lis.
                                     Finally, the script runs the ISIS program socetlinescankeywords to extract an instrument sensor description file for Socet Set.
""")
    parser.add_argument("fromlist",
                        help = "Path to file containing a list of paths to input cubes, one per line.")
    parser.add_argument("matchcube",
                        nargs='?',
                        help = """Path to ISIS cube to be used as the match cube when running noproj.\n\n
                                If omitted, script will attempt to identify and use cube corresponding to RED5. """)
    args = parser.parse_args()
    return args

def stretch_pct(cube,out,pct):
    """
    Use ISIS percent to determine pixel value at a given percentile. Run ISIS getkey to extract this value from the
    PVL output of percent.


    Parameters
    ----------
    cube : str
           path to a HiRISE image in ISIS cube format

    out : str
           path to PVL file to old output of call to ISIS percent

    pct : float
           determine what pixel value corresponds to this percentage from image histogram

    Returns
    -------
    val : float
          pixel value that corresponds to the pct % from input image histogram

    """

    try:
        isis.percent(from_=cube, to=out, percentage=pct)
        val = isis.getkey(from_=out, grpname="Results", keyword="Value").decode().replace('\n', '')
    except ProcessError as e:
        val = None
    if val:
        return float(val)
    else:
        return None

def main(args):
    fromlist = args.fromlist
    matchcube = args.matchcube
    # print(args)

    cubes = []
    with open(fromlist , 'r') as inlist:
        cubes = [cube.rstrip() for cube in inlist.readlines()]

    # Create dict of CCD IDs : cubes
    ccd_ids = []
    for cube in cubes:
        # call hiccd func from hinoproj.py
        ccd = hinoproj.hiccd(cube)
        ccd_ids.append(ccd)

    ccd_dict = dict(zip(ccd_ids,cubes))

    # If matchcube not passed in, set it to RED5
    if matchcube is None:
        if 5 in ccd_ids:
            matchcube = ccd_dict.get(5)
            matchccd = 5
        else:
            print("Input list does not contain a RED5 CCD", file=sys.stderr)
            sys.exit(1)
    else:
        matchccd = [ int(k) for k,v in ccd_dict.items() if matchcube in v ][0]

    # Following the noproj naming convention, create the noproj'ed mosaic
    # name (needed now to base the output keywords.lis file)
    # Remove all "."-delimited extensions, and then strip off last character
    # (presumably last char is CCD number) to get core_name

    # Call the get_core_name func from hinoproj.py
    core_name = hinoproj.get_core_name(matchcube)[0]
    moscube = str(core_name[0:-1]) + "mos_hijitreged.balance.noproj.cub" 

    # Call the main loop from hinoproj.py
    # This is actually doing the bulk of the processing, including noproj and handmos
    hinoproj.main(args)

    # Get stretch values
    minpvl = "p0005.temp.txt"
    maxpvl = "p9995.temp.txt"
    try:
        minval = stretch_pct(moscube, minpvl,  0.05)
        maxval = stretch_pct(moscube, maxpvl, 99.95)
    except ProcessError as e:
        print(e)
        sys.exit()

    # delete files output by stretch_pct
    os.remove(minpvl)
    os.remove(maxpvl)

    # Save stretch pairs as formatted string to a file
    stretch_rpt = (core_name[0:-5] + "_STRETCH_PAIRS.lis")
    if minval>0:
        s = ("image: " + moscube + "\n" + "stretch pairs: \"0:0 " + str(minval) + ":1 " + str(maxval) + ":254\"\n")
    else:
        s = ("image: " + moscube + "\n" + "stretch pairs: \"" + str(minval-1) + ":0 " + str(minval) + ":1 " + str(maxval) + ":254\"\n")

    try:
        with open(stretch_rpt, 'w') as rpt:
            r = rpt.write(s)
            rpt.close()
    except:
        print("Failed to write " + stretch_rpt, file=sys.stderr)
        sys.exit(1)
        
    # Stretch 32-bit mosaic to 8-bit
    bytecube = path.splitext(moscube)[0] + ".8bit.cub"
    if minval > 0:
        try:
            isis.stretch(from_=moscube, to=bytecube+"+8bit+1:254",
                         pairs="0:0 " + str(minval) + ":1 " + str(maxval) + ":254",
                         lis=1.0, lrs=1.0, his=254, hrs=254)
        except ProcessError as e:
            print(e)
            sys.exit(1)
    else:
        try:
            isis.stretch(from_=moscube, to=bytecube+"+8bit+1:254",
                         pairs=str(minval-1) + ":0 " + str(minval) + ":1 " + str(maxval) + ":254",
                         lis=1.0, lrs=1.0, his=254, hrs=254)
        except ProcessError as e:
            print(e)
            sys.exit(1)
            

    # isis2raw on 8-bit mosaic
    byteraw = path.splitext(bytecube)[0] + ".raw"
    try:
        isis.isis2raw(from_=bytecube, to=byteraw, bittype="8bit", stretch="none")
    except ProcessError as e:
        print(e)
        sys.exit(1)

    # socetlinescankeywords on 32-bit mosaic
    keyfile = (hinoproj.get_core_name(moscube)[0] + "_keywords.lis") 
    try:
        isis.socetlinescankeywords(from_=moscube, to=keyfile)
    except ProcessError as e:
        print(e)
        sys.exit(1)

    # rename print.prt to hi4socet.prt
    os.replace("print.prt","hi4socet.prt")


if __name__ == "__main__":
    sys.exit(main(parse_args()))
