#!/usr/bin/env python
import sys
import argparse
import os
from os import path
from shutil import copy2
from pysis import isis
from pysis.exceptions import ProcessError

def parse_args():
    parser = argparse.ArgumentParser(
                                     description="""Create a series of reconstructed and undistorted HiRISE images from radiometrically balanced images from individual CCDs in ISIS cube format, and mosaic them into a single image. 
                                     The script first runs the ISIS programs spiceinit and spicefit on each cube to update and smooth their SPICE, followed by the ISIS program noproj. The user can specify which cube to use as the \"match cube\" for noproj. By default, the script will attempt to use the cube corresponding to the RED5 CCD. The ISIS program hijitreg is run to gather line/sample translations between adjacent noproj'd CCDs.  The resulting line/sample translations are applied when mosaicking the indivdual CCDs to
                                     reconstruct the image using the ISIS program handmos. During mosaicking, the translations from hijitreg are applied such that the position of the match cube is held fixed in the final mosaic.
""")
    parser.add_argument("fromlist",
                        help = "Path to file containing a list of paths to input cubes, one per line.")
    parser.add_argument("matchcube",
                        nargs='?',
                        help = """Path to ISIS cube to be used as the match cube when running noproj.\n\n
                                If omitted, script will attempt to identify and use cube corresponding to RED5. """)
    args = parser.parse_args()
    return args

def get_core_name(cube):
    core_name = cube
    ext = ''
    while core_name != path.splitext(core_name)[0]:
        ext = path.splitext(core_name)[1] + ext
        core_name = path.splitext(core_name)[0]

    return core_name,ext

def hiccd(cube):
    """
    Use ISIS getkey to identify CCD name of a given HiRISE cube (i.e. RED5) and return only its number (i.e. 5)


    Parameters
    ----------
    cube : str
           path to a HiRISE image in ISIS cube format

    Returns
    -------
    ccd : int
          The integer part of the CCD name. Returns None on ProcessError (e.g. keyword not found in cube)

    """

    try:
        ccd = isis.getkey(from_=cube, grpname="Instrument", keyword="CcdId").decode().replace('\n', '')
        for filt in ["RED","IR","BG"]:
            ccd = ccd.replace(filt,'')
    except ProcessError as e:
        ccd = None
    
    if ccd:
        return int(ccd)
    else:
        return None

def get_hijitreg_translations(flat):
    """
    Parse a flatfile created by the ISIS program hijitreg and return a list of the average sample and line translations,
    rounded to the nearest int.


    Parameters
    ----------
    flat : str
           path to a flatfile (text file) created by the ISIS program hijitreg

    Returns
    -------
    st,lt : list
          List of integers representing the average sample,line offsets from input file, rounded to nearest int.

    """
    st = lt = None
    # Get average sample and line translations, rounded to nearest int
    with open(flat, 'r') as f:
        for line in f.readlines():
            if "Average Sample Offset" in line:
                st = round(float(line.split()[4]))
            elif "Average Line Offset" in line:
                lt = round(float(line.split()[4]))
            # Break out of the FOR loop as soon as both offsets are found
            if (st and lt):
                break
    return st,lt

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
        ccd = hiccd(cube)
        ccd_ids.append(ccd)

    ccd_dict = dict(zip(ccd_ids,cubes))

    # Create dict of RED CCDs only
    red_ccd_dict = {k:v for k,v in ccd_dict.items() if k<10}
    minredccd = min([ int(k) for k,v in red_ccd_dict.items() ])
    maxredccd = max([ int(k) for k,v in red_ccd_dict.items() ])

    # If input list does not contain continuous list if RED CCD IDs, error out
    if not len(range(minredccd,maxredccd))+1 == len(red_ccd_dict):
        print("RED CCDs must be continuous", file=sys.stderr)
        print(len(range(minredccd,maxredccd))+1)
        print(len(red_ccd_dict))
        sys.exit(1)
    
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

    # spiceinit all the cubes
    for cube in cubes:
        try:
            isis.spiceinit(from_=cube, attach="yes")
        except ProcessError as e:
            print(e)
            sys.exit(1)

    # spicefit all the cubes
    for cube in cubes:
        try:
            isis.spicefit(from_=cube)
        except ProcessError as e:
            print(e)
            sys.exit(1)

    # noproj all the cubes
    for cube in cubes:
        noprojcube = path.splitext(cube)[0] + ".noproj.cub"
        try:
            isis.noproj(from_=cube, match=matchcube, to=noprojcube, source="frommatch", interp="bilinear")
        except ProcessError as e:
            print(e)
            sys.exit(1)

    # hijitreg on adjacent RED CCDs in order from smallest to largest CCD number
    for i in range(minredccd, maxredccd):
        noprojcube = path.splitext(red_ccd_dict.get(i))[0] + ".noproj.cub"
        match = path.splitext(red_ccd_dict.get(i+1))[0] + ".noproj.cub"
        flat = ("flat.f" + str(i) + "m" + str(i+1) + ".txt")
        try:
            isis.hijitreg(from_=noprojcube, match=match, flatfile=flat)
        except ProcessError as e:
            print(e)
            sys.exit(1)

    # Build a dict of the line/sample offsets from hijitreg, keyed on CCD ID
    trans_dict = dict((ccd, list([None,None]))
                      for ccd in list(range(minredccd,maxredccd+1)) )
                      
    for i in range(minredccd, maxredccd):
        flat = ("flat.f" + str(i) + "m" + str(i+1) + ".txt")
        st,lt = get_hijitreg_translations(flat)
        if (i < matchccd):
            trans_dict.get(i)[0] = st
            trans_dict.get(i)[1] = lt
        else:
            trans_dict.get(i+1)[0] = st
            trans_dict.get(i+1)[1] = lt

    # Name of moscube is based on historical convention in hinoproj Perl script,
    # which assumed matchcube named like "ESP_037396_1985_RED5.balance.cub".
    # The method below is ugly, but included for backwards compatibility.
    
    # Remove all "."-delimited extensions, and then strip off last character
    # (presumably last char is CCD number) to get core_name
    noprojmatchcube = path.splitext(matchcube)[0] + ".noproj.cub"
    core_name = noprojmatchcube
    ext = ''
    while core_name != path.splitext(core_name)[0]:
        ext = path.splitext(core_name)[1] + ext
        core_name = path.splitext(core_name)[0]

    moscube = core_name[0:-1] + "mos_hijitreged" + ext 

    # Copy the noproj'ed matchcube to moscube so as to maintain label info
    # and skip the need to run getkey for number of samples and lines
    # when creating an output cube via handmos
    copy2(noprojmatchcube,moscube)

    # handmos from matchcube to the left
    ssm = slm = 1
    fromccd = (matchccd-1)
    for i in range(fromccd,minredccd-1, -1):
        fromcube = path.splitext(red_ccd_dict.get(i))[0] + ".noproj.cub"
        ssm = ssm + trans_dict.get(i)[0]
        slm = slm + trans_dict.get(i)[1]
        try:
            isis.handmos(from_=fromcube, mosaic=moscube, outsample=ssm, outline=slm, outband=1, priority="beneath")
        except ProcessError as e:
            print(e)
            sys.exit(1)

    # handmos from matchcube to the right
    ssm = slm = 1
    fromccd = matchccd+1
    for i in range(fromccd,maxredccd+1):
        fromcube = path.splitext(red_ccd_dict.get(i))[0] + ".noproj.cub"
        ssm = ssm - trans_dict.get(i)[0]
        slm = slm - trans_dict.get(i)[1]
        try:
            isis.handmos(from_=fromcube, mosaic=moscube, outsample=ssm, outline=slm, outband=1, priority="beneath")
        except ProcessError as e:
            print(e)
            sys.exit(1)

    # Rename the ISIS print.prt file (for backwards compatibility)
    os.replace("print.prt","hinoproj.prt")

if __name__ == "__main__":
    sys.exit(main(parse_args()))
