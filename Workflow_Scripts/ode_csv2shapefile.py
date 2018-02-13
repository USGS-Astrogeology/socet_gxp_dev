#!/usr/bin/env python3
###############################################################################
#_TITLE  ode_csv2shapefile.py
#
#_ARGS  
#  target =  Mars | Moon | Mercury
#  --input input.csv (as downloaded from PDS ODE using ode_get_laser_alt.py)
#
#_USER  Command line options [optional parameters]:
#   target:  {Mars, Moon, Mercury}
#   --input input.csv
#  
#   ode_csv2shapefile.py target --input in.csv
#   ode_csv2shapefile.py Moon --input LolaRDR_24N25N_341E342E_pts_csv.csv
#   ode_csv2shapefile.py Mars --input MOLApedr_24N25N_341E342E_pts_csv.csv
#   -- to convert many files
#   ode_csv2shapefile.py Mars --pattern "*_pts_csv.csv"
#
#_REQUIRES
#        Python 3.x and argparse
#        GDAL for Python 
#        --recommended Anaconda Python 3.x environment w/ gdal, once installed:
#        $ conda install -c conda-forge gdal 
#
#_DESC 
#        Convert from a ODE lidar shot CSV to an Esri shapefile PointZ.
#        The CSV is expected to be created using ode_get_laser_alt.py which
#        uses the PDS ODE REST calls to download LOLA, MOLA, and MLA laser
#        altimeter pont shots. for more: http://oderest.rsl.wustl.edu/
#
#        For all bodies, Longitudes will be converted to -180 to 180.
#        For Mars, latitudes are converted from ocentric to ographic. This
#        can be easily changed below.
#
#_CALLS  List of calls:
#
#_HIST
#        Aug 17 2017 - Trent Hare (thare@usgs.gov) - original version
#        Sep 14 2017 - added MLA support
#
#_LICENSE
#        Public domain (unlicense)
#   
#_END
###############################################################################
import os
import sys
import csv
import glob
import math
import argparse
from osgeo import ogr

#ocentric to ographic latitudes
def oc2og(dlat, dMajorRadius, dMinorRadius):
    try:    
        dlat = math.radians(dlat)
        dlat = math.atan(((dMajorRadius / dMinorRadius)**2) * (math.tan(dlat)))
        dlat = math.degrees(dlat)
    except:
        print ("Error in oc2og conversion")
    return dlat

#ographic to ocentric latitudes
def og2oc(dlat, dMajorRadius, dMinorRadius):
    try:
        dlat = math.radians(dlat)
        dlat = math.atan((math.tan(dlat) / ((dMajorRadius / dMinorRadius)**2)))
        dlat = math.degrees(dlat)
    except:
        print ("Error in oc2og conversion")
    return dlat

# Convert longitudes to -180 to 180 degrees
def LonTo180(dlon):
    if (dlon > 180.0):
        dlon = dlon - 360.0
    return dlon

def parse_arguments():
    ## Parse commandline args with argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
             description="""Convert ODE created LOLA, MOLA, or MLA shot data to an Esri pointZ Shapefile.
    The CSV is expected to be generated from the script ode_get_laser_alt.py""",
             epilog="""EXAMPLES:
             %(prog)s Mars --input ode_lolardr.csv 
             %(prog)s Moon --input ode_molapedr.csv
             %(prog)s Mercury --pattern "*_pts_csv.csv"
    
    """)
    
    #parser.add_argument('product', choices=["lolardr","molapedr","mla"], 
    #                    help="Specify desired product type: LOLA RDR or MOLA PEDR")
    parser.add_argument('target', choices=["mars","moon","mercury"], 
                        type = str.lower, 
                        help="Specify which target: Mars, Moon, Mercury")
    
    ## User must specify exactly one of --coords or --raster
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', nargs=1, metavar="file.csv",
                        help='an input ODE csv file as downloaded using ode_get_laser_alt.py.')
    group.add_argument('--pattern',  nargs=1, metavar='"*_pts_csv.csv"', 
                        default='"*_pts_csv.csv"', dest='pattern',
                        help='pattern to find and run on many strings. Default pattern is "*_pts_csv.csv".')
    
    args = parser.parse_args()
    return args


args = parse_arguments()
target = args.target

if args.input is not None:
    files = args.input
elif args.pattern is not None:
    pattern = args.pattern[0]
    files = glob.glob(pattern)
else:
    argparse.parser.print_usage()
    sys.exit(1)
    
	
if target == "mars":
    targetWKT = "Mars"
    majorRadius = 3396190.0 
    minorRadius = 3376200.0
    year = 2000
    #From ode Mars run the fields should be:
    #LONG_EAST, LAT_NORTH, TOPOGRAPHY, MOLA_RANGE, PLANET_RAD, C, A, EMPHEMERIS_TIME, UTC, ORBIT
    longField = 'LONG_EAST'
    latField = 'LAT_NORTH'
    elevField = 'TOPOGRAPHY'
    radiusField = 'PLANET_RAD'
    utcField = 'UTC'
    orbitField = 'ORBIT'
elif target == "mercury":
    targetWKT = "Mercury"
    majorRadius = minorRadius = 2439400.0
    year = 2015
    #From ode Mercury run the fields should be:
    #longitude,latitude,altitude,radius,EphemerisTime,MET,frm,chn,Pulswd,thrsh,gain,1way_range,Emiss,TXmJ,UTC,TOF_ns_ET,Sat_long,Sat_lat,Sat_alt,Offnad,Phase,Sol_inc,SCRNGE,seqid,Product_id
    longField = 'longitude'
    latField = 'latitude'
    elevField = 'altitude'
    radiusField = 'radius'
    utcField = 'UTC'
    orbitField = 'chn'
elif target == "moon":
    targetWKT = "Moon"
    majorRadius = minorRadius = 1737400.0 
    year = 2000
    #From ode Moon run the fields should be:
    #Coordinated_Universal_Time,Pt_Longitude,Pt_Latitude,Pt_Radius,Pt_Range,Pt_PulseW,Pt_Energy,Pt_noi,Pt_Thr,Pt_Gn,Flg,S,Frm,Mission_ET,Subseconds,Terrestrial_Dyn_Time,TX_Energy_mJ,TX_PulseW,SC_Longitude,SC_Latitude,SC_radius,Geoid,Offnadir,Emission,Sol_INC,Sol_Phs,Earth_Centr.,Earth_PW,Earth_E.
    longField = 'Pt_Longitude'
    latField = 'Pt_Latitude'
    elevField = 'Pt_Radius'
    radiusField = 'Pt_Radius'
    utcField = 'Coordinated_Universal_Time'
    orbitField = 'S' #not named correctly in the CSV, should be PRODUCT_SHOT_NUMBER
else:
    print("Error: " + target + " currently not supported.")
    argparse.parser.print_usage()
    sys.exit(1)
    
#based on target and radius write out projection
#
#New prj for GXP
#GEOGCS["GCS_Mars_2000",DATUM["D_Mars_2000",SPHEROID["Mars_2000_IAU_IAG",3396190.0,169.8944472]],PRIMEM["Reference_Meridian",0.0],UNIT["Degree",0.0174532925199433]],VERTCS["Mars_2000",DATUM["D_Mars_2000",SPHEROID["Mars_2000_IAU_IAG",3396190.0,169.8944472]],PARAMETER["Vertical_Shift",0.0],PARAMETER["Direction",1.0],UNIT["Meter",1.0]]
#
if majorRadius - minorRadius > 0.00001:
    ecc = majorRadius / (majorRadius - minorRadius)
else:
    ecc = 0.0
thePrj = 'GEOGCS["GCS_{0}_{1}",DATUM["D_{0}_{1}",SPHEROID["{0}_{1}_IAU",{2:.1f},{3:.14f}]],PRIMEM["Reference_Meridian",0.0],UNIT["Degree",0.0174532925199433]],VERTCS["Mars_2000",DATUM["D_{0}_{1}",SPHEROID["{0}_{1}_IAU",{2:.1f},{3:.14f}]],PARAMETER["Vertical_Shift",0.0],PARAMETER["Direction",1.0],UNIT["Meter",1.0]]' \
             .format(targetWKT,year,majorRadius,ecc)

#loop over files, if the user passed --input then just one file
for input in files:
    filename = os.path.basename(input)
    nameList = os.path.splitext(filename)[0].split("_")
    #VRT didn't like long "name" so, I am creating a shorter name
    if (len(nameList) > 3):
        shortName = nameList[0]+"_"+nameList[1]+"_"+nameList[2]
    else:
        shortName = os.path.splitext(filename)[0]

    #create output shapefile and temporary csv
    outcsv = shortName+"_tmp.csv"
    outvrt = shortName+"_tmp.vrt"
    outprj = shortName+"_Z.prj"
    outshp = shortName+"_Z.shp"
    
    #open ODE csv file for reformating, Lons -180 to 180, and for Mars oc2og lats
    outCSV = open(outcsv,'w')
    fieldnames = 'Longitude,Latitude,Elev_m,Radius_m,UTC,Orbit\n'
    outCSV.write(fieldnames)
    with open(input) as csvfile:
        reader = csv.DictReader(csvfile, skipinitialspace=True)
        header = reader.fieldnames
        for row in reader:
            #convert to -180 to 180 Longitude domain
            lon180 = LonTo180(float(row[longField]))
            latOG = float(row[latField])
            #if Mars convert to ographic Latitudes
            if target == "mars":
                latOG = oc2og(latOG, majorRadius, minorRadius)
                newl = '{0:.5f},{1:.5f},'.format(lon180, latOG)
                newl = newl + row[elevField]+','+row[radiusField]+','+row[utcField]+','+row[orbitField]
            if target == "moon":
                #convert radius from km to meters
                radius = float(row[radiusField]) * 1000.0
                #subtract radius from LOLA radius to get 'elevation' in meters
                elev = radius - majorRadius
                newl = '{0:.5f},{1:.5f},{2:.5f},{3:.2f},'.format(lon180, latOG, elev, radius)
                newl = newl + row[utcField] +','+ row[orbitField]
            if target == "mercury":
                #convert radius from km to meters
                radius = float(row[radiusField]) * 1000.0
                #subtract radius from LOLA radius to get 'elevation' in meters
                #elev = radius - majorRadius
                        #OR
                #convert elevation from km to meters
                elev = float(row[elevField]) * 1000.0
                newl = '{0:.5f},{1:.5f},{2:.5f},{3:.2f},'.format(lon180, latOG, elev, radius)
                newl = newl + row[utcField] +','+ row[orbitField]
            outCSV.write(newl+'\n')
        outCSV.close()
    
    # Create ogr2ogr virtual header (*.vrt)
    outVRT  = open(outvrt , 'w')
    outVRT.write('<OGRVRTDataSource>\n')
    outVRT.write('    <OGRVRTLayer name="'+shortName+'_tmp">\n')
    outVRT.write('        <SrcDataSource>'+outcsv+'</SrcDataSource>\n')
    outVRT.write('        <LayerSRS>'+thePrj+'</LayerSRS>\n')
    outVRT.write('        <GeometryType>wkbPoint</GeometryType>\n')
    outVRT.write('        <GeometryField encoding="PointFromColumns" x="Longitude" y="Latitude" z="Elev_m"/>\n')
    outVRT.write('        <Field name=\"Longitude\" src=\"Longitude\" type=\"Real\"/>\n')
    outVRT.write('        <Field name=\"Latitude\" src=\"Latitude\" type=\"Real\"/>\n')
    outVRT.write('        <Field name=\"Elev_m\" src=\"Elev_m\" type=\"Real\"/>\n')
    outVRT.write('        <Field name=\"Radius_m\" src=\"Radius_m\" type=\"Real\"/>\n')
    outVRT.write('        <Field name=\"UTC\" src=\"UTC\" type=\"String\"/>\n')
    outVRT.write('        <Field name=\"Orbit\" src=\"Orbit\" type=\"Integer\"/>\n')
    outVRT.write('    </OGRVRTLayer>\n')
    outVRT.write('</OGRVRTDataSource>\n')
    outVRT.close()
    
    #convert to shapefile using GDAL's ogr
    in_ds = ogr.Open(outvrt)
    ogr.GetDriverByName("ESRI Shapefile").CopyDataSource(in_ds, outshp)
    in_ds = None

    outPRJ  = open(outprj , 'w')
    outPRJ.write(thePrj)
    outPRJ.close()
    
    #clean up temporary files
    if os.path.exists(outshp):
        print (" -deleting temporary csv and vrt files")
        os.remove(outcsv)
        os.remove(outvrt)
        print (" -output shapefile file generated: "+ outshp)
    else: 
        print ("\n Shapefile not generated...something's wrong\n\n")
     