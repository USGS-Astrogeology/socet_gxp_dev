#!/usr/bin/perl -s

################################################################################
# NOTICE NOTICE NOTICE NOTICE NOTICE NOTICE NOTICE NOTICE NOTICE NOTICE NOTICE
#
#      This script is not supported by ISIS.
#      If you have problems please contact the Astrogeology Photogrammetry group
#      at PlanetaryPhotogrammetry@usgs.gov
#
# NOTICE NOTICE NOTICE NOTICE NOTICE NOTICE NOTICE NOTICE NOTICE NOTICE NOTICE
################################################################################

use File::Copy;
use File::Basename;
use Cwd;

my ($progname) = ($0 =~ m#([^/]+)$#);  # get the name of this program

#--------------------------------------------------------------------
# Location-dependent paths, fix for your system:

#Location of Socet Set Utility programs
$SS_utilities_path = "/home/ahowington/SOCET_UTILITIES/FOR_STEREO_PIPELINE";

# End of Location-dependent paths
################################################################################

$email = "PlanetaryPhotogrammetry\@usgs.gov";

my $usage = "

Command:  $progname Reference_pc Reference_pc_format SS_DTM_ASCII SS_gpf tfm_SS_gpf {sphere}

Where:
       Reference_pc        = Refererence (fixed) point cloud
       Reference_pc_format = table or ascii_dtm
       SS_DTM_Ascii        = Socet Set DTM in ASCII format 
                             (i.e., the source (movable) point cloud)
       SS_gpf              = Socet Set ground point file corresponding
                             to SS_DTM_Ascii
       tfm_SS_gpf          = output transformed SS ground point file
                             *must* have .gpf extension 
       sphere              = optional parameter. if MOLA tracks
                             are based on a spheroid (e.g. HRSC) 

Description:

       $progname will run the Ames Stereo Pipelline tool, pc_align, to
       surface fit a relatively oriented Socet Set DTM (i.e, a source
       (movable) point clould) to either MOLA tracks or an absolutely
       controled Socet Set DTM (i.e, a Reference (fixed) point cloud).
       Point Cloud

       For details on pc_align, see section A.12 in the the Ames
       Stereo Pipeline manual at:
       http://byss.arc.nasa.gov/stereopipeline/binaries/asp_book-2.4.0.pdf 
       
       $progname expects as input:

       Reference_pc        = The Refererence (fixed) point cloud. This is
                             either
                              (1) the MOLA table (.tab) file generated
                                  by hidata4socet, or
                              (2) an absolutely controlled Socet Set DTM
                                  in the Socet Set ASCII format

       Reference_pc_format = A flag indicating the format of the Reference_pc.
                             Allowable options are:
                              (1) table
                              (2) ascii_dtm

       SS_DTM_Ascii        = The relatively oriented Socet Set DTM in ASCII format 
                             (i.e., the source (movable) point cloud)

       SS_gpf              = The Socet Set ground point file corresponding
                             to SS_DTM_Ascii.  All *tie* points in the gpf
                             will be transformed by pc_align.

       tfm_SS_gpf          = The output transformed/aligned Socet Set
                             ground point file.  Note:
                              (1)   You must include the .gpf extension
                              (2) All tranformed tie points will
                                  be set as XYZ control with default
                                  accuracy values of 1.0 1.0 1.0
                              (3) Any non-tie points from the input SS_gpf file
                                  will be set as tie points in the output
                                  tfm_SS_gpf.

       sphere             = optional parameter. Add sphere only if MOLA tracks 
                            are based on a spheroid (e.g. HRSC). This grabs 
                            column 5 from the MOLA tab file (which should be
                            planet_rad field) instead of column 3 (topography 
                            field).


       A report of errors encountered in the processing goes to file:
       \"surfaceFitPcAlign.err\".

       Note that any errors will cause this script to abort.


**************************************************************************
**************************************************************************
NOTICE:
       This script is not supported by ISIS.
       If you have problems please contact the Astrogeology Photogrammetry group
       at $email
**************************************************************************
**************************************************************************
";

#####################################################################
#  MAIN APPLICATION SECTION
#  Author: Elpitha Howington-Kraus
#  Date:   February 05 2013
#  Version: 1.1
#  History: JUN 17 2014 - E Howington-Kraus, USGS, Flagstaff Original Version
#  Sep 28, 2016 - Trent Hare, added Sphere option (for MOLA sphere TABs)
#
#####################################################################


#---------------------------------------------------------------------
# Check the argument list
#---------------------------------------------------------------------
   if ($#ARGV < 5)
      {
      print "\n\nRun $progname as follows:";
      print "$usage\n";
      exit 1;
      }

#---------------------------------------------------------------------
# Obtain the input parameters
#---------------------------------------------------------------------

   $referencePC = $ARGV[0];
   $referenceFormat = $ARGV[1];
   $ssAsciiDtm = $ARGV[2];
   $ssGpf = $ARGV[3];
   $tfmSsGpf = $ARGV[4];
   if ($#ARGV > 4)
   {
     $sphere = $ARGV[5];
   }
 
   print "in: $#ARGV $sphere\n";

#---------------------------------------------------------------------
# If "surfaceFitPcAlign.err" file exist, deleteit 
#---------------------------------------------------------------------

   if (-e "surfaceFitPcAlign.err") {unlink("surfaceFitPcAlign.err");}

#---------------------------------------------------------------------
# Make sure input files exist
#---------------------------------------------------------------------

   if (!(-e $referencePC))
      {
      print "*** ERROR *** Input Reference Point Cloud file does not exist: $referencePC\n";
      print "$progname terminating...\n";
      exit 1;
      }

   if (!(-e $ssAsciiDtm))
      {
      print "*** ERROR *** Input SS ASCII DTM does not exist: $ssAsciiDtm\n";
      print "$progname terminating...\n";
      exit 1;
      }

   if (!(-e $ssGpf))
      {
      print "*** ERROR *** Input Socet Set Ground Point file does not exist: $ssGpf\n";
      print "$progname terminating...\n";
      exit 1;
      }


#---------------------------------------------------------------------
# Make sure allowable options for referenceFormat were entered
#---------------------------------------------------------------------

   if ( $referenceFormat ne "table" && $referenceFormat ne "ascii_dtm")
      {
      print "*** ERROR *** invalid option for Reference_pc_format entered: $referenceFormat\n";
      print "              Allowable options are:\n";
      print "                 table\n";
      print "                 ascii_dtm\n";
      print "$progname terminating...\n";
      exit 1;
      }
#---------------------------------------------------------------------
# Convert the input Socet Set ground point file to UNIX format
#---------------------------------------------------------------------

  $cmd = "dos2unix $referencePC";
  system($cmd) == 0 || ReportErrAndDie ("dos2unix failed on command:\n$cmd");
 
  $cmd = "dos2unix $ssGpf";
  system($cmd) == 0 || ReportErrAndDie ("dos2unix failed on command:\n$cmd");
 
#---------------------------------------------------------------------
# Open LOG file
#---------------------------------------------------------------------

   $log = "surfaceFitPcAlign.err";
   open (LOG,">$log") or die "\n Cannot open $log\n";

#---------------------------------------------------------------------
# Strip header from Reference PC file and output Decimal Degrees,
# LatLonH, space delimited 360sys CSV...if the CSV file does not
# already exist
#---------------------------------------------------------------------

  print("\nGenerating Reference Point Cloud CSV file...\n");

  # Generate output CSV file name, and if the file does not already
  # exist, generate the CSV file
  $firstdot = index($referencePC,".");
  $coreName = substr($referencePC,0,$firstdot);
  #$referencePCCsv = $coreName . "_RefPC_ogLat_360ELon_H.csv";
  $referencePCCsv = $coreName . "_RefPC.csv";
   
  if (-e $referencePCCsv) {unlink ($referencePCCsv); print ("file removed: $referencePCCsv \n");}

  if (!(-e $referencePCCsv)) {

    if ( $referenceFormat eq "table")
    {
      # Get number of lines in table file
      $lineCount = `wc $referencePC | awk '{print \$1}'`;
      chomp ($lineCount);

      # Subtract two lines for the header (we will skip it for the CSV file)
      $lineCount = $lineCount - 2;

      # Create CSV file
      if ( $sphere eq "sphere")
      {
        print("...set to sphere...\n");
        print("output file: $referencePCCsv \n");
        $lineCount = $lineCount - 1;
        #$cmd = "tail \-$lineCount $referencePC | awk -F' ' '{print \$10,\$1,\$5-3396000}' > $referencePCCsv";
        $cmd = "tail \-$lineCount $referencePC | awk '{printf\"%s, %s, %.8f\\n\",\$10,\$1,\$5-3396000}' > $referencePCCsv";
        print("$cmd \n");
      } 
      else 
      {
        $cmd = "tail \-$lineCount $referencePC | awk '{print \$10,\$1,\$3}' > $referencePCCsv";
      }
      system($cmd) == 0 || ReportErrAndDie ("Failed on command:\n$cmd");
    }
    else
    {
      # Get number of lines in ascii DTM file
      $lineCount = `wc $referencePC | awk '{print \$1}'`;
      chomp ($lineCount);

      # Subtract 14 lines for the header (we will skip it for the CSV file)
      $lineCount = $lineCount - 14;

      # Create CSV file
      #$cmd = "tail \-$lineCount $referencePC | awk '{printf\"%s, %.10f, %s\\n\",\$2,\$1+360,\$3}' > $referencePCCsv";
      $cmd = "tail \-$lineCount $referencePC | awk '{printf\"%s, %.10f, %s\\n\",\$2,\$1,\$3}' > $referencePCCsv";
      system($cmd) == 0 || ReportErrAndDie ("Failed on command:\n$cmd");
    }
  }

#---------------------------------------------------------------------
# Strip Header from SS Ascii DTM Decimal Degrees, output Decimal Degrees,
# LatLonH, space delimited 360sys CSV
#---------------------------------------------------------------------

  print("\nGenerating Source Point Cloud CSV file...\n");

  # Generate output CSV file name, and if the file already
  # exists, delete it
  $firstdot = index($ssAsciiDtm,".");
  $coreName = substr($ssAsciiDtm,0,$firstdot);
  #$ssDtmCsv = $coreName . "_DTM_ogLat_360ELon_H.csv";
  $ssDtmCsv = $coreName . "_DTM.csv";

  if (-e $ssDtmCsv) {unlink ($ssDtmCsv)}
  
  # Get number of lines in ascii DTM file
  $lineCount = `wc $ssAsciiDtm | awk '{print \$1}'`;
  chomp ($lineCount);

  # Subtract 14 lines for the header (we will skip it for the CSV file)
  $lineCount = $lineCount - 14;

  # Create CSV file
  $cmd = "tail \-$lineCount $ssAsciiDtm | awk '{printf\"%s, %.10f, %s\\n\",\$2,\$1+360,\$3}' > $ssDtmCsv";
  #$cmd = "tail \-$lineCount $ssAsciiDtm | awk '{printf\"%s, %.10f, %s\\n\",\$2,\$1,\$3}' > $ssDtmCsv";
  system($cmd) == 0 || ReportErrAndDie ("Failed on command:\n$cmd");

#---------------------------------------------------------------------
# Convert GPF Tiepoint to 360sys csv by running gpfTies2LatLonHeightCSV_360sys 
# gpfTies2LatLonHeightCSV_360sys will output the following two files:
# 1) The Tie Point coordinates, with the same core name  of the input file,
#    and a .csv extension
# 2) The Tie Point IDs, with the same core name  of the input file,
#    and a .tiePointIds.txt extension.
#---------------------------------------------------------------------

  print("\nGenerating Tie Points CSV file...\n");

  # Generate output CSV file name,
  $firstdot = index($ssGpf,".");
  $coreName = substr($ssGpf,0,$firstdot);
  $ssGpfTiesCsv = $coreName . ".csv";
  
  # Create CSV file
  $cmd = "$SS_utilities_path/gpfTies2LatLonHeightCSV_360sys $ssGpf";
  system($cmd) == 0 || ReportErrAndDie ("Failed on command:\n$cmd");

#---------------------------------------------------------------------
# For now append the GPF points to the DEM to carry them through the
# surface fitting.
#---------------------------------------------------------------------

  print("\nAppending Tie Points CSV file to Reference PC CSV files...\n");

  # Generate output CSV file name,
  $firstdot = index($ssAsciiDtm,".");
  $coreName = substr($ssAsciiDtm,0,$firstdot);
  #$ssDtmGpfCsv = $coreName . "_DTM_gpfTies_ogLat_360ELon_H.csv";
  $ssDtmGpfCsv = $coreName . "_DTM_gpfTies.csv";
  
  # Create CSV file
  $cmd = "cat $ssDtmCsv $ssGpfTiesCsv > $ssDtmGpfCsv";
  system($cmd) == 0 || ReportErrAndDie ("Failed on command:\n$cmd");

#---------------------------------------------------------------------
# Run PC_ALIGN
#---------------------------------------------------------------------

  print("\nRunning pc_align...\n");

  # Generate output CSV file name,
  $firstdot = index($ssAsciiDtm,".");
  $coreName = substr($ssAsciiDtm,0,$firstdot);
  #$pcAlignedCoreName = $coreName . "_pcALigned_DTM_gpfTies_ogLat_360ELon_H";
  $pcAlignedCoreName = $coreName . "_pcALigned_DTM_gpfTies";
  
  # Create CSV file
  $cmd = "pc_align --max-displacement 300 --datum D_MARS -o $pcAlignedCoreName --save-inv-transformed-reference-points $ssDtmGpfCsv $referencePCCsv";
  system($cmd) == 0 || ReportErrAndDie ("Failed on command:\n$cmd");

#---------------------------------------------------------------------
# Extract transformed GPF coordinates from pc_align results and
# generate transformed gpf file
#---------------------------------------------------------------------

  print("\nGenerating transformed Socet Set ground point file...\n");

  # Get the number of tie points
  $lineCount = `wc $ssGpfTiesCsv | awk '{print \$1}'`;
  chomp ($lineCount);

  # Extract the transformed coordinates from the transformed CSV
  $tfmPcAlignedCsv = $pcAlignedCoreName ."-trans_reference.csv";
  $tfmCoordinates = "tfm_tiePoint_coordinates.csv";

  if (-e $tfmCoordinates) {unlink $tfmCoordinates;}
  $cmd = "tail \-$lineCount $tfmPcAlignedCsv > $tfmCoordinates";
  system($cmd) == 0 || ReportErrAndDie ("Failed on command:\n$cmd");

  # Generate transformed gpf file
  $cmd = "$SS_utilities_path/mergeTransformedGPFties $ssGpf $tfmCoordinates $tfmSsGpf";
  system($cmd) == 0 || ReportErrAndDie ("Failed on command:\n$cmd");

  print("\nDone\n");

#---------------------------------------------------------------------
# Close the LOG file.
# If an error was detected, print out the log file
#---------------------------------------------------------------------

   close (LOG);

   @lines = `cat $log`;
   if (scalar(@lines) > 0)
      {
      print "\n*** Errors detected in processing ***\n\n";
      print @lines;
      print "\n";
      print "\n*** See surfaceFitPcAlign.err for details ***\n\n";
      }
   else
      {
      unlink ($log);
      }

   exit;

##############################################################################
#  Error Handling Subroutine
##############################################################################
sub ReportErrAndDie
    {
    my $ERROR=shift;

    print "$ERROR\n";
    print "$progname aborted\n";

    print LOG "$ERROR\n";
    close(LOG);

    exit 1;

    }

