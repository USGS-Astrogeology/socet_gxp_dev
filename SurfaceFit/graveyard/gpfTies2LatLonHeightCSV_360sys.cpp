#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <ctype.h>
#include <stdlib.h>

#define LINELENGTH 200
#define FILELEN 512
#define MAXFILES 50
#define MAXPOINTS 2000

main(int argc, char *argv[])
{

  char     gpfFile[FILELEN];
  char     csvFile[FILELEN];
  char     pointIDsFile[FILELEN];

  FILE     *gpfFp;     // file pointer to input csv file
  FILE     *csvFp;     // file pointer to output csv file
  FILE     *ptsFp;     // file pointer to output point ids list file

  char     gpfLine[LINELENGTH];

  // check number of command line args and issue help if needed
  //-----------------------------------------------------------
  if (argc != 2) {
     printf ("\nrun %s as follows:\n",argv[0]);
     printf ("   %s SSgpfFile\n",
             argv[0]);
     printf ("\nwhere:\n");
     printf ("  SSgpfFile = Socet Set *.gpf file, from a geographic project\n\n");
     printf ("  This program will convert a Socet Set ground point file into a CSV\n");
     printf ("  of lat,lon,height.  The output file will have the same core name\n");
     printf ("  of the input *.gpf file, but with a .csv extension\n\n");
     printf ("  Also output is the list of point IDs that were converted.  This file\n");
     printf ("  will be used to port the points back to Socet Set later on.  The output\n");
     printf ("  file will have the same core name as the input file, but with a .pointids\n");
     printf ("  .tiePointIds.txt extension.\n");
     exit(1);
  }

  //------------------------------------------------
  // get input arguments entered at the command line
  //------------------------------------------------

  strcpy (gpfFile,argv[1]);

  //-----------------------------
  // generate ouput file names
  //-----------------------------

  char corename[FILELEN];
  strcpy (corename,gpfFile);
  int len = strlen(corename);
  corename[len-4] = '\0';

  strcpy (csvFile,corename);
  strcat (csvFile,".csv");

  strcpy (pointIDsFile,corename);
  strcat (pointIDsFile,".tiePointIds.txt");

  /////////////////////////////////////////////////////////////////////////////
  // open files 
  /////////////////////////////////////////////////////////////////////////////

  gpfFp = fopen (gpfFile,"r");
  if (gpfFp == NULL) {
    printf ("unable to open input gpf file: %s\n",gpfFile);
    exit (1);
  }

  csvFp = fopen (csvFile,"w");
  if (csvFp == NULL) {
    printf ("unable to open output csv file: %s\n",csvFile);
    fclose(gpfFp);
    exit (1);
  }

  ptsFp = fopen (pointIDsFile,"w");
  if (csvFp == NULL) {
    printf ("unable to open output list file of tie point ids: %s\n",pointIDsFile);
    fclose(gpfFp);
    fclose(csvFp);
    exit (1);
  }

  //------------------------------------------------
  //------------------------------------------------

  // skip the header
  fgets(gpfLine,LINELENGTH,gpfFp);

  // read in number of points in *.gpf file
  char value[50]; 
  fgets(gpfLine,LINELENGTH,gpfFp);
  sscanf (gpfLine,"%s",value);
  int numpts = atoi(value);

  // skip next header
  fgets(gpfLine,LINELENGTH,gpfFp);

  // Parse gpf, output csv
  char pointID[50], valStat[2], valKnown[2]; 
  char valLon[50], valLat[50], Height[50];
  double rad2dd = 57.295779513082320876798154814105;

  for (int i=0; i<numpts; i++) {
    fgets(gpfLine,LINELENGTH,gpfFp);
    sscanf (gpfLine,"%s %s %s",pointID,valStat,valKnown);
    int stat = atoi(valStat);
    int known = atoi(valKnown);

    // get coordinate
    fgets(gpfLine,LINELENGTH,gpfFp);

    //only output tie points that are on
    if (stat == 1 && known == 0) {
      sscanf (gpfLine,"%s %s %s",valLat,valLon,Height);
      double radLat = atof(valLat);
      double radLon = atof(valLon);
      if(radLon < 0.0)
        fprintf(csvFp,"%.14lf,%.14lf,%s\n",rad2dd*radLat,rad2dd*radLon+360.0,Height);
      else
        fprintf(csvFp,"%.14lf,%.14lf,%s\n",rad2dd*radLat,rad2dd*radLon,Height);
      fprintf(ptsFp,"%s\n",pointID);
    }

    // skip next three lines in input *.gpf file
    for (int j=0; j<3; j++)
      fgets(gpfLine,LINELENGTH,gpfFp);
  }

  fclose(gpfFp);
  fclose(csvFp);

} // end of program

