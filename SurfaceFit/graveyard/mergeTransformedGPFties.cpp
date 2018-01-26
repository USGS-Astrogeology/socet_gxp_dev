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

  char     origGPFFile[FILELEN];
  char     tfmGPFFile[FILELEN];
  char     tfmCSVFile[FILELEN];

  FILE     *origgpfFp;  // file pointer to input gpf prior to transformation
  FILE     *tfmcsvFp;  // file pointer to input csv file of transformed ground coordiantes
  FILE     *tfmgpfFp;  // file pointer to output gpf with tranformed coordiantes

  char     csvLine[LINELENGTH];
  char     gpfLine[LINELENGTH];

  // check number of command line args and issue help if needed
  //-----------------------------------------------------------
  if (argc != 4) {
     printf ("\nrun %s as follows:\n",argv[0]);
     printf ("   %s origGPF tfmCSV tfmGPF\n",
             argv[0]);
     printf ("\nwhere:\n");
     printf ("  origGPF = Socet Set *.gpf file for a geographic project, prior to running pc_align\n\n");
     printf ("  tfmCSV = tranformed *.csv file of original tie ground point coordinates generaged by pc_align\n\n");
     printf ("  tfmGPF = Socet Set *.gpf containing transformed ground control\n\n");
     printf ("  transformed points will be set to XYZ control with default sigmas of 1.0 1.0 1.0\n");
     printf ("  Preexisting ground control in the origGPF will be set to tie points\n");
     exit(1);
  }

  //------------------------------------------------
  // get input arguments entered at the command line
  //------------------------------------------------

  strcpy (origGPFFile,argv[1]);
  strcpy (tfmCSVFile,argv[2]);
  strcpy (tfmGPFFile,argv[3]);

  /////////////////////////////////////////////////////////////////////////////
  // open files 
  /////////////////////////////////////////////////////////////////////////////

  origgpfFp = fopen (origGPFFile,"r");
  if (origgpfFp == NULL) {
    printf ("unable to open original input gpf file: %s\n",origGPFFile);
    exit (1);
  }

  tfmcsvFp = fopen (tfmCSVFile,"r");
  if (tfmcsvFp == NULL) {
    printf ("unable to open input transformed csv file: %s\n",tfmCSVFile);
    fclose(origgpfFp);
    exit (1);
  }

  tfmgpfFp = fopen (tfmGPFFile,"w");
  if (tfmgpfFp == NULL) {
    printf ("unable to open output transformed ground point file: %s\n",tfmGPFFile);
    fclose(origgpfFp);
    fclose(tfmcsvFp);
    exit (1);
  }

  //------------------------------------------------
  // Copy the header of the original gpf to the
  // tranformed gpf, and record the number of points
  //------------------------------------------------

  // output first header line to transformed gpf
  fgets(gpfLine,LINELENGTH,origgpfFp);
  fputs(gpfLine,tfmgpfFp);

  // read in number of points in the orginal *.gpf file
  // and output to transformed gpf
  char value[50]; 
  fgets(gpfLine,LINELENGTH,origgpfFp);
  sscanf (gpfLine,"%s",value);
  int numpts = atoi(value);
  fputs(gpfLine,tfmgpfFp);

  // ouput third head line to transformed gpf
  fgets(gpfLine,LINELENGTH,origgpfFp);
  fputs(gpfLine,tfmgpfFp);

  // Parse original gpf & tfm csv, and output tfm gpf
  char pointID[50], valStat[2], valKnown[2]; 
  char valLon360[50], valLat[50], Height[50];
  double rad2dd = 57.295779513082320876798154814105;

  for (int i=0; i<numpts; i++) {
    fgets(gpfLine,LINELENGTH,origgpfFp);
    sscanf (gpfLine,"%s %s %s",pointID,valStat,valKnown);
    int stat = atoi(valStat);
    int known = atoi(valKnown);

    if (known > 0) {
      // Change a non-tie point to tie point in the tfm GPF
      // regardless if it was used or not
      fprintf(tfmgpfFp,"%s %d 0\n",pointID,stat);
      for (int j=0; j<4; j++) {
        fgets(gpfLine,LINELENGTH,origgpfFp);
        fputs(gpfLine,tfmgpfFp);
      }
    }

    if (stat == 0 & known == 0) {
      // This is a tie point that was off in the original GPF
      // so just copy it into the trm GPF
      fputs(gpfLine,tfmgpfFp);
      for (int j=0; j<4; j++) {
        fgets(gpfLine,LINELENGTH,origgpfFp);
        fputs(gpfLine,tfmgpfFp);
      }
    }

    if (stat == 1 && known == 0) {
      // This point was transformed.  Make it an XYZ control point
      // in the GPF file, and get the coordinate from the tfm CSV
      // file

      fprintf(tfmgpfFp,"%s %d 3\n",pointID,stat);

      // get transformed coordinate string and replace commas with spaces in csvLine
      fgets(csvLine,LINELENGTH,tfmcsvFp);
      int len = strlen(csvLine);
      for (int j=0; j<len; j++)
        if(csvLine[j] == ',') csvLine[j] = ' ';

      // parse transformed coordinate, convert 360 lon domain to 180 lon domain
      char valLon360[50], valLat[50], Height[50];
      double rad2dd = 57.295779513082320876798154814105;

      sscanf (csvLine,"%s %s %s",valLat,valLon360,Height);
      double radLat = atof(valLat) / rad2dd;
      double ddLon360 = atof(valLon360);
      double radLon180;
      if (ddLon360 > 180)
        radLon180 = (ddLon360-360) / rad2dd;
      else
        radLon180 = ddLon360 / rad2dd;

      // output coordinate, weights and residuals to tfm GPF
      fprintf(tfmgpfFp,"%.14lf    %.14lf    %s\n",radLat,radLon180,Height);
      fprintf(tfmgpfFp,"1.0 1.0 1.0\n");
      fprintf(tfmgpfFp,"0.0 0.0 0.0\n\n");

      // skip the next four lines in the original gpf
      for (int j=0; j<4; j++) 
        fgets(gpfLine,LINELENGTH,origgpfFp);

    } //end if (stat == 1 && known == 0)

  } // end for (int i=0; i<numpts; i++)

  fclose(origgpfFp);
  fclose(tfmcsvFp);
  fclose(tfmgpfFp);

} // end of program

