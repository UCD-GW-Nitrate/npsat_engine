\page inputfiles Input files
\section Introduction
The NPSAT engine uses one parameter file that follows the 
deal.ii input file structure. 
The file is structured into sections and subsection. 
The following paragraphs give information for each section of the main parameter file.

\section mpf Main Parameter file
The main parameter file is split into a number of sections where each section may have one or more 
subsections with associated parameters. 
You can generate a template parameter file bu running the program without input arguments

```
npsat
```

\subsection wd Workspace directories
* Input directory : This is the directory where all the input files are expected to be.
* Output directory : This is the directory where all the output files will be written.

\subsection gopt Geometry options
* Geometry Type : The geometry of the domain can be either a Box or a file. 

	(These should be capitalized BOX or FILE)
	- BOX : When the domain is Box the following parameters in the subsection Box have to be set
	- FILE :






