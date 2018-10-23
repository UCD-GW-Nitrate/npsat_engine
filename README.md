
<img src="Logo/logo_npsat_600.png" alt="Wells in Central Valley" width="700"/>


## Overview
Non Point Source Assessment Tool




## Dependencies
- [Deal.ii 9](https://www.dealii.org/) 
Deal library has quite a few options on how to compile. The easiest seems to be the ([Candii](https://github.com/koecher/candi)) distribution which involves just one line:

    
    ```
    ./candi.sh -j 4 --prefix=Path/to/candi/compiled/libs
    ```

    (If you dont have more than 16 GB ram then use `-j 2` or even without `j`). 

- [CGAL](https://www.cgal.org/).
It is highly recommended to use version [4.11.3](https://github.com/CGAL/cgal/releases/tag/releases%2FCGAL-4.11.3)  as earlier or later compiled versions have failed to be compiled together with NPSAT. 
If you already have a CGAL installation from sources (e.g.`sudo apt-get install libcgal-dev`) then it may work even if the installed version is not the specific one. 
To find out about the compiled version compile and run this [program](https://gist.github.com/alecsphys/7398446).

If you follow the installation guide from the library then you will do the following
```
cd /path/to/cgal-releases-CGAL-4.11.3
mkdir -p build/release
cmake -DCMAKE_BUILD_TYPE=Release ../..
make
```


## Build NPSAT
To compile the NPSAT code run the following command from the directory where the npsat.cc file is.
```
cmake -DDEAL_II_DIR=Path/to/candi/compiled/libs/deal.II-v9.0.0 -DCGAL_DIR:PATH=/path/to/cgal-releases-CGAL-4.11.3/build/release .
make
```


## Run
#### Calculate flow & particle tracking
To run NPSAT you need to prepare a main parameter file. then you can do
```
mpirun -n nproc path/to/executable/npsat -p parameter_file.npsat
```
This is going to execute the problem described in the parameter file. using `nproc` processors. If the particle tracking is on, then most likely the particle trajectories have been written in several files per processor, where each file contains segments of the particle trajectories. To do anything useful with them you need first to gather them. You can do so by running the following
```
path/to/executable/npsat -p parameter_file.npsat -g nproc nchunk
```
where `nproc` is the number of processors that where used during the simulation and `nchunk` is a number that is specified at the end of the simulation. 

#### Compute URFs
This gather step is going to generate one or more files with the suffix *.urfs. This contains the data in a suitable format for Unit Response Function calculation. 

From matlab or octave simply call 
```
WellURF = readURFs(filename);
```

Keep in mind that this is going to look for some [msim](http://subsurface.gr/software/msim/) commands. Therefore you need to make sure that the msin folders have been added to matlab search path.

That concludes the workflow of this tool.

For using the URFs see [Mantis](https://github.com/giorgk/Mantis) 


Happy pollution predicting!





