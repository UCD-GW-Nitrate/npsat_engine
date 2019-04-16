
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
    
    _Changes on Candi distribution_
    The last time we tried to build deal.ii using hte candii distribution we had to make a number of changes to make it work. Under the candi/deal.II-toolchain/packages folder there is a list of packages that containts the version numbers among other information. 
    Do the following changes:
    - hdf5 : MAJORVER=1.10, CHECKSUM=e115eeb66e944fa7814482415dd21cc4
    - petsc : VERSION=3.10.4 CHECKSUM=06205518845df58771d579eab4e4e01b
    - slepc : VERSION=3.10.2 CHECKSUM=92d4eef844b11f5ff673959f6f00e1cf
    

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

Although the simulation of non point source pollution is a 3D problem the code is designed to run 2D problems, where only a cross section of the domain is considered. 
The NPSAT code can be compiled to run either 2D or 3D problem. 
Inside the headers folder there is a header file *my_macros.h* which includes the definition of the dimension. Change this to 2 or 3 according to the problem being solved

```
 #define _DIM 2
```


#### Calculate flow & particle tracking
To run NPSAT you need to prepare a main parameter file. Then you can do:
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

## Run NPSAT in cluster
Both dependencies of NPSAT can be compiled and installed locally. Therefore to compile the code in a cluster should be similar to a desktop. If the cluster uses [SLURM](https://slurm.schedmd.com/) for managing the workload then you can submit jobs to the cluster using the following guide:

* Create a file for example *run_job.sbatch* with content similar to the following:
```
#!/bin/bash
#
# job name:
#SBATCH --job-name='CVHM'
#
# Number of tasks (cores):
#
#SBATCH --ntasks=64
#
#SBATCH --output=out%j.log
#SBATCH --error=out%j.err
#
#####SBATCH --dependency=singleton
#
# Load your modules
module purge
module load system-gcc/openmpi-2.1.5
#
# Set up your environment 
cd /path/to/the/parameter/file
#
# Start your MPI job
mpirun /path/to/executable/npsat -p parameter_file.npsat

```
 
* Most of the options are quite self explainatory. 
    * **job-name** is just a text name to identify the job when for example use the `squeue -u username` to see the progress of the jobs. 
    * **ntasks** is similar to `nproc`. defines the number o processors that are requested to run the job.
    *  **output** is the file where the output of the programm will be printed. Note that the format out%j.log will write the output of the program to the file out####.log where #### is a unique id for the submitted job
    * **error** Possible errors during the run will be reported here.
    * There are many more options one can define. These is just a minimum list of options.
    
* Next we define the required modules. Note that the options here depended highly on the cluster.

* Setting up the environment simply amounts to navigating to the folder where the files are 

* Finally we start the parallel process using a command similar to the one was used in the desktop. However we omit the `-n` option.
* Run the job:
```
sbatch run_job.sbatch
```


---
### Usefull commands
* **Slurm common commands**
When a job is running on the cluster use `squeue -u username` to see the status of the job. 
If for any reason you need to kill the job, find the job id by running the squeue and then ```scancel jobid```


* **Transfer results from cluster**
To transfer the result from the cluster to local machine the admins suggest the rsync over scp. For example to tranfer the *urfs files on the current folder do:
```
rsync -azvhe ssh name@aqua.lawr.ucdavis.edu:/path/of/the/results/*.urfs .
```
or for the farm cluster
```
rsync -e "ssh -p 2022" --archive name@farm.cse.ucdavis.edu:/path/of/the/results/*.urfs .
```
* **Delete many files**
The simulation depending the number of processors and the number of particle tracking may create a large number of files. The command `rm -r` to delete all files let's say from the output folder may fail.
An alternative command is ([see details](https://stackoverflow.com/questions/14731133/how-to-delete-all-files-older-than-3-days-when-argument-list-too-long?rq=1)):
```
find . -mtime +3 | xargs rm -Rf
```
It is also usefull to find the number of files in a directory. The following does that ([see more](https://askubuntu.com/questions/370697/how-to-count-number-of-files-in-a-directory-but-not-recursively)):
```
find -maxdepth 1 -type f | wc -l
```

    
    
    
    





