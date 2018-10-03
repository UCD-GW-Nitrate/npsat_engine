# NPSAT
Non Point Source Assessment Tool

This tool uses Adaptive Mesh Refinement


# Dependencies
- Deal.ii 9 ([Candii](https://github.com/koecher/candi))

    Usually I compile the library with the following command:
    ```
    ./candi.sh -j 4 --prefix=Path/to/candi/compiled/libs
    ```
    (If you dont have more than 16 GB ram then use `-j 2` or even without `j`) 

- CGAL version [4.11.3](https://github.com/CGAL/cgal/releases/tag/releases%2FCGAL-4.11.3) (Earlier or later self-compiled versions have failed to be compiled with NPSAT. If you already have a CGAL installation from sources (e.g.`sudo apt-get install libcgal-dev`) then this may work. 
To find out about the compiled version compile and run this [program](https://gist.github.com/alecsphys/7398446).

If you follow the installation guide from the library then you will do the following
```
cd /path/to/cgal-releases-CGAL-4.11.3
mkdir -p build/release
cmake -DCMAKE_BUILD_TYPE=Release ../..
make
```

To compile the NPSAT code run the following command from the directory where the npsat.cc file is.
```
cmake -DDEAL_II_DIR=Path/to/candi/compiled/libs/deal.II-v9.0.0 -DCGAL_DIR:PATH=/path/to/cgal-releases-CGAL-4.11.3/build/release .
make
```
