#ifndef BOUNDARYINTERP_H
#define BOUNDARYINTERP_H

#include <fstream>

#include <deal.II/base/point.h>

#include "helper_functions.h"

using namespace dealii;

template <int dim>
class BoundaryInterp{
public:
    //! An empty constructor
    BoundaryInterp();

    //! returns the interpolated value
    double interpolate(Point<dim> p);

    //! read data from file
    void get_data(std::string filename);

private:

    //! This is a vector with the corner points of the boundary polygon
    std::vector<Point<dim-1>> Pnts;
    std::vector<std::vector<double>> Values;
    std::vector<std::vector<double>> Elevations;
    std::vector<double> Length;

    //! Number of corner points
    unsigned int Npnts;

    //! Number of data along Z
    unsigned int Ndata;

    //! Points with distance to a boundary segment closer that the tolerance
    //! will be considered as part of the boundary
    double tolerance;

    bool isPoint_onSeg(Point<dim> p, int iSeg, double& dst_t);

    bool isPoint_onBoundary(Point<dim> p);
};


template <int dim>
BoundaryInterp<dim>::BoundaryInterp(){
    Npnts = 0;
    Ndata = 0;
}

template <int dim>
void BoundaryInterp<dim>::get_data(std::string filename){
    std::ifstream  datafile(filename.c_str());
    if (!datafile.good()){
        std::cerr << "Can't open " << filename << std::endl;
        return;
    }
    else{
        char buffer[512];

        {// Read the data type
            datafile.getline(buffer, 512);
            std::istringstream inp(buffer);
            std::string temp;
            inp >> temp;
            if (temp != "BOUNDARY_LINE"){
                std::cerr << " ScatterInterp Cannot read " << temp << " data." << std::endl;
                return;
            }
        }

        {// Read the number of data and allocate space
            datafile.getline(buffer, 512);
            std::istringstream inp(buffer);
            inp >> Npnts;
            inp >> Ndata;
            Pnts.resize(Npnts);
            Values.resize(Npnts);
            Length.resize(Npnts);
            if (Ndata > 1)
                Elevations.resize(Npnts);
        }

        {// Read the data
            double val;
            for (unsigned int i = 0; i < Npnts; ++i){
                datafile.getline(buffer, 512);
                std::istringstream inp(buffer);

                for (unsigned int idim = 0; idim < dim-1; idim++){
                    inp >> val;
                    Pnts[i][idim] = val;
                }

                if (Ndata == 1){
                    inp >> val;
                    Values[i].push_back(val);
                }
                else{
                    bool set_val = true;
                    for (unsigned int j = 0; j < Ndata; j++){
                        inp >> val;
                        if (set_val){
                            Values[i].push_back(val);
                            set_val = false;
                        }
                        else{
                            Elevations[i].push_back(val);
                            set_val = true;
                        }

                    }
                }

                if (i == 0)
                    Length[i] = 0;
                else{
                    double dst = Pnts[i].distance(Pnts[i-1]);
                    Length[i] = Length[i-1] + dst;
                }
            }
        }
    }
}

template <int dim>
bool BoundaryInterp<dim>::isPoint_onSeg(Point<dim> p, int iSeg, double& dst_t){

    dst_t = -9999999999;
    if (iSeg >= Pnts.size() - 1)
        return false;

    double dst = distance_point_line(p[0], p[1], Pnts[iSeg][0], Pnts[iSeg][2], Pnts[iSeg+1][0], Pnts[iSeg+1][2]);
    double dstA = distance_2_points(p[0], p[1], Pnts[iSeg][0], Pnts[iSeg][1]);
    double dstB = distance_2_points(p[0], p[1], Pnts[iSeg+1][0], Pnts[iSeg+1][0]);
    double min_dst = std::min(dstA, dstB);

    if (std::abs(dst) < tolerance){
        if (dst < 0){
            if (min_dst < tolerance){
                dst_t = dstA;
                return true;
            }
            else
                return false;
        }
        else{
            dst_t = dstA;
            return true;
        }
    }
    else
        return false;
}

template <int dim>
bool BoundaryInterp<dim>::isPoint_onBoundary(Point<dim> p){
    for (unsigned int ibnd = 0; ibnd < Npnts-1; ++ibnd){
        double dst = 0;
        if (isPoint_onSeg(p, ibnd, dst))
            return true;
    }

    return false;
}



#endif // BOUNDARYINTERP_H
