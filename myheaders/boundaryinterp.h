#ifndef BOUNDARYINTERP_H
#define BOUNDARYINTERP_H

#include <fstream>

#include <deal.II/base/point.h>

#include "helper_functions.h"
#include "scatterinterp.h"

using namespace dealii;

template <int dim>
class BoundaryInterp{
public:
    //! An empty constructor
    BoundaryInterp();

    //! returns the interpolated value
    double interpolate(Point<dim> p)const;

    //! read data from file
    void get_data(std::string filename);

    //! This will return true if the face defined by the two nodes is part of any segment of the boundary
    bool is_face_part_of_BND(Point<dim> A, Point<dim> B);

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

    bool isPoint_onSeg(Point<dim> p, int iSeg, double& dst_t)const;

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
                std::cerr << " BOUNDARY_LINE Cannot read " << temp << " data." << std::endl;
                return;
            }
        }

        {// Read the number of data and allocate space
            datafile.getline(buffer, 512);
            std::istringstream inp(buffer);
            inp >> Npnts;
            inp >> Ndata;
            inp >> tolerance;
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
bool BoundaryInterp<dim>::isPoint_onSeg(Point<dim> p, int iSeg, double& dst_t) const{

    dst_t = -9999999999;
    if (iSeg >= Pnts.size() - 1)
        return false;

    double dst = distance_point_line(p[0], p[1], Pnts[iSeg][0], Pnts[iSeg][1], Pnts[iSeg+1][0], Pnts[iSeg+1][1]);
    double dstA = distance_2_points(p[0], p[1], Pnts[iSeg][0], Pnts[iSeg][1]);
    double dstB = distance_2_points(p[0], p[1], Pnts[iSeg+1][0], Pnts[iSeg+1][1]);
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

template <int dim>
bool BoundaryInterp<dim>::is_face_part_of_BND(Point<dim> A, Point<dim> B){
    double lx1, ly1, lx2, ly2; // variables for storing the boundary coordinates
    double cx3, cy3, cx4, cy4; // variables for storing the cell face coordinates

    cx3 = A[0]; cy3 = A[1];
    cx4 = B[0]; cy4 = B[1];
    if (Point<2>(cx3,cy3).distance(Point<2>(cx4,cy4)) < 0.1){
        std::cerr << " This face is too small. Maybe its an error" << std::endl;
    }

    for (unsigned int ii = 0; ii < Npnts - 1; ii++){
        lx1 = Pnts[ii][0]; ly1 = Pnts[ii][1];
        lx2 = Pnts[ii+1][0]; ly2 = Pnts[ii+1][1];

        // Calculate the distance of the two cell points from the boundary line
        double dst1 = distance_point_line(cx3,cy3,lx1,ly1,lx2,ly2);
        double dst2 = distance_point_line(cx4,cy4,lx1,ly1,lx2,ly2);

        // The cell face is collinear with the boundary line if the distances are very close to zero
        // and one of the two distances is positive.
        bool are_colinear = false;
        if (std::abs(dst1) < tolerance && std::abs(dst2) < tolerance){
            if ( !(dst1 < 0) || !(dst2 < 0)){
                are_colinear = true;
            }
            else {
                // It may be possible due to numerical errors that the distances are both negative
                // This can happen under two circumstances.
                // 1) The boundary line is smaller than the cell face. This means that the boundary condition
                //    lines have not been set correctly.
                //    FUTURE VERSION OF THE CODE SHOULD ADDRESS THIS CASE
                // 2) The boundary segment is identical with the cell face. Then it is possible that both points
                //    of the cell may appear outside of the boundary by very small amount.

                //====== Case 2 ======
                { // Take care the second case
                    double min_dst1 = std::min(distance_2_points(cx3,cy3,lx1,ly1),distance_2_points(cx3,cy3,lx2,ly2));
                    double min_dst2 = std::min(distance_2_points(cx4,cy4,lx1,ly1),distance_2_points(cx4,cy4,lx2,ly2));
                    if (min_dst1 < 0.1 && min_dst2 < 0.1){
                        are_colinear = true;
                    }
                }
            }
        }
        if (are_colinear){
            return true;
        }
    }
    return false;
}

template <int dim>
double BoundaryInterp<dim>::interpolate(Point<dim> p)const{

    for (unsigned int i = 0; i < Npnts - 1; ++i){
        double dst_t = 0;
        if (isPoint_onSeg(p, i, dst_t)){
            double t = dst_t/(Length[i+1] - Length[i]);
            if (Ndata == 1){
                //std::cout << "plot(" << p[0] << "," << p[1] << ",'xg')" << std::endl;
                return Values[i][0] * (1-t) + Values[i+1][0]*t;
            }
            else{
                std::cout << "WARNING: This part of the BoundaryInterp<dim>::interpolate has NOT been debuged" << std::endl;
                std::cout << "The interpolations are more than likely wrong!!!" << std::endl;
                double zup, zdown;
                zup = Elevations[i][0] * t + Elevations[i+1][0]*(1-t);
                if (p(2) > zup){
                    return Values[i][0] * t + Values[i+1][0]*(1-t);
                }
                else{
                    for (unsigned int j = 1; j < Elevations[i].size(); ++j){
                        zdown = Elevations[i][j] * t + Elevations[i+1][j]*(1-t);
                        if (p(2) < zup && p(2) > zdown){
                            double u = (p(2) - zdown)/(zup - zdown);
                            double vup = Values[i][j-1] * t + Values[i+1][j-1]*(1-t);
                            double vdown = Values[i][j] * t + Values[i+1][j]*(1-t);
                            return u*vdown + (1-u)*vup;
                        }
                    }
                    if (p(2) < zdown){
                        unsigned int j = Elevations[i].size()-1;
                        return Values[i][j] * t + Values[i+1][j]*(1-t);
                    }
                    else{
                        std::cerr << "The z of point " << p << " is out of this world!" << std::endl;
                    }
                }
            }
            break;
        }
    }

    //std::cerr << "The Interpolation should never reach this point. There must be something wrong when assigning BC" << std::endl;
    std::cout << "plot(" << p[0] << "," << p[1] << ",'or')" << std::endl;
    return -999999999;
}



#endif // BOUNDARYINTERP_H
