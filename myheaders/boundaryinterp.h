#ifndef BOUNDARYINTERP_H
#define BOUNDARYINTERP_H

#include <fstream>

#include <deal.II/base/point.h>

#include "helper_functions.h"
//#include "scatterinterp.h"

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
    std::vector<Point<dim>> Pnts;
    std::vector<std::vector<double>> Values;
    std::vector<std::vector<double>> Elevations;
    std::vector<double> Length;

    //! Number of corner points
    unsigned int Npnts;

    //! Number of data along Z
    unsigned int Ndata;

    int Nlayers;

    SCI_TYPE sci_type;
    SCI_METHOD sci_methodXY;
    SCI_METHOD sci_methodZ;

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

        {// Read Interpolation type
            datafile.getline(buffer, 512);
            std::istringstream inp(buffer);
            std::string temp;
            inp >> temp;
            if (temp == "3D")
                sci_type = SCI_TYPE::DIM3;
            else if (temp == "2D")
                sci_type = SCI_TYPE::DIM2;
            else if (temp == "VERT")
                sci_type = SCI_TYPE::VERT;
            else
                std::cout << "Unknown interpolation type. Valid options are DIM3, DIM2, VERT" << std::endl;
        }

        {// Read interpolation method
            datafile.getline(buffer, 512);
            std::istringstream inp(buffer);
            std::string temp;
            inp >> temp;
            if (temp == "LINEAR"){
                sci_methodXY = SCI_METHOD::LINEAR;
            }
            else if (temp == "NEAREST"){
                sci_methodXY = SCI_METHOD::NEAREST;
            }
            else
                std::cout << "Unknown interpolation style. Valid options are LINEAR or NEAREST" << std::endl;

            if (sci_type == SCI_TYPE::DIM3){
                // Read the interpolation method along the z
                inp >> temp;
                if (temp == "LINEAR")
                    sci_methodZ = SCI_METHOD::LINEAR;
                else if (temp == "NEAREST")
                    sci_methodZ = SCI_METHOD::NEAREST;
                else
                    std::cout << "Unknown interpolation style. Valid options are LINEAR or NEAREST" << std::endl;
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
            if (Ndata == 1)
                Nlayers = 1;
            else if (Ndata > 1){
                Elevations.resize(Npnts);
                if (sci_methodZ == SCI_METHOD::LINEAR){
                    if (Ndata % 2 != 0){
                        std::cerr << " In ScatterInterp file: " << filename
                                  << " The number of data are incompatible h the Z layer interpolation type" << std::endl;
                    }
                    else{
                        Nlayers = Ndata/2;
                    }
                }
                else if (sci_methodZ == SCI_METHOD::NEAREST){
                    if (Ndata % 2 == 0){
                        std::cerr << " In ScatterInterp file: " << filename
                                  << " The number of data are incompatible h the Z layer interpolation type" << std::endl;
                    }
                    else{
                        Nlayers = (Ndata-1)/2;
                    }
                }
            }
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
    if (iSeg >= static_cast<int>(Pnts.size() - 1))
        return false;

    // Calculate the projection of the point p onto the segment iSeg
    Point<dim> pp = project_point_onLine<dim>(p, Pnts[iSeg], Pnts[iSeg+1]);
    // Compute the distance between the point and its projection
    double dst = distance_2_points(pp[0], pp[1], p[0],p[1]);
    if (dst < 0.1){
        double t = 0.0;
        if (std::abs(Pnts[iSeg+1][0] - Pnts[iSeg][0]) > std::abs(Pnts[iSeg+1][1] - Pnts[iSeg][1]) ){
            t = (pp[0] - Pnts[iSeg][0])/(Pnts[iSeg+1][0] - Pnts[iSeg][0]);
        }
        else{
            t = (pp[1] - Pnts[iSeg][1])/(Pnts[iSeg+1][1] - Pnts[iSeg][1]);
        }

        dst_t = distance_2_points(p[0], p[1], Pnts[iSeg][0], Pnts[iSeg][1]);

        if (std::abs(t) < 0.001){
            if (dst_t < 0.1){
                t = 0.00000001;
                dst_t = 0;
            }
        }
        else if (std::abs(t-1) < 0.001){
            double dst = distance_2_points(p[0], p[1], Pnts[iSeg+1][0], Pnts[iSeg+1][1]);
            if (dst < 0.1){
                t = 0.99999999;
            }
        }

        if ( t >= 0.0 && t <=1 ){
            return true;
        }
    }
    return false;


    /*
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

    return false;
    */
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
        bool tf = areSegmentsCollinear(A, B, Pnts[ii], Pnts[ii+1]);
        if (!tf)
            continue;

        Point<dim> Q;
        Q[0] = A[0]*0.7887 + B[0]*0.2113;
        Q[1] = A[1]*0.7887 + B[1]*0.2113;
        tf = onSegment<dim>(Pnts[ii], Q, Pnts[ii+1]);
        if(!tf){
            Q[0] = B[0]*0.7887 + A[0]*0.2113;
            Q[1] = B[1]*0.7887 + A[1]*0.2113;
            tf = onSegment<dim>(Pnts[ii], Q, Pnts[ii+1]);
            if (tf)
                return true;
        }
        else
            return true;
        /*
        //Calculate the parametric coordinate of the face points onto this segment
        double ta = 0.0;
        double tb = 0.0;
        if (std::abs(Pnts[ii+1][0] - Pnts[ii][0]) > std::abs(Pnts[ii+1][1] - Pnts[ii][1]) ){
            ta = (A[0] - Pnts[ii][0])/(Pnts[ii+1][0] - Pnts[ii][0]);
            tb = (B[0] - Pnts[ii][0])/(Pnts[ii+1][0] - Pnts[ii][0]);
        }
        else{
            ta = (A[1] - Pnts[ii][1])/(Pnts[ii+1][1] - Pnts[ii][1]);
            tb = (B[1] - Pnts[ii][1])/(Pnts[ii+1][1] - Pnts[ii][1]);
        }
        if (std::abs(ta) < 0.001){
            double dst = distance_2_points(A[0], A[1], Pnts[ii][0], Pnts[ii][1]);
            if (dst < 0.1)
                ta = 0.00000001;
        }
        else if (std::abs(ta-1) < 0.001){
            double dst = distance_2_points(A[0], A[1], Pnts[ii+1][0], Pnts[ii+1][1]);
            if (dst < 0.1)
                ta = 0.99999999;
        }

        if (std::abs(tb) < 0.001){
            double dst = distance_2_points(B[0], B[1], Pnts[ii][0], Pnts[ii][1]);
            if (dst < 0.1)
                tb = 0.00000001;
        }
        else if (std::abs(tb-1) < 0.001){
            double dst = distance_2_points(B[0], B[1], Pnts[ii+1][0], Pnts[ii+1][1]);
            if (dst < 0.1)
                tb = 0.99999999;
        }

        if ( (ta >= 0.0 && ta <=1) || (tb >= 0.0 && tb <=1)){
            return true;
        }
        */
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

                if (sci_methodXY == SCI_METHOD::LINEAR){
                    return Values[i][0] * (1-t) + Values[i+1][0]*t;
                }
                else if (sci_methodXY == SCI_METHOD::NEAREST){
                    return (t <= 0.5)? Values[i][0]: Values[i+1][0];
                }
            }
            else{
                double zup, zdown, vup, vdown;
                zup = (sci_methodXY == SCI_METHOD::LINEAR)? Elevations[i][0] * (1-t) + Elevations[i+1][0] * t:
                        (t < 0.5)? Elevations[i][0]: Elevations[i+1][0];

                if (p(2) >= zup){
                    return (sci_methodXY == SCI_METHOD::LINEAR)? Values[i][0] * (1-t) + Values[i+1][0]*t:
                            (t <= 0.5)? Values[i][0]: Values[i+1][0];
                }
                else{
                    for (unsigned int j = 1; j < Nlayers; ++j){
                        zdown = (sci_methodXY == SCI_METHOD::LINEAR)? Elevations[i][j] * (1-t) + Elevations[i+1][j]*t:
                                (t <= 0.5)? Elevations[i][j]: Elevations[i+1][j];
                        if (p[2] <= zup && p[2] >= zdown){
                            if (sci_methodZ == SCI_METHOD::NEAREST){
                                return (sci_methodXY == SCI_METHOD::LINEAR)? Values[i][j] * (1-t) + Values[i+1][j] * t:
                                       Values[i][j];
                            }
                            else{
                                double tz = (p[2] - zdown)/(zup - zdown);
                                vup = (sci_methodXY == SCI_METHOD::LINEAR)? Values[i][j-1] * (1-t) + Values[i+1][j-1] * t:
                                        (t < 0.5)? Values[i][j-1]: Values[i+1][j-1];
                                vdown = (sci_methodXY == SCI_METHOD::LINEAR)? Values[i][j] * (1-t) + Values[i+1][j] * t:
                                        (t < 0.5)? Values[i][j]: Values[i+1][j];
                                return vup * tz + vdown * (1-tz);
                            }
                        }
                        zup = zdown;
                    }
                    if (p[2] <= zdown){
                        return (sci_methodXY == SCI_METHOD::LINEAR)? Values[i][Values[i].size()-1] * (1-t) + Values[i+1][Values[i+1].size()-1]*t:
                               (t <= 0.5)? Values[i][Values[i].size()-1]: Values[i+1][Values[i+1].size()-1];
                    }
                }
            }
            break;
        }
    }

    std::cerr << "The Interpolation should never reach this point. There must be something wrong when assigning BC" << std::endl;
    std::cout << "plot(" << p[0] << "," << p[1] << ",'or')" << std::endl;
    return -999999999;
}



#endif // BOUNDARYINTERP_H
