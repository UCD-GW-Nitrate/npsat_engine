#ifndef NEUMANN_BOUNDARY_H
#define NEUMANN_BOUNDARY_H


//#include <deal.II/dofs/function_map.h>
#include <deal.II/distributed/tria.h>

#include "interpolation/interpolation_function.h"

//#include "cgal_functions.h"

#include "boost_functions.h"


namespace BoundaryConditions{

using namespace dealii;

enum class BoundaryType{BOT,BOUNDARY_LINE};

/*! A primitive boundary shape.
 *
 * A primitive shape boundary is a polygon that is associated with a single constant value.
 *
 */

class BoundPrim{
public:
    //! This holds the X coordinates of the polygon
    std::vector<double> Xcoords;

    //! This holds the Y coordinates of the polygon
    std::vector<double> Ycoords;

    //! This is the dirichlet value
    std::string value;

    //! The type of boundary. It can be any of the keywords TOP, BOT, EDGE
    std::string TYPE;

    //! This is the minimum point of the polygon bounding box
    Point<2> BBmin;

    //! This is the maximum point of the polygon bounding box
    Point<2> BBmax;

    //! A method that checks if any point in the list is inside the bounding box of the primitive polygon
    bool is_any_point_insideBB(std::vector<double> x, std::vector<double> y);

    bool is_any_point_insideBB(double x, double y);

    bool Point_in_BB(double xmin, double ymin, double xmax, double ymax, double x, double y);

    bool is_point_inside_poly(double x, double y);

    boost_polygon poly;
};

bool BoundPrim::is_any_point_insideBB(std::vector<double> x, std::vector<double> y){
    bool outcome = false;
    double xmin = 99999999999;
    double ymin = 99999999999;
    double xmax = -99999999999;
    double ymax = -99999999999;
    for (unsigned int i = 0; i < x.size(); ++i){
        if (xmin > x[i]) xmin = x[i];
        if (ymin > y[i]) ymin = y[i];
        if (xmax < x[i]) xmax = x[i];
        if (ymax < y[i]) ymax = y[i];
        if (Point_in_BB(BBmin[0], BBmin[1], BBmax[0], BBmax[1], x[i], y[i])){
            outcome = true;
            break;
        }
    }

    if (!outcome){
        if (Point_in_BB(xmin, ymin, xmax, ymax, BBmin[0], BBmin[1]))
            outcome = true;
        else{
            if (Point_in_BB(xmin, ymin, xmax, ymax, BBmax[0], BBmax[1]))
                outcome = true;
            else{
                if (Point_in_BB(xmin, ymin, xmax, ymax, BBmin[0], BBmax[1]))
                    outcome = true;
                else{
                    if (Point_in_BB(xmin, ymin, xmax, ymax, BBmax[0], BBmin[1]))
                        outcome = true;
                }
            }
        }
    }
    return outcome;
}

bool BoundPrim::Point_in_BB(double xmin, double ymin, double xmax, double ymax, double x, double y){
    if (x > xmin && x < xmax && y > ymin && y < ymax)
        return true;
    else
        return false;
}

bool BoundPrim::is_any_point_insideBB(double x, double y){
    if (Point_in_BB(BBmin[0], BBmin[1], BBmax[0], BBmax[1], x, y)){
        return true;
    }
    return false;
}

bool BoundPrim::is_point_inside_poly(double x, double y){
    boost_point pnt(x,y);
    return boost::geometry::within(pnt, poly);
}


template<int dim>
class Neumann{
public:
    //! A constructor that does nothing
    Neumann();

    double interpolate(Point<dim> p, int ibnd);

    /*!
     * \brief Reads the Neumann boundary condition file. This file should contain flow boundaries around the domain
     * or at the bottom. For the flows from the top set as recharge.
     *
     * The format of the input file is as follows
     * NBND : the number of boundaries in the file
     * next line
     * TYPE N mult VALUE
     * where
     * TYPE : is either BOT or BOUNDARY_LINE
     * N : if the type is BOT N is the number of points that define the polygon
     * For BOUNDARY_LINE N must be 0
     * mult: is a multiplier for the boundary values. set it to 1 if no multiplier is used
     * VALUE: interpolation master file relative to the input directory.
     * For BOUNDARY_LINE the master must contain a LATERAL_POLYLINE interpolant.
     * next lines
     * For BOT the next N lines should contain the x and y coordinates of the polygon
     * X Y
     *
     *
     * @param filename
     * @return
     */
    bool getData(std::string filename, const std::string &input_path, MPI_Comm comm);
    int Nbnd();
    BoundaryType getType(int i);
    bool isPartofBoundary(Point<dim> p, int ibnd);
    bool isPartofBoundary(Point<dim> pa, Point<dim> pb, int ibnd);


private:
    int Nb = 0;
    std::vector<BoundPrim> boundary_parts;
    std::vector<std::shared_ptr<npsat_flow::InterpInterface<dim>>> interp_funct;
    std::vector<BoundaryType> bnd_type;
    std::vector<double> multiplier;

};

template<int dim>
Neumann<dim>::Neumann(){
    Nb = 0;
}

template<int dim>
int Neumann<dim>::Nbnd(){
    return Nb;
}

template<int dim>
BoundaryType Neumann<dim>::getType(int i){
    return bnd_type[i];
}

template<int dim>
bool Neumann<dim>::isPartofBoundary(Point<dim> p, int ibnd) {
    bool out = false;
    if (ibnd >= bnd_type.size())
        return out;

    if (bnd_type[ibnd] == BoundaryType::BOT){
        if (boundary_parts[ibnd].is_any_point_insideBB(p[0], p[1])){
            if (boundary_parts[ibnd].is_point_inside_poly(p[0], p[1])){
                out = true;
            }
        }
    }
    return out;
}

template<int dim>
bool Neumann<dim>::isPartofBoundary(Point<dim> pa, Point<dim> pb, int ibnd) {
    bool out = false;
    if (ibnd >= bnd_type.size())
        return out;
    if (bnd_type[ibnd] == BoundaryType::BOUNDARY_LINE){
        out = interp_funct[ibnd]->face_matches_lateral_overlap(pa,pb);
    }
    return out;
}

template<int dim>
double Neumann<dim>::interpolate(Point<dim> p, int ibnd){

    double out = 0.0;
    if (dim == 2){
        std::cout << "Neuman conditions in 2D are not impemented yet" << std::endl;
    }
    else if (dim == 3){
        out = interp_funct[ibnd]->interpolate(p, 0) * multiplier[ibnd];
        //if (bnd_type[ibnd] == BoundaryType::BOT){
        //    if (boundary_parts[ibnd].is_any_point_insideBB(p[0], p[1])){
        //        if (boundary_parts[ibnd].is_point_inside_poly(p[0], p[1])){
        //            out = interp_funct[ibnd]->interpolate(p, 0) * multiplier[ibnd];
        //        }
        //    }
        //}
        //else if (bnd_type[ibnd] == BoundaryType::BOUNDARY_LINE){
        //    out = interp_funct[ibnd]->interpolate(p, 0) * multiplier[ibnd];
        //}
    }
    return out;
}

template<int dim>
bool Neumann<dim>::getData(std::string filename, const std::string &input_path, MPI_Comm comm){
    std::ifstream  datafile(filename.c_str());
    if (!datafile.good()){
        std::cout << "Can't open " << filename << std::endl;
        return false;
    }
    else{
        std::string line, tmp;
        {// NBND number of boundaries
            getline(datafile, line);
            std::istringstream inp(line.c_str());
            inp >> Nb;
            boundary_parts.resize(Nb);
            interp_funct.resize(Nb);
            bnd_type.resize(Nb);
            multiplier.resize(Nb);
        }
        for (int i = 0; i < Nb; ++i){
            std::string type, value;
            int np;
            double mult;

            {// TYPE N VALUE (For each boundary)
                getline(datafile, line);
                std::istringstream inp(line.c_str());
                inp >>type;
                if (type.compare("BOT") == 0){
                    bnd_type[i] = BoundaryType::BOT;
                }
                else if (type.compare("BOUNDARY_LINE") == 0){
                    bnd_type[i] = BoundaryType::BOUNDARY_LINE;
                }
                else{
                    std::cout << "Unknown type of Neumann condition: " << type << std::endl;
                }
                inp >> np;
                inp >> mult;
                multiplier[i] = mult;
                inp >> value;
            }
            if (dim == 2){
                std::cout << "Neumann readers are not implemented for 2D" << std::endl;
                return false;
            }
            else if (dim == 3){
                switch (bnd_type[i]) {
                case BoundaryType::BOT:
                {// in BOT case the geometry is a polygon
                    double x, y;
                    std::vector<boost_point> tmp_pnts;
                    boundary_parts[i].BBmin[0] = 99999999999;
                    boundary_parts[i].BBmin[1] = 99999999999;
                    boundary_parts[i].BBmax[0] = -99999999999;
                    boundary_parts[i].BBmax[1] = -99999999999;
                    for (int iv = 0; iv < np; ++iv){
                        getline(datafile, line);
                        std::istringstream inp(line.c_str());
                        inp >> x;
                        // X coordinate
                        if (x < boundary_parts[i].BBmin[0])
                            boundary_parts[i].BBmin[0] = x;
                        if (x > boundary_parts[i].BBmax[0])
                            boundary_parts[i].BBmax[0] = x;
                         boundary_parts[i].Xcoords.push_back(x);

                         // Y coordinate
                         inp >> y;
                         if (y < boundary_parts[i].BBmin[1])
                             boundary_parts[i].BBmin[1] = y;
                         if (y > boundary_parts[i].BBmax[1])
                             boundary_parts[i].BBmax[1] = y;
                         boundary_parts[i].Ycoords.push_back(y);
                         tmp_pnts.push_back(boost_point(x,y));

                    }
                    boost::geometry::assign_points(boundary_parts[i].poly, tmp_pnts);
                    boost::geometry::correct(boundary_parts[i].poly);
                    interp_funct[i] = std::make_shared<npsat_flow::InterpInterface<dim>>();
                    interp_funct[i]->read_master_file(value, 1.0, comm, input_path);
                }
                break;
                case BoundaryType::BOUNDARY_LINE:{
                    interp_funct[i] = std::make_shared<npsat_flow::InterpInterface<dim>>();
                    interp_funct[i]->read_master_file(value, 1.0, comm, input_path);
                }
                break;
                }
            }
        }
    }
    datafile.close();
    return true;
}

}

#endif // NEUMANN_BOUNDARY_H
