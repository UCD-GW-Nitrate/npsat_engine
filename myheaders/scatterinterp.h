#ifndef SCATTERINTERP_H
#define SCATTERINTERP_H

#include <fstream>

#include <deal.II/base/point.h>

#include "cgal_functions.h"


using namespace dealii;

//! Interpolation Class
/*!
 * The class it is written in a dimension independent style however has some limitations on 3D interpolation.
 * For the 2D case works as expected. For 3D however the function assumes that scattered points are structured
 * into layers and it does not support fully unstructured 3D interpolation. For 1D interpolation the class uses
 * a simple linear interpolation.
 *
 * The class has to main functionalities
 * i) reads scattered data from a file. ii) interpolates value of a given point.
 */
template<int dim>
class ScatterInterp{
public:
    //! The constractor just initialize default value for the number of data and the number of points.
    ScatterInterp();

    //! Reads the data from file.
    /*!
     * \brief get_data
     * \param filename is the name of the file that contains the data.
     *
     * The format of the file must be the following:
     * The first line should have the keyword SCATTERED which indicates that the data
     * that follow data correspond to scattered interpolation
     *
     * The second line provides two numbers #Npnts and #Ndata.
     *
     * The following lines after that expect an array of #Npnts rows and #Ndata+2 columns with the following format
     * X Y V_1 Z_1 V_2 Z_2 ... V_lay-1 Z_lay-1 V_lay
     *
     * The above line is repeated #Npnts times
     *
     * For example if we assume that the interpolation data are structured into 3 layers the on should provide
     * x and y coordinates, 2 pairs of values v, z, and an additional value. It the point in question has higher z
     * that the elevation of the first layer the method returns the value that corresponds to the first layer. If the
     * elevation of the point is lower than the layer above the last minus one layer then the interpolation returns the last value
     * In short the method expects for each 2d scattered point the following info
     *
     *      v1
     * ----------z1
     *      v2
     * ----------z2
     *      v3
     * ----------z3
     *      .
     *      .
     *      v(lay-1)
     * ----------z(lay-1)
     *      vlay
     */
    void get_data(std::string filename);

    /*!
     * \brief interpolate calculates the interpolation.
     * \param p The point which we want to find its value
     * \return The interpolated value
     */
    double interpolate(Point<dim> p)const;

private:

    //! this is a container to hold the triangulation of the 2D scattered data
    ine_Delaunay_triangulation T;

    //! This is a map between the triangulation and the values that correspond to each 2D point
    std::vector<std::map<ine_Point2, ine_Coord_type, ine_Kernel::Less_xy_2> > function_values;

    //! Ndata is the number of values for interpolation. For a 3D layer structured this number must be equal to (Nlay-1)*2 +1
    unsigned int Ndata;

    //! Npnts is the number of 2D points that the scattered interpolation set contains
    unsigned int Npnts;

    //! X_1D is the vector of x of points for the 1D interpolation of a function y=f(x)
    std::vector<double> X_1D;

    //! V_1D is the vector of y of points for the 1D interpolation of a function y=f(x)
    std::vector<double> V_1D;

};

template<int dim>
ScatterInterp<dim>::ScatterInterp(){
    Ndata = 0;
    Npnts = 0;
}

template <int dim>
void ScatterInterp<dim>::get_data(std::string filename){
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
            if (temp != "SCATTERED"){
                std::cerr << " ScatterInterp Cannot read " << temp << " data." << std::endl;
                return;
            }
        }
        {//Read number of points and number of data
            datafile.getline(buffer,512);
            std::istringstream inp(buffer);
            inp >> Npnts;
            inp >> Ndata;
            function_values.resize(Ndata);
        }
        {// Read the actual data
            double x, y, v;
            for (unsigned int i = 0; i < Npnts; ++i){
                datafile.getline(buffer, 512);
                std::istringstream inp(buffer);
                if (dim == 1){
                    inp >> x;
                    inp >> v;
                    // Here we need to store the values
                    X_1D.push_back(x);
                    V_1D.push_back(v);
                }
                else{
                    inp >> x;
                    inp >> y;
                    ine_Point2 p(x, y);
                    T.insert(p);
                    for (unsigned int j = 0; j < Ndata; ++j){
                        inp >> v;
                        ine_Coord_type ct(v);
                        function_values[j].insert(std::make_pair(p,ct));
                    }
                }
            }
        }// Read data block
    }
}

template <int dim>
double ScatterInterp<dim>::interpolate(Point<dim> point)const{
    Point<3> pp;
    if (dim == 1){
        std::cerr << "Not implemented yet" << std::endl;
        return 0;
    }
    else if (dim == 2) {
        pp[0] = point[0];
        pp[1] = point[1];
        pp[2] = 0;
    } else if (dim == 3){
        pp[0] = point[0];
        pp[1] = point[1];
        pp[2] = point[2];
    }

    double value = scatter_2D_interpolation(T, function_values, pp);
    return  value;
}


#endif // SCATTERINTERP_H
