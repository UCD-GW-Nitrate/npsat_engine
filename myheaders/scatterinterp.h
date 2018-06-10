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
     * The second line is the keyward STRATIFIED or SIMPLE
     *
     * The second line provides two numbers #Npnts and #Ndata.
     *
     * The following lines after that expect an array of #Npnts rows and #Ndata+2 columns with the following format
     * X Y V_1 Z_1 V_2 Z_2 ... V_lay-1 Z_lay-1 V_lay
     *
     * The above line is repeated #Npnts times
     *
     * For example if we assume that the interpolation data are structured into 3 layers then one should provide
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

    //! Ndata is the number of values for interpolation. For the STRITIFIED option this number must be equal to (Nlay-1)*2 +1
    unsigned int Ndata;

    //! Npnts is the number of 2D points that the scattered interpolation set contains
    unsigned int Npnts;

    //! X_1D is the vector of x of points for the 1D interpolation of a function y=f(x)
    std::vector<double> X_1D;

    //! V_1D is the vector of vectors y of points for the 1D interpolation of list of functions of the form y_1=f(x), y_2=f(x), y_3=f(x) etc.
    //! It make sense to define more than one data for stratified hydraulic conductivity in 2D. See the discussion above.
    std::vector<std::vector<double>> V_1D;

    bool Stratified;

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

        {// Read interpolation style
            datafile.getline(buffer, 512);
            std::istringstream inp(buffer);
            std::string temp;
            inp >> temp;
            if (temp == "STRATIFIED")
                Stratified = true;
            else if (temp == "SIMPLE")
                Stratified = false;
            else
                std::cout << "Unknown interpolation style. Valid options are STRATIFIED or SIMPLE" << std::endl;

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
                if (dim == 1 || (dim == 2 && Stratified)){
                    inp >> x;
                    X_1D.push_back(x);
                    std::vector<double> temp;
                    for (unsigned int j = 0; j < Ndata; ++j){
                        inp >> v;
                        temp.push_back(v);
                    }
                    // Here we need to store the values
                    V_1D.push_back(temp);
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
    if (dim == 1 || (dim == 2 && Stratified)){
        // find main parametric value
        double t;
        int ind;
        bool first = false;
        bool last = false;

        if (point[0] <=X_1D[0])
            first = true;
        else if (point[0] >= X_1D[X_1D.size()-1])
            last = true;
        else{
            for (unsigned int i = 0; i < X_1D.size()-1; ++i){
                if (point[0] >= X_1D[i] && point[0] <= X_1D[i+1]){
                    t = (point[0] - X_1D[i]) / (X_1D[i+1] - X_1D[i]);
                    ind = i;
                    break;
                }
            }
        }

        if (dim == 1){
            if (first)
                return V_1D[0][0];
            else if (last)
                return  V_1D[V_1D.size()-1][0];
            else{
                return V_1D[ind][0]*(1-t) + V_1D[ind+1][0]*t;
            }
        }
        else if (dim == 2 && Stratified){
            std::vector<double> v;
            std::vector<double> el;
            bool push_v = true;
            double v_temp;
            for (unsigned int i = 0; i < Ndata; ++i){
                if (first)
                    v_temp = V_1D[0][i];
                else if (last)
                    v_temp = V_1D[V_1D.size()-1][i];
                else{
                    v_temp = V_1D[ind][i]*(1-t) + V_1D[ind+1][i]*t;
                }

                if (push_v){
                    v.push_back(v_temp);
                    push_v = false;
                }
                else{
                    el.push_back(v_temp);
                    push_v = true;
                }
            }

            if (point[1] <= el[0])
                return v[0];
            else if (point[1]>=el[el.size()-1])
                return v[v.size()-1];
            else{
                for (unsigned int i = 0; i < el.size()-1; ++i){
                    if (point[1] >= el[i] && point[1] <= el[i+1]){
                        double u = (point[1] - el[i])/(el[i+1] - el[i]);
                        return v[i]*(1-u) + v[i+1]*u;
                    }
                }
            }

        }
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
