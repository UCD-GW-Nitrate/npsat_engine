#ifndef SCATTERINTERP_H
#define SCATTERINTERP_H

#include <fstream>

#include <deal.II/base/point.h>

//#include "cgal_functions.h"
#include "nanoflann_structures.h"
#include "helper_functions.h"

/*!
 * \brief The SCI_TYPE enum can take one of the 3 values
 * - FULL: In 3D this interpolates in 3D space. However this is possible only if the interpolation
 * has z information therefore it has to be STRATIFIED. The SIMPLE option is not valid for 3D FULL types
 * In 2D problems this interpolates along the x-y plane. The y coordinate it is supposed to be the z coordinate.
 * In 2D, it can be either SIMPLE or STRATIFIED. It is simple if the given points form an unstructure grid of points.
 * If the points form layers then the STRATIFIED option can be used.
 * - HOR interpolation in 3D is a 2d interpolation across x-y. It can be used to interpolate recharge or top and bottom elevation
 * In 2D this defines an interpolation along the x axis. In 2D the SIMPLE or STRATIFIED options are not taken into consideration
 * - VERT is defined in 3D problems and it is an interpolation across an a vertical plane which is defined by 2
 * points.
 */
enum SCI_TYPE { DIM3, DIM2, VERT };
enum SCI_METHOD{LINEAR, NEAREST};

using namespace dealii;

//! Interpolation Class
/*!
 * The class it is written in a dimension independent style however has some limitations on 3D interpolation.
 * For the 2D case works as expected. For 3D however the function assumes that scattered points are structured
 * into layers and it does not support fully unstructured 3D interpolation. For 1D interpolation the class uses
 * a simple linear interpolation.
 *
 * The class has two main functionalities
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
     * The second line is one of the three keywords FULL, HOR, VERT
     * Change the keywords to make more sense 3D 2D, VERT
     *
     * The third line is the keyword STRATIFIED or SIMPLE
     * Change the keywords to LINEAR, NEAREST
     *
     * The fourth line provides two numbers #Npnts and #Ndata.
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

    /*!
     * \brief set_edge_points In the case of VERT interpolation the class has to know the coordinates of the
     * line that defines the vertical plane. Note the order of the points is important. The first point should
     * correspond to coordinate 0.
     * \param a is the first point of the line
     * \param b is the last point of the line
     */
    //void set_edge_points(Point<dim> a, Point<dim> b);

private:

    //! this is a container to hold the triangulation of the 2D scattered data
    //ine_Delaunay_triangulation T;
    PointVectorCloud interpCloud;
    //PointVectorCloud* ptrCloud= &interpCloud;
    std::shared_ptr<pointVector_kd_tree> interpIndex;


    //! This is a map between the triangulation and the values that correspond to each 2D point
    //std::vector<std::map<ine_Point2, ine_Coord_type, ine_Kernel::Less_xy_2> > function_values;

    //! Ndata is the number of values for interpolation. For the STRATIFIED option this number must be equal to (Nlay-1)*2 +1
    unsigned int Ndata;

    //! Npnts is the number of 2D points that the scattered interpolation set contains
    unsigned int Npnts;

    //! X_1D is the vector of x of points for the 1D interpolation of a function y=f(x)
    std::vector<double> X_1D;

    //! V_1D is the vector of vectors y of points for the 1D interpolation of list of functions of the form y_1=f(x), y_2=f(x), y_3=f(x) etc.
    //! It make sense to define more than one data for stratified hydraulic conductivity in 2D. See the discussion above.
    std::vector<std::vector<double>> V_1D;

    //bool Stratified;

    //bool interpolated;
    size_t NinterpPoints = 1;

    int Nlayers;
    double power = 2.0;
    double radius = 1000;
    double threshold = 0.1;

    /*!
     * \brief sci_type gets
     * * 0 -> FULL
     * * 1 -> HOR
     * * 2 -> VERT
     */
    SCI_TYPE sci_type;
    SCI_METHOD sci_method;
    //Point<dim> P1;
    //Point<dim> P2;
    //bool points_known;

    //void interp_X1D(double x, int &ind, double &t)const;
    //double interp_V1D(int ind, double t)const;
    //double interp_V1D_stratified(double z, double t, int ind)const;

    //double interp3D(Point<dim> p)const;

};

template<int dim>
ScatterInterp<dim>::ScatterInterp(){
    Ndata = 0;
    Npnts = 0;
    //points_known = false;
}

/*
template <int dim>
void ScatterInterp<dim>::set_edge_points(Point<dim> a, Point<dim> b){
    if (sci_type == 2){
        P1 = a;
        P2 = b;
        points_known = true;
    }
    else{
        std::cerr << "You tried to assign points on SCI_TYPE " << sci_type << " and not stratified " << std::endl;
    }
}
*/

template <int dim>
void ScatterInterp<dim>::get_data(std::string filename){
    std::ifstream  datafile(filename.c_str());
    if (!datafile.good()){
        std::cerr << "Can't open " << filename << std::endl;
        return;
    }
    else{
        int Nleafs = 10;
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
                std::cout << "Unkown interpolation type. Valid options are FULL, HOR, VERT" << std::endl;
        }

        {// Read interpolation method
            datafile.getline(buffer, 512);
            std::istringstream inp(buffer);
            std::string temp;
            inp >> temp;
            if (temp == "LINEAR"){
                sci_method = SCI_METHOD::LINEAR;
                //Stratified = true;
                //interpolated = false;
            }
            else if (temp == "NEAREST"){
                sci_method = SCI_METHOD::NEAREST;
                //Stratified = false;
                //interpolated = false;
            }
            else
                std::cout << "Unknown interpolation style. Valid options are STRATIFIED or SIMPLE" << std::endl;

        }

        {//Read number of points and number of data
            datafile.getline(buffer,512);
            std::istringstream inp(buffer);
            inp >> Npnts;
            inp >> Ndata;
            if (Ndata == 1)
                Nlayers = 1;
            if (sci_method == SCI_METHOD::LINEAR){
                inp >> power;
                inp >> radius;
                radius = radius*radius;
                inp >> threshold;
                inp >> Nleafs;
            }
        }
        {// Read the actual data
            double x, y, v;
            for (unsigned int i = 0; i < Npnts; ++i){
                PointVector pv;
                datafile.getline(buffer, 512);
                std::istringstream inp(buffer);
                /*if ((dim == 2 && sci_type == 1) || // 2D horizontal interpolation
                    (dim == 2 && (Stratified || interpolated)  ) || // 2D stratified interpolation
                    (dim == 3 && sci_type == 2) //3D vertical interpolation
                    )
                {
                    inp >> x;
                    X_1D.push_back(x);
                    std::vector<double> temp;
                    for (unsigned int j = 0; j < Ndata; ++j){
                        inp >> v;
                        temp.push_back(v);
                    }
                    // Here we need to store the values
                    V_1D.push_back(temp);
                }*/
                //else{
                inp >> x;
                pv.x = x;
                if (dim == 3){
                    inp >> y;
                    pv.y = y;
                }
                else{
                    pv.y = 0;
                }
                pv.values.resize(Ndata);
                //ine_Point2 p(x, y);
                //T.insert(p);
                for (unsigned int j = 0; j < Ndata; ++j){
                    inp >> v;
                    pv.values[j] = v;
                    //ine_Coord_type ct(v);
                    //function_values[j].insert(std::make_pair(p,ct));
                }
                interpCloud.pts.push_back(pv);
                //}
            }

            //if ((dim == 2 && Stratified) || (dim == 3 && sci_type == 2)){
            //    function_values.clear();
            //}
        }// Read data block
        //pointVector_kd_tree temp_index(2, interpCloud, nanoflann::KDTreeSingleIndexAdaptorParams(Nleafs));

        //temp_index.buildIndex();
        //std::vector<std::pair<size_t,double> >   ret_matches;
        //nanoflann::SearchParams params;
        //params.sorted = false;
        //double query_pt[2] = { 550, 0};
        //size_t nMatches = temp_index.radiusSearch(&query_pt[0],radius, ret_matches, params);

        //interpIndex = new pointVector_kd_tree(2, interpCloud, nanoflann::KDTreeSingleIndexAdaptorParams(Nleafs));
        interpIndex = std::shared_ptr<pointVector_kd_tree>(
                new pointVector_kd_tree(2, interpCloud,
                                        nanoflann::KDTreeSingleIndexAdaptorParams(Nleafs))
                );
        interpIndex->buildIndex();

        //double tmp = interpolate(Point<dim>(550,0));
        //std::cout << tmp << std::endl;
        //nMatches = interpIndex->radiusSearch(&query_pt[0],radius, ret_matches, params);
        //bool stop_here = true;

    }
}
/*
template <int dim>
double ScatterInterp<dim>::interpolate(Point<dim> point)const{    
    if (dim == 3){
        return interp3D(point);
        if (sci_type == 0){// FULL 3D INTERPOLATION
            Point<3> pp;
            pp[0] = point[0];
            pp[1] = point[1];
            pp[2] = point[2];
            return scatter_2D_interpolation(T, function_values, pp);
        }
        else if (sci_type == 1){// HORIZONTAL 3D INTERPOLATION
            Point<3> pp;
            pp[0] = point[0];
            pp[1] = point[1];
            pp[2] = 0;
            return scatter_2D_interpolation(T, function_values, pp);
        }
        else if (sci_type == 2){// VERTICAL INTERPOLATION
            if (!points_known){
                std::cerr << "You attempt to use VERT interpolation but the two points are not known" << std::endl;
                return -9999.9;
            }
            else{
                // find the distance from the first point
                double xx = distance_on_2D_line(P1[0], P1[1], P2[0], P2[1], point[0], point[1]);
                double t;
                int ind;
                interp_X1D(xx, ind, t);
                if (!Stratified){
                    return interp_V1D(ind, t);
                }
                else{
                    return interp_V1D_stratified(point[2], t, ind);
                }
            }
        }
    }
    else if (dim == 2){
        if (sci_type == 0){
            if (!Stratified){
                Point<3> pp;
                pp[0] = point[0];
                pp[1] = point[1];
                pp[2] = 0;
                return scatter_2D_interpolation(T, function_values, pp);
            }
            else{
                double t;
                int ind;
                interp_X1D(point[0], ind, t);
                return interp_V1D_stratified(point[1], t, ind);
            }
        }
        else if (sci_type == 1){
            double t;
            int ind;
            interp_X1D(point[0], ind, t);
            return interp_V1D(ind, t);
        }
        else if (sci_type == 2){
            std::cerr << "Vertical interpolation in 2D is not yet implemented" << std::endl;
            return 0;
        }
    }
    return 0;
}

template <int dim>
void ScatterInterp<dim>::interp_X1D(double x, int &ind, double &t)const{
    t = -9999;
    if (x <= X_1D[0]){
        ind = 0;
        t = 0.0;
    }
    else if (x >= X_1D[X_1D.size()-1]){
        ind = X_1D.size()-2;
        t = 1.0;
    }
    else{
        for (unsigned int i = 0; i < X_1D.size()-1; ++i){
            if (x >= X_1D[i] && x <= X_1D[i+1]){
                t = (x - X_1D[i]) / (X_1D[i+1] - X_1D[i]);
                ind = static_cast<int>(i);
                break;
            }
        }
    }
}

template <int dim>
double ScatterInterp<dim>::interp_V1D(int ind, double t)const{
    return V_1D[ind][0]*(1-t) + V_1D[ind+1][0]*t;
}

template <int dim>
double ScatterInterp<dim>::interp_V1D_stratified(double z, double t, int ind)const{
    std::vector<double> v;
    std::vector<double> el;
    bool push_v = true;
    double v_temp;
    for (unsigned int i = 0; i < Ndata; ++i){
        if (t < -9990){
            v_temp = V_1D[ind][i];
        }
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


    if (z >= el[0])
        return v[0];
    else if (z <= el[el.size()-1])
        return v[v.size()-1];
    else{
        for (unsigned int i = 0; i < el.size()-1; ++i){
            if (z >= el[i+1] && z <= el[i]){
                double u = (z - el[i+1])/(el[i] - el[i+1]);
                return v[i+1]*(1-u) + v[i]*u;
            }
        }
    }
    return 0;
}*/

template<int dim>
double ScatterInterp<dim>::interpolate(Point<dim> p)const{

    double query_pt[2];
    query_pt[0] = p[0];
    if (dim == 3){
        query_pt[1] = p[1];
    }
    else if(dim == 2){
        query_pt[1] = 0;
    }
    std::vector<std::pair<size_t,double> > ret_matches;
    nanoflann::SearchParams params;
    params.sorted = false;

    std::vector<size_t>   ret_index(NinterpPoints);
    std::vector<double> out_dist_sqr(NinterpPoints);
    size_t num_results;
    if (sci_method == SCI_METHOD::NEAREST){
        num_results = interpIndex->knnSearch(&query_pt[0],
                                             NinterpPoints,
                                             &ret_index[0],
                                             &out_dist_sqr[0]);
    }
    else if (sci_method == SCI_METHOD::LINEAR){
        //num_results = interpIndex->knnSearch(&query_pt[0],
        //                                     3,
        //                                     &ret_index[0],
        //                                     &out_dist_sqr[0]);
        num_results = interpIndex->radiusSearch(&query_pt[0],radius,ret_matches,params);
    }


    if (Nlayers == 1){
        if (sci_method == SCI_METHOD::NEAREST){
            return interpCloud.pts[ret_index[0]].values[0];
        }
        else{
            std::vector<double> values, distances;
            for(size_t i = 0; i < num_results; ++i){
                values.push_back(interpCloud.pts[ret_matches[i].first].values[0]);
                distances.push_back(std::sqrt(ret_matches[i].second));
            }
            return IDWinterp(values, distances, power, threshold);
        }
    }
    else{
        std::vector< std::vector<double> > values;
        std::vector<double> distances, interpRes;
        if (sci_method == SCI_METHOD::NEAREST){
            interpRes = interpCloud.pts[ret_index[0]].values;
        }
        else if (sci_method == SCI_METHOD::LINEAR){
            for(size_t i = 0; i < num_results; ++i){
                values.push_back(interpCloud.pts[ret_matches[i].first].values);
                distances.push_back(std::sqrt(ret_matches[i].second));
            }
            interpRes = IDWinterp(values,distances,power,threshold);
        }


        int vidx = 0;
        int lidx = 1;
        //double v1, l1;
        for (int i = 0; i < Nlayers; ++i) {
            //l1 = interpRes[lidx];
            if (i == 0 && p[2] <= interpRes[lidx]){
                return interpRes[vidx];
            }
            else if (i == Nlayers - 1 && p[2] >= interpRes[lidx]){
                if (sci_method == SCI_METHOD::LINEAR)
                    return interpRes[vidx];
                else
                    return interpRes[vidx+1];
            }
            else if (p[2] >= interpRes[lidx] && p[2] <= interpRes[lidx+1] ){
                if (sci_method == SCI_METHOD::LINEAR){
                    double t = (p[2] - interpRes[lidx+1])/(interpRes[lidx] - interpRes[lidx+1]);
                    std::cout << "This is not checked" << std::endl;
                    return t*interpRes[lidx] + (1-t)*interpRes[lidx+1];
                }
                else{
                    return interpRes[vidx];
                }
            }
            vidx = vidx + 2;
            lidx = lidx + 2;
        }
    }
}

#endif // SCATTERINTERP_H
