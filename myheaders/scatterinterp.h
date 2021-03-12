#ifndef SCATTERINTERP_H
#define SCATTERINTERP_H

#include <fstream>

#include <deal.II/base/point.h>

//#include "cgal_functions.h"
#include "nanoflann_structures.h"
#include "helper_functions.h"



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
    PointIdCloud interpCloud;
    //PointVectorCloud* ptrCloud= &interpCloud;
    std::shared_ptr<pointid_kd_tree> interpIndex;
    std::vector<std::vector<int>> triangulation;
    std::vector<Point<dim>> Vertices;
    std::vector<std::vector<double>> Data;
    std::vector<double> triangleArea;


    //! This is a map between the triangulation and the values that correspond to each 2D point
    //std::vector<std::map<ine_Point2, ine_Coord_type, ine_Kernel::Less_xy_2> > function_values;

    //! Ndata is the number of values for interpolation. For the STRATIFIED option this number must be equal to (Nlay-1)*2 +1
    unsigned int Ndata;

    //! Npnts is the number of 2D points that the scattered interpolation set contains
    unsigned int Npnts;

    unsigned int Ntris;

    //! X_1D is the vector of x of points for the 1D interpolation of a function y=f(x)
    std::vector<double> X_1D;

    //! V_1D is the vector of vectors y of points for the 1D interpolation of list of functions of the form y_1=f(x), y_2=f(x), y_3=f(x) etc.
    //! It make sense to define more than one data for stratified hydraulic conductivity in 2D. See the discussion above.
    std::vector<std::vector<double>> V_1D;

    //bool Stratified;

    //bool interpolated;
    size_t NinterpPoints;

    int Nlayers;
    //double power = 2.0;
    //double radius = 1000;
    //double threshold = 0.1;

    /*!
     * \brief sci_type gets
     * * 0 -> FULL
     * * 1 -> HOR
     * * 2 -> VERT
     */
    SCI_TYPE sci_type;
    SCI_METHOD sci_methodXY;
    SCI_METHOD sci_methodZ;
    //Point<dim> P1;
    //Point<dim> P2;
    //bool points_known;

    //void interp_X1D(double x, int &ind, double &t)const;
    //double interp_V1D(int ind, double t)const;
    //double interp_V1D_stratified(double z, double t, int ind)const;

    //double interp3D(Point<dim> p)const;

    bool findElemId(Point<dim> p, int& elemId, double& u, double& v, double& w)const;

    std::string namefile;

};

template<int dim>
ScatterInterp<dim>::ScatterInterp(){
    Ndata = 0;
    Npnts = 0;
    NinterpPoints = (dim == 2)? 3: 10;
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
        namefile = filename;
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
                std::cout << "Unknown interpolation type. Valid options are DIM3, DIM2, VERT" << std::endl;
        }


        {// Read interpolation method
            datafile.getline(buffer, 512);
            std::istringstream inp(buffer);
            std::string temp;
            inp >> temp;
            if (temp == "LINEAR"){
                sci_methodXY = SCI_METHOD::LINEAR;
                //Stratified = true;
                //interpolated = false;
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


        {//Read number of points and number of data
            datafile.getline(buffer,512);
            std::istringstream inp(buffer);
            inp >> Npnts;
            inp >> Ndata;
            inp >> Ntris;
            if (Ndata == 1)
                Nlayers = 1;
            else{ // make sure the number of data are compatible with the z interpolation method
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
            //if (sci_methodXY == SCI_METHOD::LINEAR){
            //    inp >> power;
            //    inp >> radius;
            //    radius = radius*radius;
            //    inp >> threshold;
            //    inp >> Nleafs;
            //}
        }
        {// Read the actual data
            Point<dim> p;
            double vtmp;
            for (unsigned int i = 0; i < Npnts; ++i){
                //PointVector pv;
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
                inp >> p[0];
                if (dim == 3){
                    inp >> p[1];
                }
                Vertices.push_back(p);

                std::vector<double> v;
                for (unsigned int j = 0; j < Ndata; ++j){
                    inp >> vtmp;
                    v.push_back(vtmp);
                }
                Data.push_back(v);
            }
        }// Read data block

        {// Read the triangulation
            double bcx, bcy;
            std::vector<int> id(dim);
            PointId pid;
            Point<dim> A, B, C;
            for (unsigned int i = 0; i < Ntris; ++i) {
                bcx = 0;
                bcy = 0;
                datafile.getline(buffer, 512);
                std::istringstream inp(buffer);
                for (int j = 0; j < dim; ++j) {
                    inp >> id[j];
                    bcx += Vertices[id[j]][0];
                    if (dim == 3){
                        bcy += Vertices[id[j]][1];
                        if (j == 0) A = Vertices[id[j]];
                        if (j == 1) B = Vertices[id[j]];
                        if (j == 2) C = Vertices[id[j]];
                    }
                }
                triangulation.push_back(id);
                if (dim == 3){
                    pid.x = bcx/3.0;
                    pid.y = bcy/3.0;
                    double area = triangle_area(A,B,C, true);
                    triangleArea.push_back(area);
                }
                else if (dim == 2){
                    pid.x = bcx/2.0;
                    pid.y = 0;
                }
                pid.id = i;
                interpCloud.pts.push_back(pid);
            }
            interpIndex = std::shared_ptr<pointid_kd_tree>(
                    new pointid_kd_tree (2, interpCloud,
                                            nanoflann::KDTreeSingleIndexAdaptorParams(Nleafs)));
            interpIndex->buildIndex();
        }
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

    double u = 0;
    double v = 0;
    double w = 0;
    int id;
    bool isInElem = findElemId(p, id, u,v,w);
    std::vector<double> interpData;
    for (int i = 0; i < Ndata; ++i){
        double d;
        if (sci_methodXY == SCI_METHOD::LINEAR && isInElem){
            // interpolate the layer values
            if (dim == 2){
                d = Data[triangulation[id][0]][i] * (1 - u)  + Data[triangulation[id][1]][i] * u;
            }
            else if (dim == 3){
                d = Data[triangulation[id][0]][i] * w +
                    Data[triangulation[id][1]][i] * u +
                    Data[triangulation[id][2]][i] * v;
            }
        }
        else{
            // select the values that correspond to nearest node
            d = Data[id][i];
        }
        interpData.push_back(d);
    }

    // So far we have a list of points in the vertical direction
    if (Ndata == 1){
        return interpData[0];
    }
    else{
        int vidx = 0;
        int lidx = 1;
        for (int i = 0; i < Nlayers; ++i) {
            if (i == 0 && p[dim-1] >= interpData[lidx] ){
                return interpData[vidx];
            }
            else if (i == Nlayers - 1 && p[dim-1] <= interpData[lidx]){
                if (sci_methodZ == SCI_METHOD::NEAREST){
                    return interpData[lidx+1];
                }
                else if (sci_methodZ == SCI_METHOD::LINEAR){
                    return interpData[lidx-1];
                }
            }
            else if (p[dim-1] <= interpData[lidx] && p[dim-1] >= interpData[lidx+2]){
                if (sci_methodZ == SCI_METHOD::NEAREST){
                    return interpData[vidx+2];
                }
                else if (sci_methodZ == SCI_METHOD::LINEAR){
                    w = (p[dim-1] - interpData[lidx+2])/(interpData[lidx] - interpData[lidx+2]);
                    return interpData[vidx] * w + interpData[vidx+2] * (1 - w);
                }
            }
            vidx = vidx + 2;
            lidx = lidx + 2;
        }
    }
    std::cout << "The interpolation from " << namefile << "was not successful at point (" << p[0] << "," << p[1] << ")" << std::endl;
    return -9999999.9;
}

template <int dim>
bool ScatterInterp<dim>::findElemId(Point<dim> p, int& elemId, double& u, double& v, double& w)const {
    bool out = false;
    double query_pt[2];
    query_pt[0] = p[0];
    query_pt[1] = p[1];

    elemId = -9;
    std::vector<std::pair<size_t,double> > ret_matches;
    nanoflann::SearchParams params;
    params.sorted = true;
    std::vector<size_t>   ret_index(NinterpPoints);
    std::vector<double> out_dist_sqr(NinterpPoints);
    size_t num_results;
    num_results = interpIndex->knnSearch(&query_pt[0],
                                         NinterpPoints,
                                         &ret_index[0],
                                         &out_dist_sqr[0]);
    std::map<int,int> triangleIds;
    if (sci_methodXY == SCI_METHOD::LINEAR) {
        for (unsigned int i = 0; i < ret_index.size(); ++i) {
            triangleIds.insert(std::pair<int,int>(triangulation[ret_index[i]][0],triangulation[ret_index[i]][0]));
            triangleIds.insert(std::pair<int,int>(triangulation[ret_index[i]][1],triangulation[ret_index[i]][1]));
            if (dim == 2) {
                double x1 = Vertices[triangulation[ret_index[i]][0]][0];
                double x2 = Vertices[triangulation[ret_index[i]][1]][0];
                if (x1 < x2) {
                    if (x1 <= p[0] && x2 >= p[0]) {
                        elemId = ret_index[i];
                        out = true;
                        u = (p[0] - x1)/(x2 - x1);
                        break;
                    }
                } else {
                    if (x2 <= p[0] && x1 >= p[0]) {
                        elemId = ret_index[i];
                        out = true;
                        u = (p[0] - x2)/(x1 - x2);
                        break;
                    }
                }
            } else if (dim == 3) {
                triangleIds.insert(std::pair<int,int>(triangulation[ret_index[i]][2],triangulation[ret_index[i]][2]));
                Point<dim> A, B, C;
                A = Vertices[triangulation[ret_index[i]][0]];
                B = Vertices[triangulation[ret_index[i]][1]];
                C = Vertices[triangulation[ret_index[i]][2]];
                // https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/barycentric-coordinates
                double CAP = triangle_area(C, A, p, true);
                double ABP = triangle_area(A, B, p, true);
                double BCP = triangle_area(B, C, p, true);
                if (std::abs(CAP + ABP + BCP - triangleArea[ret_index[i]]) < 0.01) {
                    u = CAP / triangleArea[ret_index[i]];
                    v = ABP / triangleArea[ret_index[i]];
                    w = BCP / triangleArea[ret_index[i]];
                    //double ww = 1 - u - v;
                    elemId = ret_index[i];
                    out = true;
                    break;
                }
            }
        }
    }
    if (!out){
        if (sci_methodXY == SCI_METHOD::NEAREST) {
            for (unsigned int i = 0; i < ret_index.size(); ++i) {
                triangleIds.insert(std::pair<int, int>(triangulation[ret_index[i]][0], triangulation[ret_index[i]][0]));
                triangleIds.insert(std::pair<int, int>(triangulation[ret_index[i]][0], triangulation[ret_index[i]][1]));
                if (dim == 3)
                    triangleIds.insert(
                            std::pair<int, int>(triangulation[ret_index[i]][0], triangulation[ret_index[i]][2]));
            }
        }
        // if the point is outside of the triangulation return the nearest node instead of element id
        std::map<int,int>::iterator it;
        double mindst = 999999999999;
        for (it = triangleIds.begin(); it != triangleIds.end(); ++it){
            double dst = (dim == 2)? distance_2_points(p[0], 0.0, Vertices[it->first][0], 0.0):
                         distance_2_points(p[0], p[1], Vertices[it->first][0], Vertices[it->first][1]);
            if (dst < mindst){
                elemId = it->first;
                mindst = dst;
            }
        }
    }
    return out;
}

#endif // SCATTERINTERP_H
