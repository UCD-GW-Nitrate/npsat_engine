#ifndef INTERPINTERFACE_H
#define INTERPINTERFACE_H

#include <fstream>

#include <deal.II/base/point.h>

#include "constantinterp.h"
#include "helper_functions.h"
#include "scatterinterp.h"
#include "boundaryinterp.h"
#include <gridInterp.h>
#include "boost_functions.h"

using namespace dealii;

template<int dim>
struct RectPoly{
    Point<dim> minP;
    Point<dim> maxP;
    bool isPointIn(double x, double y) const{
         return x >= minP[0] && x <= maxP[0] && y >= minP[1] && y <= maxP[1];
    }
};

template<int dim>
class MultiPolyClass;

//!The InterpInterface class is an umbrella of all available interface function
template <int dim>
class InterpInterface{
public:
    //!An empty constructor that does nothing
    InterpInterface();

    //!Copy constructor
    InterpInterface(const InterpInterface<dim>& Interp_in);

    //!This is the core of the class that switches to the right type and calls
    //! the respective function
    double interpolate(Point<dim>)const;

    //! reads the interpolation data from a file. Currently there are two available options
    //! -A scalar value as string. This option sets a constant value interpolation method
    //! -A filename that contains the interpolation data. The first line of the file must be
    //! SCATTERED or GRIDDED. The format of the remaining data is described in grid_interp#get_data_file
    //! or in ScatterInterp#get_data.
    //! For the GRIDDED option we pass in the same line GRIDDED and name of input file for the GRIDDED
    void get_data(std::string namefile);

    //void set_SCI_EDGE_points(Point<dim> a, Point<dim> b);

    //void copy_from(InterpInterface<dim> interp_in);

    bool is_face_part_of_BND(Point<dim> A, Point<dim> B);
    unsigned int get_type();

private:
    //! The type of interpolation
    //! * 0 -> Constrant interpolation
    //! * 1 -> Scattered interpolation
    //! * 2 -> Boundary Line interpolation
    //! * 3 -> Gridded interpolation
    //! * 4 -> Multi polygon
    //! * 5 -> Multi Rectangle
    unsigned int TYPE;

    unsigned int Npoly;

    //! Constant interpolation function
    std::vector<ConstInterp<dim>> CNI;

    //! Container for scattered interpolation data
    std::vector<ScatterInterp<dim>> SCI;
    //ScatterInterp<dim> SCI;

    //! Container for boundary line interpolation
    std::vector<BoundaryInterp<dim>> BND_LINE;

    //! Container for gridded interpolation
    std::vector<GRID_INTERP::interp<dim>> GRD;

    std::vector<boost_polygon> polygons;

    std::vector<RectPoly<dim> > rectpolys;

    /**
     * The length of the vector is Npoly.
     * Each element of the vector is a pair between
     * the interpolation type and the index in the corresponding vector
     * Type ids:
     * 0 -> Constant
     * 1 -> Scattered
     * 2 -> Gridded
     *
     */
    std::vector<std::pair<int,int>> PolyInterpMap;



};

template <int dim>
InterpInterface<dim>::InterpInterface(){}

template <int dim>
InterpInterface<dim>::InterpInterface(const InterpInterface<dim>& Interp_in)
    :
    TYPE(Interp_in.TYPE),
    Npoly(Interp_in.Npoly),
    CNI(Interp_in.CNI),
    SCI(Interp_in.SCI),
    BND_LINE(Interp_in.BND_LINE),
    GRD(Interp_in.GRD),
    polygons(Interp_in.polygons),
    rectpolys(Interp_in.rectpolys),
    PolyInterpMap(Interp_in.PolyInterpMap)
{}


template <int dim>
void InterpInterface<dim>::get_data(std::string namefile){
    if (is_input_a_scalar(namefile)){
        double value = dealii::Utilities::string_to_double(namefile);
        CNI.resize(CNI.size()+1);
        CNI[CNI.size()-1].set_value(value);
        TYPE = 0;
        Npoly = 0;
    }else{
        std::ifstream  datafile(namefile.c_str());
        if (!datafile.good()){
            std::cerr << "Can't open " << namefile << std::endl;
        }
        else{
            // read the first line to determine what type of interpolant is
            char buffer[512];
            datafile.getline(buffer,512);
            std::istringstream inp(buffer);
            std::string type_temp;
            inp >> type_temp;
            if (type_temp == "SCATTERED"){
                TYPE = 1;
                datafile.close();
                SCI.resize(SCI.size()+1);
                SCI[SCI.size()-1].get_data(namefile);
                //ScatterInterp<dim> tmp;
                //tmp.get_data(namefile);
                //SCI.push_back(tmp);
            }
            else if(type_temp == "BOUNDARY_LINE"){
                TYPE = 2;
                datafile.close();
                BND_LINE.resize(BND_LINE.size()+1);
                BND_LINE[BND_LINE.size()-1].get_data(namefile);
                //BoundaryInterp<dim> tmp;
                //tmp.get_data(namefile);
                //BOUNDARY_LINE.push_back(tmp);
            }
            else if (type_temp.compare("GRIDDED") == 0 ){
                TYPE = 3;
                datafile.close();
                GRD.resize(GRD.size()+1);
                GRD[GRD.size()-1].getDataFromFile(namefile);
            }
            else if (type_temp.compare("MULTIPOLY") == 0 || type_temp.compare("MULTIRECT") == 0){
                if (type_temp.compare("MULTIPOLY") == 0)
                    TYPE = 4;
                else if (type_temp.compare("MULTIRECT") == 0)
                    TYPE = 5;

                std::string line;
                getline(datafile, line);
                {// Get the number of polygons
                    std::istringstream inp(line.c_str());
                    inp >> Npoly;
                }
                for(unsigned int ipoly = 0; ipoly < Npoly; ++ipoly){
                    getline(datafile, line);
                    std::istringstream inp(line.c_str());
                    int N = 0;
                    std::string type;
                    std::string func;
                    if (TYPE == 4)
                        inp >> N;
                    else if (TYPE == 5)
                        N = 1;
                    inp >> type;
                    inp >> func;
                    if (TYPE == 4){
                        std::vector<boost_point> pnts;
                        for (int i = 0; i < N; ++i) {
                            getline(datafile, line);
                            std::istringstream inp1(line.c_str());
                            float x, y;
                            inp1 >> x;
                            inp1 >> y;
                            pnts.push_back(boost_point(x, y));
                        }
                        boost_polygon poly;
                        boost::geometry::assign_points(poly, pnts);
                        boost::geometry::correct(poly);
                        polygons.push_back(poly);
                    }
                    else if (TYPE == 5){
                        getline(datafile, line);
                        std::istringstream inp1(line.c_str());
                        RectPoly<dim> rp;
                        double x1,y1,x2,y2;
                        Point<dim> p1, p2;
                        inp1 >> x1;
                        inp1 >> y1;
                        inp1 >> x2;
                        inp1 >> y2;
                        rp.minP[0] = x1;
                        rp.minP[1] = y1;
                        rp.maxP[0] = x2;
                        rp.maxP[1] = y2;
                        rectpolys.push_back(rp);
                    }

                    if (type.compare("CONST") == 0){
                        double value = dealii::Utilities::string_to_double(func);
                        CNI.resize(CNI.size()+1);
                        CNI[CNI.size()-1].set_value(value);
                        PolyInterpMap.push_back(std::pair<int, int>(0,CNI.size()-1));
                    }
                    else if (type.compare("SCATTERED") == 0){
                        //ScatterInterp<dim> tmp;
                        //tmp.get_data(func);
                        SCI.resize(SCI.size()+1);
                        SCI[SCI.size()-1].get_data(func);
                        PolyInterpMap.push_back(std::pair<int, int>(1,SCI.size()-1));
                    }
                    else if (type.compare("BOUNDARY_LINE") == 0){
                        std::cerr << "I cant think why one would want to split a boundary line" << std::endl;
                    }
                    else if (type.compare("GRIDDED") == 0){
                        GRD.resize(GRD.size()+1);
                        GRD[GRD.size()-1].getDataFromFile(func);
                        PolyInterpMap.push_back(std::pair<int, int>(2,GRD.size()-1));
                    }
                    else{
                        std::cerr << "Unknown interpolation method Under MULTIPOLYGON " << namefile << std::endl;
                    }
                }
            }
            else{
                std::cerr << "Unknown interpolation method on " << namefile << std::endl;
            }
        }
    }
}

template <int dim>
double InterpInterface<dim>::interpolate(Point<dim> p)const{

    if (TYPE == 0){
        return CNI[0].interpolate(p);
    }
    else if (TYPE == 1) {
        return SCI[0].interpolate(p);
    }
    else if (TYPE == 2){
        return BND_LINE[0].interpolate(p);
    }
    else if (TYPE == 3) {
        if (dim == 1)
           return GRD[0].interpolate(p[0]);
        else if (dim == 2)
            return GRD[0].interpolate(p[0], p[1]);
        else if (dim == 3)
            return GRD[0].interpolate(p[0], p[1], p[2]);
    }
    else if (TYPE == 4 || TYPE == 5){
        for (unsigned int i = 0; i < Npoly; ++i){
            // Find the polygon or rectangle that containt the interpolation point
            bool point_found = false;
            if (TYPE == 4){
                if (boost::geometry::within(boost_point(p[0], p[1]),polygons[i]))
                    point_found = true;
            }
            else if (TYPE == 5){
                if (rectpolys[i].isPointIn(p[0], p[1]))
                    point_found = true;
            }

            if (point_found){
                // Find out the interpolation method that corresponds to this polygon
                // and the index in the vector
                if (PolyInterpMap[i].first == 0){
                    return CNI[PolyInterpMap[i].second].interpolate(p);
                }
                else if (PolyInterpMap[i].first == 1){
                    //return SCI.interpolate(p);
                    return SCI[PolyInterpMap[i].second].interpolate(p);
                }
                else if (PolyInterpMap[i].first == 2){
                    if (dim == 1)
                        return GRD[PolyInterpMap[i].second].interpolate(p[0]);
                    else if (dim == 2)
                        return GRD[PolyInterpMap[i].second].interpolate(p[0], p[1]);
                    else if (dim == 3)
                        return GRD[PolyInterpMap[i].second].interpolate(p[0], p[1], p[2]);
                }
                else{
                    std::cerr << "Unknown method for MULTIPOLY interpolation" << std::endl;
                }
                break;
            }
        }
    }
    else{
        std::cerr << "Unknown interpolation method" << std::endl;
    }
    return 0;
}

//template <int dim>
//void InterpInterface<dim>::set_SCI_EDGE_points(Point<dim> a, Point<dim> b){
//    SCI[0].set_edge_points(a,b);
//}

/*
template <int dim>
void InterpInterface<dim>::copy_from(InterpInterface<dim> interp_in){
    TYPE = interp_in.TYPE;
    if (TYPE == 0){
        CNI = interp_in.CNI;
    }
    else if (TYPE == 1){
        SCI = interp_in.SCI;
    }
    else if (TYPE = 3){
        GRD = interp_in.GRD;
    }
}
 */

template <int dim>
bool InterpInterface<dim>::is_face_part_of_BND(Point<dim> A, Point<dim> B){
    if (TYPE == 2){
        return BND_LINE[0].is_face_part_of_BND(A, B);
    }
    else
        return false;
}

template <int dim>
unsigned int InterpInterface<dim>::get_type(){
    return TYPE;
}

#endif // INTERPINTERFACE_H
