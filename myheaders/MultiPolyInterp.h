//
// Created by giorgk on 1/28/21.
//

#ifndef NPSAT_MULTIPOLYINTERP_H
#define NPSAT_MULTIPOLYINTERP_H

#include "boost_functions.h"
#include "interpinterface.h"

using namespace dealii;

template<int dim>
class MultiPolyInterface : public InterpInterface<dim> {
public:
    MultiPolyInterface();

    void get_data(std::string filename);
    double interpolate(Point<dim> p)const;

private:
    int Npoly;
    std::vector<boost_polygon> polygons;
    std::vector<InterpInterface<dim> > interp_func;
};

template<int dim>
MultiPolyInterface<dim>::MultiPolyInterface() {}

template<int dim>
void MultiPolyInterface<dim>::get_data(std::string filename) {
    std::ifstream datafile(filename.c_str());
    if (!datafile.good()) {
        std::cerr << "Can't open " << filename << std::endl;
    } else {
        std::string line;
        getline(datafile, line);
        {// Get the number of polygons
            std::istringstream inp(line.c_str());
            inp >> Npoly;
        }
        getline(datafile, line);
        {// Read the polygon information and the interpolation function
            std::istringstream inp(line.c_str());
            int N;
            std::string func;
            inp >> N;
            inp >> func;
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
            InterpInterface<dim> tmpInterp;
            tmpInterp.get_data(func);
            interp_func.push_back(tmpInterp);
        }
        datafile.close();
    }
}

template<int dim>
double MultiPolyInterface<dim>::interpolate(Point<dim> p) const {
    for (int i = 0; i < Npoly; ++i) {
        if (boost::geometry::within(boost_point(),polygons[i])){
            return interp_func[i].interpolate(p);
        }
    }
    return 0.0;
}


#endif //NPSAT_MULTIPOLYINTERP_H
