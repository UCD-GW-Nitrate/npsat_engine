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
    double interpolate(Point<dim> p)const override;

private:
    int Npoly;
    std::vector<boost_polygon> polygons;
    std::vector<InterpInterface<dim> > interp_func;
    std::vector<std::vector<Point<dim> > > RectBound;
};

template<int dim>
MultiPolyInterface<dim>::MultiPolyInterface() {}

template<int dim>
void MultiPolyInterface<dim>::get_data(std::string filename) {
    std::ifstream datafile(filename.c_str());
    if (!datafile.good()) {// Then it may be a scalar
        // If not the error will be captured on the InterpInterface
        Npoly = 0;
        InterpInterface<dim> tmpInterp;
        tmpInterp.get_data(filename);
        interp_func.push_back(tmpInterp);
        return;
    } else {
        std::string Interpolation_type;
        {// Is its a file but the file its not MULTIPOLY or MULTIRECT then
            // switch to one polygon interpolation
            char buffer[512];
            datafile.getline(buffer,512);
            std::istringstream inp(buffer);
            inp >> Interpolation_type;
            if (!(Interpolation_type == "MULTIPOLY"  || Interpolation_type == "MULTIRECT")){
                Npoly = 0;
                InterpInterface<dim> tmpInterp;
                tmpInterp.get_data(filename);
                interp_func.push_back(tmpInterp);
                datafile.close();
                return;
            }
        }
        std::string line;
        getline(datafile, line);
        {// Get the number of polygons
            std::istringstream inp(line.c_str());
            inp >> Npoly;
        }
        getline(datafile, line);
        if (Interpolation_type == "MULTIPOLY"){// Read the polygon information and the interpolation function
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
        else if (Interpolation_type == "MULTIRECT"){

        }
        datafile.close();
    }
}

template<int dim>
double MultiPolyInterface<dim>::interpolate(Point<dim> p) const {
    if (Npoly == 0 && interp_func.size() > 0){
        return interp_func[0].interpolate(p);
    }
    for (int i = 0; i < Npoly; ++i) {
        if (boost::geometry::within(boost_point(),polygons[i])){
            return interp_func[i].interpolate(p);
        }
    }
    return 0.0;
}


#endif //NPSAT_MULTIPOLYINTERP_H
