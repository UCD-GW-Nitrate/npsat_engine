#ifndef INTERPINTERFACE_H
#define INTERPINTERFACE_H

#include <fstream>

#include <deal.II/base/point.h>

#include "constantinterp.h"
#include "helper_functions.h"
#include "scatterinterp.h"

using namespace dealii;

//!The InterpInterface class is an umbrella of all available interface function
template <int dim>
class InterpInterface{
public:
    //!An empty constructor that does nothing
    InterpInterface();

    //!This is the core of the class that swithes to the right type and calls
    //! the respective function
    double interpolate(Point<dim>)const;

    //! reads the interpolation data from a file. Currently there are two available options
    //! -A scalar value as string. This option sets a constant value interpolation method
    //! -A filename that containts the interpolation data. The first line of the file must be
    //! SCATTERED or GRIDDED. The format of the remaining data is descibed in grid_interp#get_data_file
    //! or in ScatterInterp#get_data.
    void get_data(std::string namefile);

private:
    //! The type of interpolation
    std::string TYPE;

     //! Constant interpolation function
     ConstInterp<dim> CNI;

     //! Container for scattered interpolation data
     ScatterInterp<dim> SCI;
};

template <int dim>
InterpInterface<dim>::InterpInterface(){}

template <int dim>
void InterpInterface<dim>::get_data(std::string namefile){
    if (is_input_a_scalar(namefile)){
        double value = dealii::Utilities::string_to_double(namefile);
        CNI.set_value(value);
        TYPE = "CONST";
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
            inp >> TYPE;
            if (TYPE == "SCATTERED"){
                SCI.get_data(namefile);
            }
            else{
                std::cerr << "Unknown interpolation method on " << namefile << std::endl;
            }
        }
    }
}

template <int dim>
double InterpInterface<dim>::interpolate(Point<dim> p)const{
    if (TYPE == "CONST"){
        return CNI.interpolate(p);
    }
    else if (TYPE == "SCATTERED") {
        return SCI.interpolate(p);
    }
    else if (TYPE == "GRIDDED") {
        std::cerr << "Not Implemented yet" << std::endl;
        return 0;
    }else{
        std::cerr << "Unknown interpolation method" << std::endl;
    }
    return 0;
}

#endif // INTERPINTERFACE_H
