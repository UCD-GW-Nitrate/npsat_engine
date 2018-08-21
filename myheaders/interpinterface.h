#ifndef INTERPINTERFACE_H
#define INTERPINTERFACE_H

#include <fstream>

#include <deal.II/base/point.h>

#include "constantinterp.h"
#include "helper_functions.h"
#include "scatterinterp.h"
#include "boundaryinterp.h"

using namespace dealii;

//!The InterpInterface class is an umbrella of all available interface function
template <int dim>
class InterpInterface{
public:
    //!An empty constructor that does nothing
    InterpInterface();

    //!Copy constructor
    InterpInterface(const InterpInterface<dim>& Interp_in);

    //!This is the core of the class that swithes to the right type and calls
    //! the respective function
    double interpolate(Point<dim>)const;

    //! reads the interpolation data from a file. Currently there are two available options
    //! -A scalar value as string. This option sets a constant value interpolation method
    //! -A filename that containts the interpolation data. The first line of the file must be
    //! SCATTERED or GRIDDED. The format of the remaining data is descibed in grid_interp#get_data_file
    //! or in ScatterInterp#get_data.
    void get_data(std::string namefile);

    void set_SCI_EDGE_points(Point<dim> a, Point<dim> b);

    void copy_from(InterpInterface<dim> interp_in);

    bool is_face_part_of_BND(Point<dim> A, Point<dim> B);

private:
    //! The type of interpolation
    //! * 0 -> Constrant interpolation
    //! * 1 -> Scattered interpolation
    //! * 2 -> Boundary Line interpolation
    unsigned int TYPE;

     //! Constant interpolation function
     ConstInterp<dim> CNI;

     //! Container for scattered interpolation data
     ScatterInterp<dim> SCI;

     //! Container for boundary line interpolation
     BoundaryInterp<dim> BND_LINE;
};

template <int dim>
InterpInterface<dim>::InterpInterface(){}

template <int dim>
InterpInterface<dim>::InterpInterface(const InterpInterface<dim>& Interp_in)
    :
      TYPE(Interp_in.TYPE),
      CNI(Interp_in.CNI),
      SCI(Interp_in.SCI),
      BND_LINE(Interp_in.BND_LINE)
{}


template <int dim>
void InterpInterface<dim>::get_data(std::string namefile){
    if (is_input_a_scalar(namefile)){
        double value = dealii::Utilities::string_to_double(namefile);
        CNI.set_value(value);
        TYPE = 0;
    }else{
        std::ifstream  datafile(namefile.c_str());
        if (!datafile.good()){
            std::cerr << "Can't open " << namefile << std::endl;
        }
        else{
            {
                //ConstInterp<dim> cni_temp;
                CNI = ConstInterp<dim>();
            }

            // read the first line to determine what type of interpolant is
            char buffer[512];
            datafile.getline(buffer,512);
            std::istringstream inp(buffer);
            std::string type_temp;
            inp >> type_temp;
            if (type_temp == "SCATTERED"){
                TYPE = 1;
                SCI.get_data(namefile);
            }
            else if(type_temp == "BOUNDARY_LINE"){
                TYPE = 2;
                BND_LINE.get_data(namefile);
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
        return CNI.interpolate(p);
    }
    else if (TYPE == 1) {
        return SCI.interpolate(p);
    }
    else if (TYPE == 2){
        return BND_LINE.interpolate(p);
    }
    else if (TYPE == 3) {
        std::cerr << "Not Implemented yet" << std::endl;
        return 0;
    }else{
        std::cerr << "Unknown interpolation method" << std::endl;
    }
    return 0;
}

template <int dim>
void InterpInterface<dim>::set_SCI_EDGE_points(Point<dim> a, Point<dim> b){
    SCI.set_edge_points(a,b);
}

template <int dim>
void InterpInterface<dim>::copy_from(InterpInterface<dim> interp_in){
    TYPE = interp_in.TYPE;
    if (TYPE == 0){
        CNI = interp_in.CNI;
    }
    else if (TYPE == 1){
        SCI = interp_in.SCI;
    }
}

template <int dim>
bool InterpInterface<dim>::is_face_part_of_BND(Point<dim> A, Point<dim> B){
    if (TYPE == 2){
        return BND_LINE.is_face_part_of_BND(A, B);
    }
    else
        return false;
}

#endif // INTERPINTERFACE_H
