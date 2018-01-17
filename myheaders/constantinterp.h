#ifndef CONSTANTINTERP_H
#define CONSTANTINTERP_H

#include <deal.II/base/point.h>

using namespace dealii;

//! The ConstInterp method defines a constant value function.
//! The reason for this function is to allow a unified framework for
//! the interpolation functions
template <int dim>
class ConstInterp{
public:
    //! The default empty constructor. Initialize a large negative dummy value
    ConstInterp();

    //!This constructor initialize the constant value
    ConstInterp(double value);

    //! Returns the constant value
    double interpolate(Point<dim> p)const;

    //! A method to set the constant value
    void set_value(double val);

private:
    //! The constant value
    double value;
};

template <int dim>
ConstInterp<dim>::ConstInterp(){value = -987654321.0;}

template <int dim>
ConstInterp<dim>::ConstInterp(double val){
    value = val;
}

template <int dim>
void ConstInterp<dim>::set_value(double val){
    value = val;
}

template <int dim>
double ConstInterp<dim>::interpolate(Point<dim> p)const{
    //std::cout <<"in CNI " << value << std::endl;
    p[0] = p[0];// something to suppress the unused warning
    return value;
}

#endif // CONSTANTINTERP_H
