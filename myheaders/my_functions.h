#ifndef MY_FUNCTIONS_H
#define MY_FUNCTIONS_H

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

#include "interpinterface.h"

using namespace dealii;

/*!
 * \brief The MyFunction class is a function that inherits from the dealii::Function<dim> and used as the
 * communicator between the deal and the interpolation functions developed here. This class returns scalar interpolations
 */
template <int dim, int griddim>
class MyFunction : public Function<dim>{
public:

    /*!
     * \brief MyFunction is the constructor which initialize the interpolation interface variable
     * \param grid_in is the interpolation interface
     */
    MyFunction(InterpInterface<griddim>& grid_in);

    //! This overrides the value method. All it does is to call the InterpInterface#interpolate method
    virtual double value (const Point<dim> &point,
                          const unsigned int component = 0)const;

     //! Overrides the value list which simply loops through the points and calls the value method
    virtual void value_list(const std::vector<Point<dim> >	&points,
                            std::vector<double>             &values,
                            const unsigned int              component = 0)const;
private:
    const InterpInterface<griddim>&	interpolant;
};

template<int dim, int griddim>
MyFunction<dim, griddim>::MyFunction(InterpInterface<griddim>& grid_in)
    :
    Function<dim>(),
    interpolant(grid_in)
{if (dim < griddim){std::cerr << "dim must be greater or equal to griddim" << std::endl;}}


template <int dim, int griddim>
double MyFunction<dim, griddim>::value(const Point<dim> &point,
                          const unsigned int component)const{
    //std::cout << point << std::endl;
    Point<griddim> v;
    for (unsigned int i = 0; i < griddim; ++i)
            v[i] = point[i];
    double val = interpolant.interpolate(v);
    //if (abs(point[0] - 0) < 1 && abs(point[1]-2500)<1)
    //    std::cout << point << " val " << val << std::endl;
    return val;
}

template <int dim, int griddim>
void MyFunction<dim, griddim>::value_list(const std::vector<Point<dim> >	&points,
                                          std::vector<double>               &values,
                                          const unsigned int                component)const{
    for (unsigned int i = 0; i < points.size(); ++i)
        values[i] = MyFunction::value(points[i]);
}

/*!
 * \brief The MyTensorFunction class is a function that inherits from the dealii::TensorFunction<dim> and used as the
 * communicator between the deal and the interpolation functions developed here.
 * This class returns Tensor interpolations. At the moment the only use of this class is to interpolate the hydraulic conductivity,
 * and that's explains why we use K symbols below
 */
template<int dim>
class MyTensorFunction : public TensorFunction<2,dim>{
public:

    /*!
     * \brief MyTensorFunction This version of constructor is used when the aquifer is isotropic (KX==KY==KZ)
     * \param KX_in
     */
    MyTensorFunction(InterpInterface<dim>& KX_in);

    /*!
     * \brief MyTensorFunction This version of constructor is used for aquifer that are isotropic in x-y (KX == KY != KZ)
     * \param KX_in
     * \param KZ_in
     */
    MyTensorFunction(InterpInterface<dim>& KX_in,
                    InterpInterface<dim>& KZ_in);

    /*!
     * \brief MyTensorFunction This version of contructor is used for anisotropic aquifers (KX != KY!= KZ)
     * \param KX_in
     * \param KY_in
     * \param KZ_in
     */
    MyTensorFunction(InterpInterface<dim>& KX_in,
                     InterpInterface<dim>& KY_in,
                     InterpInterface<dim>& KZ_in);

    //! This overrides the value function
    virtual Tensor<2,dim> value (const Point<dim> &point,
                                 const unsigned int component = 0)const;

    //! This overrides the value_list function
    virtual void value_list(const std::vector<Point<dim> >	&points,
                            std::vector<Tensor<2,dim>> 		&values,
                            const unsigned int              component = 0)const;
private:
    //! KX hydraulic conductivity interpolation function
    const InterpInterface<dim>&	KX;

    //! KY hydraulic conductivity interpolation function
    const InterpInterface<dim>&	KY;

    //! KZ hydraulic conductivity interpolation function
    const InterpInterface<dim>&	KZ;

    //! this is set to true for the different interpolation functions that are in use
    std::vector<bool> useit;
};

template<int dim>
MyTensorFunction<dim>::MyTensorFunction(InterpInterface<dim>& KX_in)
    :
    KX(KX_in),
    KY(KX_in),
    KZ(KX_in)
{
    useit.clear();
    useit.resize(3,false);
    useit[0] = true;
}

template<int dim>
MyTensorFunction<dim>::MyTensorFunction(InterpInterface<dim>& KX_in,
                                        InterpInterface<dim>& KZ_in)
    :
    KX(KX_in),
    KY(KX_in),
    KZ(KZ_in)
{
    useit.clear();
    useit.resize(3,true);
    useit[1] = false;
}

template<int dim>
MyTensorFunction<dim>::MyTensorFunction(InterpInterface<dim>& KX_in,
                                        InterpInterface<dim>& KY_in,
                                        InterpInterface<dim>& KZ_in)
    :
    KX(KX_in),
    KY(KY_in),
    KZ(KZ_in)
{
    if (dim == 2)
        std::cerr << "This constructor should be used only for 3D" << std::endl;
    useit.clear();
    useit.resize(3,true);
}

template<int dim>
Tensor<2,dim> MyTensorFunction<dim>::value (const Point<dim> &point,
                                            const unsigned int component)const{
    Tensor<2,dim> value;
    value[0][0] = KX.interpolate(point);

    if (dim  == 2){
        if (useit[2])
            value[1][1] = KZ.interpolate(point);
        else
            value[1][1] = value[0][0];
        value[0][1] = 0.0; // Kxz
        value[1][0] = 0.0; //Kzx
    }
    else if (dim == 3){
        if (useit[1])
            value[1][1] = KY.interpolate(point);
        else
            value[1][1] = value[0][0];

        if (useit[2])
            value[2][2] = KZ.interpolate(point);
        else
            value[2][2] = value[0][0];

        value[0][1] = 0; value[0][2] = 0;
        value[1][0] = 0; value[1][2] = 0;
        value[2][0] = 0; value[2][1] = 0;
    }
    return value;
}

template <int dim>
void MyTensorFunction<dim>::value_list(const std::vector<Point<dim> >	&points,
                                       std::vector<Tensor<2,dim>>       &values,
                                       const unsigned int               component) const{

    for (unsigned int i=0; i<points.size(); ++i)
        values[i] = MyTensorFunction::value(points[i]);
}

#endif // MY_FUNCTIONS_H
