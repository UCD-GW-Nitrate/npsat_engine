#ifndef CONDUCTIVITY_FUNCTION_H
#define CONDUCTIVITY_FUNCTION_H
#include <deal.II/base/tensor_function.h>
#include "interpolation/interpolation_function.h"
using namespace dealii;
template<int dim>
class ConductivityFunction : public TensorFunction<2,dim>{
public:

    /*!
     * \brief ConductivityFunction This version of constructor is used when the aquifer is isotropic (KX==KY==KZ)
     * \param KX_in
     */
    ConductivityFunction(std::shared_ptr<const npsat_flow::InterpInterface<dim>> KX_in);

    /*!
     * \brief ConductivityFunction This version of constructor is used for aquifer that are isotropic in x-y (KX == KY != KZ)
     * \param KX_in
     * \param KZ_in
     */
    ConductivityFunction(std::shared_ptr<const npsat_flow::InterpInterface<dim>> KX_in,
                    std::shared_ptr<const npsat_flow::InterpInterface<dim>> KZ_in);

    /*!
     * \brief ConductivityFunction This version of contructor is used for anisotropic aquifers (KX != KY!= KZ)
     * \param KX_in
     * \param KY_in
     * \param KZ_in
     */
    ConductivityFunction(std::shared_ptr<const npsat_flow::InterpInterface<dim>> KX_in,
                     std::shared_ptr<const npsat_flow::InterpInterface<dim>> KY_in,
                     std::shared_ptr<const npsat_flow::InterpInterface<dim>> KZ_in);

    //! This overrides the value function
    virtual Tensor<2,dim> value (const Point<dim> &point,
                                 const unsigned int component = 0)const;

    //! This overrides the value_list function
    virtual void value_list(const std::vector<Point<dim> >	&points,
                            std::vector<Tensor<2,dim>> 		&values,
                            const unsigned int              component = 0)const;
    void set_time_index(unsigned int index) { time_index = index; }
private:
    unsigned int time_index = 0;
    //! KX hydraulic conductivity interpolation function
    std::shared_ptr<const npsat_flow::InterpInterface<dim>>	KX;

    //! KY hydraulic conductivity interpolation function
    std::shared_ptr<const npsat_flow::InterpInterface<dim>>	KY;

    //! KZ hydraulic conductivity interpolation function
    std::shared_ptr<const npsat_flow::InterpInterface<dim>>	KZ;

    //! this is set to true for the different interpolation functions that are in use
    std::vector<bool> useit;
};

template<int dim>
ConductivityFunction<dim>::ConductivityFunction(std::shared_ptr<const npsat_flow::InterpInterface<dim>> KX_in)
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
ConductivityFunction<dim>::ConductivityFunction(std::shared_ptr<const npsat_flow::InterpInterface<dim>> KX_in,
                                        std::shared_ptr<const npsat_flow::InterpInterface<dim>> KZ_in)
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
ConductivityFunction<dim>::ConductivityFunction(std::shared_ptr<const npsat_flow::InterpInterface<dim>> KX_in,
                                        std::shared_ptr<const npsat_flow::InterpInterface<dim>> KY_in,
                                        std::shared_ptr<const npsat_flow::InterpInterface<dim>> KZ_in)
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
Tensor<2,dim> ConductivityFunction<dim>::value (const Point<dim> &point,
                                            const unsigned int /*component*/)const{
    //(void)component; // does nothing. its just removes the warning

    Tensor<2,dim> value;
    value[0][0] = KX->interpolate(point, time_index);

    if (dim  == 2){
        if (useit[2])
            value[1][1] = KZ->interpolate(point, time_index);
        else
            value[1][1] = value[0][0];
        value[0][1] = 0.0; // Kxz
        value[1][0] = 0.0; //Kzx
    }
    else if (dim == 3){
        if (useit[1])
            value[1][1] = KY->interpolate(point, time_index);
        else
            value[1][1] = value[0][0];

        if (useit[2])
            value[2][2] = KZ->interpolate(point, time_index);
        else
            value[2][2] = value[0][0];

        value[0][1] = 0; value[0][2] = 0;
        value[1][0] = 0; value[1][2] = 0;
        value[2][0] = 0; value[2][1] = 0;
    }
    return value;
}

template <int dim>
void ConductivityFunction<dim>::value_list(const std::vector<Point<dim> >	&points,
                                       std::vector<Tensor<2,dim>>       &values,
                                       const unsigned int               /*component*/) const{

    //(void)component; // does nothing. its just removes the warning

    for (unsigned int i=0; i<points.size(); ++i)
        values[i] = ConductivityFunction::value(points[i]);
}

#endif // CONDUCTIVITY_FUNCTION_H
