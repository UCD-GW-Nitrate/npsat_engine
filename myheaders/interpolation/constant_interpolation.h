//
// Created by giorgk on 6/23/26.
//

#ifndef CONSTANT_INTERPOLATION_H
#define CONSTANT_INTERPOLATION_H

#include "interp_helpers.h"
#include "../TimeSeries.h"

namespace npsat_flow{
        using namespace dealii;

    /*!
    * \brief Spatially constant interpolant with time-dependent values.
    *
    * The interpolant is constant in space over its assigned 2D region.
    * The value may vary in time through the internal TimeSeriesData object.
    *
    * The input file is read directly by:
    *
    * \code
    * TimeSeriesData<double>::read_data(filename, comm)
    * \endcode
    *
    * Expected data layout:
    *
    * \code
    * value_t0 value_t1 value_t2 ...
    * \endcode
    *
    * For a spatially constant interpolant, this should normally be a
    * headerless one-row time series (dimensions are inferred by the reader). If the value is also constant in time, use a
    * one-row, one-column file.
    */

    template <int dim>
class ConstantSpatialInterpolant final
    :
    public SpatialInterpolantBase<dim>
    {
    public:
        explicit ConstantSpatialInterpolant(InterpRegion2D<dim> region_in)
        :
        SpatialInterpolantBase<dim>(std::move(region_in))
        {}

        void read_spatial_data(const std::string &filename,
                       MPI_Comm comm) override
        {
            (void)filename;
            (void)comm;
        }

        void read_values(const std::string &filename,const double factor, MPI_Comm comm) override
        {
            values.read_data(filename, comm);
            values.multiply_by_factor(factor);
        }

        double interpolate(const Point<dim> &, unsigned int time_index) const override
        {
            return values.get_value_at_step(0, time_index);
        }

        std::int64_t n_times() const override
        {
            return values.n_times();
        }

    private:
        TimeSeriesData<double> values;
    };


}

#endif //CONSTANT_INTERPOLATION_H
