//
// Created by giorgk on 6/23/26.
//

#ifndef INTERPOLATION_FUNCTION_H
#define INTERPOLATION_FUNCTION_H

#include <deal.II/base/function.h>
#include "interp_interface.h"

namespace npsat_flow{
    using namespace dealii;

    template <int dim>
    class InterpolationFunction : public Function<dim>{
    public:
        InterpolationFunction()
              :
              Function<dim>(1)
        {}

        explicit InterpolationFunction(
            std::shared_ptr<const InterpInterface<dim>> interp_in)
            :
            Function<dim>(1),
            interp(std::move(interp_in))
        {}

        void set_interpolant(std::shared_ptr<const InterpInterface<dim>> interp_in) {
            interp = std::move(interp_in);
        }

        void set_time_index(const unsigned int time_index_in)
        {
            time_index = time_index_in;
        }

        double value(const Point<dim> &point, const unsigned int component = 0) const override
        {
            (void)component;

            return interp->interpolate(point, time_index);
        }

        void value_list(const std::vector<Point<dim>> &points,
            std::vector<double> &values, const unsigned int component = 0) const override
        {
            for (unsigned int i = 0; i < points.size(); ++i)
                values[i] = value(points[i], component);
        }
    private:
        std::shared_ptr<const InterpInterface<dim>> interp;
        unsigned int time_index = 0;
    };
}

#endif //INTERPOLATION_FUNCTION_H
