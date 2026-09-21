//
// Created by giorgk on 6/24/26.
//

#ifndef BOUNDARY_BASE_H
#define BOUNDARY_BASE_H

#include "boundary_helpers.h"
#include "../interpolation/interpolation_function.h"
#include "lateral_polyline_spatial_interpolant.h"

namespace npsat_flow {
    using namespace dealii;

    template <int dim>
    class BoundaryPart{
    public:
        BoundaryKind kind;

        BoundaryRegion<dim> region;

        std::shared_ptr<InterpInterface<dim>> interp;
        std::shared_ptr<LateralPolylineSpatialInterpolant<dim>> lateral_function;

        InterpolationFunction<dim> function;

        BoundaryPart(BoundaryKind kind_in, BoundaryRegion<dim> region_in,
            std::shared_ptr<InterpInterface<dim>> interp_in)
        :
            kind(kind_in),
            region(std::move(region_in)),
            interp(std::move(interp_in)),
            function(interp)
        {}

        void set_time_index(const unsigned int time_index)
        {
            function.set_time_index(time_index);
        }
    };

    template <int dim>
    class BoundaryManagerBase
    {
    public:
        using Part = BoundaryPart<dim>;

        void set_time_index(const unsigned int time_index)
        {
            for (auto &part : parts)
                part.set_time_index(time_index);
        }

        std::vector<Part> &get_parts()
        {
            return parts;
        }

        const std::vector<Part> &get_parts() const
        {
            return parts;
        }
    protected:
      void read_boundary_file(const std::string &filename,
                                MPI_Comm comm = MPI_COMM_WORLD,
                                const std::string &input_path = "")
        {
            const std::string boundary_file = resolve_relative_path(input_path, filename);
            std::string boundary_text;
            read_text_file_mpi(boundary_file, comm, boundary_text);
            std::istringstream in(boundary_text);

            unsigned int n_boundaries = 0;
            in >> n_boundaries;
            if (!in)
                throw std::runtime_error("Invalid boundary file: " + boundary_file);

            for (unsigned int ib = 0; ib < n_boundaries; ++ib)
            {
                std::string kind_string;
                int n_points = 0;
                std::string interp_master_file;

                in >> kind_string >> n_points >> interp_master_file;

                BoundaryKind kind = boundary_kind_from_string(kind_string);

                BoundaryRegion<dim> region;

                if (n_points == 0)
                {
                    region.shape = BoundaryRegion<dim>::Shape::Empty;
                }
                else if (n_points == 2 &&
                     (kind == BoundaryKind::EDGE || kind == BoundaryKind::EDGETOP))
                {
                    region.shape = BoundaryRegion<dim>::Shape::PolylineBuffer;
                }
                else
                {
                    region.shape = BoundaryRegion<dim>::Shape::Polygon;
                }

                region.points.resize(static_cast<unsigned int>(n_points));

                for (int i = 0; i < n_points; ++i)
                {
                    double x, y;
                    in >> x >> y;
                    region.points[static_cast<unsigned int>(i)] =
                        Point<2>(x, y);
                }

                auto interp = std::make_shared<InterpInterface<dim>>();
                interp->read_master_file(interp_master_file, 1.0, comm, input_path);

                parts.emplace_back(kind, std::move(region), std::move(interp));

            }
        }

        std::vector<Part> parts;

    };

}

#endif //BOUNDARY_BASE_H
