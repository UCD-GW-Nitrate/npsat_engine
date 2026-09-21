//
// Created by giorgk on 6/23/26.
//

#ifndef INTERP_INTERFACE_H
#define INTERP_INTERFACE_H

#include <deal.II/base/point.h>
#include <memory>

#include "../helper_func.h"
#include "../reader_helper_func.h"
#include "interp_helpers.h"
#include "constant_interpolation.h"
#include "gridded_interpolation.h"
#include "scatter_interpolation.h"
#include "../BC/lateral_polyline_spatial_interpolant.h"


namespace npsat_flow{
    using namespace dealii;

    template <int dim>
    class InterpInterface{
    public:
        InterpInterface() = default;

        double interpolate(const Point<dim> &p, unsigned int time_index) const
        {
            for (const auto &interp : interpolants)
            {
                if (interp->contains(p))
                    return interp->interpolate(p, time_index);
            }

            return default_value;
        }

        /**
         * @brief Read an interpolation master file and construct its regions.
         *
         * The master file lists interpolation regions. For each region, this
         * function reads the region geometry, creates the requested spatial
         * interpolant, reads its spatial definition file, and then reads the
         * corresponding value time-series file.
         *
         * Region ownership is checked in the XY plane through InterpRegion2D,
         * even when dim == 3. The first region containing a query point is used.
         *
         * Master file format:
         *
         * @code
         * n_regions
         *
         * n_vertices interpolation_type spatial_file values_file
         * region_x0 region_y0
         * region_x1 region_y1
         * ...
         * @endcode
         *
         * For each region:
         *
         * - n_vertices == 2: rectangular region. Exactly two XY points are read,
         *   interpreted as opposite rectangle corners.
         * - n_vertices > 2: polygon region. Exactly n_vertices XY vertices are read.
         * - n_vertices < 2: invalid region definition.
         *
         * Supported interpolation_type values are:
         *
         * - CONST
         * - SCATTER
         * - GRIDDED
         * - LATERAL_POLYLINE
         * - LATERAL
         *
         * spatial_file is passed to the interpolant's read_spatial_data().
         * values_file is passed to the interpolant's read_values().
         * Relative file names are resolved against input_path.
         *
         * Example:
         *
         * @code
         * 2
         *
         * 2 CONST recharge_region.dat recharge_values.dat
         * 0 0
         * 1000 1000
         *
         * 4 GRIDDED grid_definition.dat grid_values.dat
         * 0 0
         * 1000 0
         * 1000 1000
         * 0 1000
         * @endcode
         *
         * @param filename Master-file path, or path relative to input_path.
         * @param factor Multiplicative factor applied to each values file.
         * @param comm MPI communicator used by the interpolants while reading.
         * @param input_path Base path used to resolve relative file names.
         */
        void read_master_file(const std::string &filename,
                              double factor,
                              MPI_Comm comm,
                              const std::string &input_path = "",
                              double lateral_polyline_half_width = 20.0,
                              double lateral_polyline_min_overlap = 5.0);
        std::int64_t n_times() const;
        bool requires_lateral_face_overlap() const;
        bool face_matches_lateral_overlap(const Point<dim> &a,
                                          const Point<dim> &b) const;

    private:
        static InterpRegion2D<dim> read_region(std::istream &in, const int n_vertices);
        static double region_area(const InterpRegion2D<dim> &region);
        std::unique_ptr<SpatialInterpolantBase<dim>> create_interpolant(const std::string &interp_type,
                                                                        InterpRegion2D<dim> region,
                                                                        double lateral_polyline_half_width,
                                                                        double lateral_polyline_min_overlap);
        std::vector<std::unique_ptr<SpatialInterpolantBase<dim>>> interpolants;

        double default_value = 0.0;
    };

    template <int dim>
    std::int64_t InterpInterface<dim>::n_times() const
    {
        if (interpolants.empty())
            return 0;

        return interpolants.front()->n_times();
    }

    template <int dim>
    void InterpInterface<dim>::read_master_file(const std::string& filename,
                                                double factor,
                                                MPI_Comm comm,
                                                const std::string &input_path,
                                                const double lateral_polyline_half_width,
                                                const double lateral_polyline_min_overlap){

        const std::string master_file = resolve_relative_path(input_path, filename);

        std::string master_text;
        read_text_file_mpi(master_file, comm, master_text);
        std::istringstream in(master_text);

        unsigned int n_regions = 0;
        in >> n_regions;
        AssertThrow(in, ExcMessage("Invalid interpolation master file: " + master_file));

        for (unsigned int ir = 0; ir < n_regions; ++ir) {
            std::string interp_type;
            std::string spatial_file;
            std::string values_file;

            int n_vertices = 0;
            in >> n_vertices >> interp_type >> spatial_file >> values_file;
            AssertThrow(in,
                        ExcMessage("Invalid interpolation master file entry for region " +
                                   std::to_string(ir) + " in file: " + master_file));
            AssertThrow(n_vertices >= 2,
                        ExcMessage("Invalid interpolation region " + std::to_string(ir) +
                                   " in file '" + master_file +
                                   "': n_vertices must be at least 2. Read n_vertices = " +
                                   std::to_string(n_vertices) + "."));

            InterpRegion2D<dim> region = read_region(in, n_vertices);
            const double area = region_area(region);
            constexpr double area_tolerance = 1.0e-14;
            AssertThrow(area > area_tolerance,
                        ExcMessage("Invalid interpolation region " + std::to_string(ir) +
                                   " in file '" + master_file +
                                   "': region area is too close to zero (" +
                                   std::to_string(area) +
                                   "). Check that the " + std::to_string(n_vertices) +
                                   " vertices define a non-degenerate region."));

            auto interp = create_interpolant(interp_type,
                                             std::move(region),
                                             lateral_polyline_half_width,
                                             lateral_polyline_min_overlap);
            AssertThrow(interp != nullptr,
                        ExcMessage("Unknown interpolation type: " + interp_type));

            interp->read_spatial_data(resolve_relative_path(input_path, spatial_file), comm);
            interp->read_values(resolve_relative_path(input_path, values_file), factor, comm);

            interpolants.push_back(std::move(interp));
        }
    }

    template <int dim>
    InterpRegion2D<dim> InterpInterface<dim>::read_region(std::istream& in, const int n_vertices)
    {
        InterpRegion2D<dim> region;
        if (n_vertices == 2)
        {
            region.shape = InterpRegion2D<dim>::Shape::Rectangle;
            region.vertices.resize(2);

            for (unsigned int i = 0; i < 2; ++i)
            {
                double x = 0.0;
                double y = 0.0;

                in >> x >> y;
                AssertThrow(in,
                            ExcMessage("Invalid rectangle interpolation region: failed to read vertex " +
                                       std::to_string(i) + "."));

                region.vertices[i] = dealii::Point<2>(x, y);
            }
        }
        else
        {
            region.shape = InterpRegion2D<dim>::Shape::Polygon;
            region.vertices.resize(static_cast<unsigned int>(n_vertices));

            for (int i = 0; i < n_vertices; ++i)
            {
                double x = 0.0;
                double y = 0.0;

                in >> x >> y;
                AssertThrow(in,
                            ExcMessage("Invalid polygon interpolation region with " +
                                       std::to_string(n_vertices) +
                                       " vertices: failed to read vertex " +
                                       std::to_string(i) + "."));

                region.vertices[static_cast<unsigned int>(i)] =
                    dealii::Point<2>(x, y);
            }
        }

        region.finalize();
        return region;
    }

    template <int dim>
    double InterpInterface<dim>::region_area(const InterpRegion2D<dim>& region)
    {
        if (region.shape == InterpRegion2D<dim>::Shape::Rectangle)
        {
            const auto &a = region.vertices[0];
            const auto &b = region.vertices[1];

            return std::abs((b[0] - a[0]) * (b[1] - a[1]));
        }

        double signed_area2 = 0.0;
        const unsigned int n = region.vertices.size();
        for (unsigned int i = 0, j = n - 1; i < n; j = i++)
        {
            const auto &a = region.vertices[j];
            const auto &b = region.vertices[i];
            signed_area2 += a[0] * b[1] - b[0] * a[1];
        }

        return 0.5 * std::abs(signed_area2);
    }

    template <int dim>
    std::unique_ptr<SpatialInterpolantBase<dim>>
    InterpInterface<dim>::create_interpolant(const std::string& interp_type,
                                             InterpRegion2D<dim> region,
                                             const double lateral_polyline_half_width,
                                             const double lateral_polyline_min_overlap)
    {
        if (interp_type == "CONST")
        {
            return std::make_unique<ConstantSpatialInterpolant<dim>>(std::move(region));
        }

        if (interp_type == "SCATTER")
        {
            return std::make_unique<ScatteredSpatialInterpolant<dim>>(std::move(region));
        }

        if (interp_type == "GRIDDED")
        {
            return std::make_unique<GriddedSpatialInterpolant<dim>>(std::move(region));
        }

        if (interp_type == "LATERAL_POLYLINE" ||
            interp_type == "LATERAL")
        {
             return std::make_unique<LateralPolylineSpatialInterpolant<dim>>(
                 std::move(region),
                 lateral_polyline_half_width,
                 lateral_polyline_min_overlap);
        }

        return nullptr;
    }

    template <int dim>
    bool InterpInterface<dim>::requires_lateral_face_overlap() const
    {
        for (const auto &interp : interpolants)
            if (interp->requires_lateral_face_overlap())
                return true;

        return false;
    }

    template <int dim>
    bool InterpInterface<dim>::face_matches_lateral_overlap(const Point<dim> &a,
                                                            const Point<dim> &b) const
    {
        bool has_lateral_overlap_check = false;

        for (const auto &interp : interpolants)
        {
            if (!interp->requires_lateral_face_overlap())
                continue;

            has_lateral_overlap_check = true;

            if (interp->face_matches_lateral_overlap(a, b))
                return true;
        }

        return !has_lateral_overlap_check;
    }

}

#endif //INTERP_INTERFACE_H
