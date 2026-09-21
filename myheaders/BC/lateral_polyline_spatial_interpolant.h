//
// Created by giorgk on 6/24/26.
//

#ifndef LATERAL_POLYLINE_SPATIAL_INTERPOLANT_H
#define LATERAL_POLYLINE_SPATIAL_INTERPOLANT_H

#include <deal.II/base/point.h>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>

#include "../interpolation/interp_helpers.h"
#include "../reader_helper_func.h"
#include "../TimeSeries.h"
#include "../helper_func.h"

namespace npsat_flow {
    using namespace dealii;

    namespace lpoly_bg = boost::geometry;

    using lateral_boost_point = lpoly_bg::model::d2::point_xy<double>;
    using lateral_boost_polygon = lpoly_bg::model::polygon<lateral_boost_point>;

    template <int dim>
    class LateralPolylineSpatialInterpolant final : public SpatialInterpolantBase<dim>
    {
    public:
        explicit LateralPolylineSpatialInterpolant(InterpRegion2D<dim> region_in,
                                                   const double half_width_in = 20.0,
                                                   const double min_overlap_length_in = 5.0)
            :
            SpatialInterpolantBase<dim>(std::move(region_in)),
            half_width(half_width_in),
            min_overlap_length(min_overlap_length_in)
        {}

        /**
         * @brief Read the spatial description of the lateral polyline.
         *
         * Expected file format:
         *
         * @code
         * n_polyline_points nlay
         * HOR_type VER_type
         *
         * x0 y0
         * x1 y1
         * ...
         *
         * IDs_layer_0              (1 x n_polyline_points integers)
         * Interface_0_1            (1 x n_polyline_points doubles)
         *
         * IDs_layer_1              (1 x n_polyline_points integers)
         * Interface_1_2            (1 x n_polyline_points doubles)
         *
         * ...
         *
         * IDs_layer_nlay-1         (1 x n_polyline_points integers)
         * @endcode
         *
         * If nlay == 0, no elevation interfaces are read. The boundary value
         * is independent of z and a single ID row is expected after the
         * polyline points.
         *
         * HOR_type and VER_type must be either LINEAR or NEAREST.
         * Interfaces are ordered from top to bottom:
         * interface 0 separates layer 0 and layer 1.
         */
        void read_spatial_data(const std::string &filename, MPI_Comm comm) override
        {
            std::string file_text;
            read_text_file_mpi(filename, comm, file_text);
            std::istringstream in(file_text);

            unsigned int n_points = 0;
            unsigned int nlay = 0;

            in >> n_points >> nlay;
            if (!in)
                throw std::runtime_error(
                    "Invalid lateral polyline spatial file: " + filename);

            if (n_points < 2)
                throw std::runtime_error(
                    "LateralPolylineSpatialInterpolant requires at least two polyline points.");

            const unsigned int n_input_layers = nlay;
            const unsigned int n_stored_layers = nlay == 0 ? 1 : nlay;

            std::string hor_type;
            std::string ver_type;
            in >> hor_type >> ver_type;

            horizontal_interpolation = parse_interpolation_type(hor_type);
            vertical_interpolation = parse_interpolation_type(ver_type);

            polyline_points.resize(n_points);

            for (unsigned int i = 0; i < n_points; ++i)
            {
                double x, y;
                in >> x >> y;

                Point<dim> p;
                p[0] = x;
                p[1] = y;

                polyline_points[i] = p;
            }

            ids.assign(n_stored_layers,std::vector<int>(n_points, -1));

            interfaces.assign(n_input_layers > 0 ? n_input_layers - 1 : 0,
                              std::vector<double>(n_points, 0.0));

            for (unsigned int lay = 0; lay < n_stored_layers; ++lay)
            {
                for (unsigned int i = 0; i < n_points; ++i)
                    in >> ids[lay][i];

                if (n_input_layers > 0 && lay < n_input_layers - 1)
                {
                    for (unsigned int i = 0; i < n_points; ++i)
                        in >> interfaces[lay][i];
                }
            }

            build_polyline_buffers();
            build_station();
        }

        /**
         * @brief Read the time-dependent boundary values.
         *
         * The values file is read using TimeSeriesData<double>.
         *
         * Spatial IDs in the lateral-polyline file are row indices in this
         * TimeSeriesData object.
         */
        void read_values(const std::string &filename,
                         const double factor,
                         MPI_Comm comm = MPI_COMM_WORLD) override
        {
            values.read_data(filename, comm);
            values.multiply_by_factor(factor);
        }

        /**
        * @brief Interpolate the lateral value at point p and time index.
        *
        * The x-y coordinates are projected onto the nearest polyline segment,
        * layer interfaces are sampled according to HOR_type, and layer IDs
        * are used as rows in the TimeSeriesData object. LINEAR interpolation
        * blends values; NEAREST snaps to the nearest sampled row.
        */
        double interpolate(const Point<dim> &p,
                           const unsigned int time_index) const override
        {
            if (polyline_points.size() < 2 || ids.empty())
                return 0.0;

            const Projection projection = project_to_polyline(p);

            if (horizontal_interpolation == InterpolationType::Nearest)
            {
                const unsigned int ip =
                    projection.t <= 0.5 ? projection.i0 : projection.i1;

                const VerticalSample sample =
                    vertical_sample_at_column(p[2], ip);

                return value_at_sample(time_index, ip, sample);
            }

            const VerticalSample left =
                vertical_sample_at_projection(p[2], projection);

            return value_at_projected_sample(time_index, projection, left);
        }

        std::int64_t n_times() const override
        {
            return values.n_times();
        }

        bool contains(const Point<dim> &p) const override
        {
            if (!this->region.contains(p))
                return false;

            for (const auto &buffer : polyline_buffers)
            {
                const lateral_boost_point point(p[0], p[1]);
                if (lpoly_bg::covered_by(point, buffer))
                    return true;
            }

            return false;
        }

        bool requires_lateral_face_overlap() const override
        {
            return true;
        }

        bool face_matches_lateral_overlap(const Point<dim> &a,
                                          const Point<dim> &b) const override
        {
            return buffered_intersection_length(a, b) >= min_overlap_length;
        }

    private:
        struct Projection
        {
            unsigned int i0 = 0;
            unsigned int i1 = 0;
            double t = 0.0;
            double station = 0.0;
            double distance = std::numeric_limits<double>::max();
        };

        enum class InterpolationType
        {
            Linear,
            Nearest
        };

        struct VerticalSample
        {
            unsigned int lay0 = 0;
            unsigned int lay1 = 0;
            double weight = 0.0;
        };

        //static const int invalid_id = -1;

        std::vector<Point<dim>> polyline_points;
        std::vector<double> station;
        std::vector<lateral_boost_polygon> polyline_buffers;
        double half_width = 20.0;
        double min_overlap_length = 5.0;

        /*
         * ids[lay][polyline_point]
         *
         * Layer 0 is the top layer.
         */
        std::vector<std::vector<int>> ids;

        /*
         * interfaces[k][polyline_point]
         *
         * interfaces[0] = elevation between layer 0 and 1
         * interfaces[1] = elevation between layer 1 and 2
         *
         * Size = nlay - 1.
         */
        std::vector<std::vector<double>> interfaces;

        TimeSeriesData<double> values;

        InterpolationType horizontal_interpolation = InterpolationType::Linear;
        InterpolationType vertical_interpolation = InterpolationType::Nearest;

        static InterpolationType parse_interpolation_type(const std::string &type)
        {
            if (type == "LINEAR")
                return InterpolationType::Linear;

            if (type == "NEAREST")
                return InterpolationType::Nearest;

            throw std::runtime_error(
                "LateralPolylineSpatialInterpolant interpolation type must be "
                "LINEAR or NEAREST, got: " + type);
        }

        void build_station()
        {
            station.assign(polyline_points.size(), 0.0);

            for (unsigned int i = 1; i < polyline_points.size(); ++i)
            {
                const double dx =
                    polyline_points[i][0] - polyline_points[i - 1][0];

                const double dy =
                    polyline_points[i][1] - polyline_points[i - 1][1];

                station[i] =
                    station[i - 1] + std::sqrt(dx * dx + dy * dy);
            }
        }

        void build_polyline_buffers()
        {
            if (half_width <= 0.0)
                throw std::runtime_error(
                    "LateralPolylineSpatialInterpolant half width must be positive.");

            if (min_overlap_length < 0.0)
                throw std::runtime_error(
                    "LateralPolylineSpatialInterpolant minimum overlap length must be non-negative.");

            polyline_buffers.clear();

            for (unsigned int i = 0; i + 1 < polyline_points.size(); ++i)
            {
                const auto &a = polyline_points[i];
                const auto &b = polyline_points[i + 1];

                const double dx = b[0] - a[0];
                const double dy = b[1] - a[1];
                const double length = std::sqrt(dx * dx + dy * dy);

                if (length <= 0.0)
                    continue;

                const double nx = -dy / length;
                const double ny =  dx / length;

                lateral_boost_polygon poly;
                lpoly_bg::append(poly.outer(),
                                  lateral_boost_point(a[0] + half_width * nx,
                                                      a[1] + half_width * ny));
                lpoly_bg::append(poly.outer(),
                                  lateral_boost_point(b[0] + half_width * nx,
                                                      b[1] + half_width * ny));
                lpoly_bg::append(poly.outer(),
                                  lateral_boost_point(b[0] - half_width * nx,
                                                      b[1] - half_width * ny));
                lpoly_bg::append(poly.outer(),
                                  lateral_boost_point(a[0] - half_width * nx,
                                                      a[1] - half_width * ny));
                lpoly_bg::append(poly.outer(),
                                  lateral_boost_point(a[0] + half_width * nx,
                                                      a[1] + half_width * ny));

                lpoly_bg::correct(poly);
                polyline_buffers.push_back(std::move(poly));
            }

            if (polyline_buffers.empty())
                throw std::runtime_error(
                    "LateralPolylineSpatialInterpolant could not build any non-degenerate polyline buffers.");
        }

        double buffered_intersection_length(const Point<dim> &a,
                                            const Point<dim> &b) const
        {
            lpoly_bg::model::linestring<lateral_boost_point> line;
            lpoly_bg::append(line, lateral_boost_point(a[0], a[1]));
            lpoly_bg::append(line, lateral_boost_point(b[0], b[1]));

            double length = 0.0;

            for (const auto &buffer : polyline_buffers)
            {
                std::vector<lpoly_bg::model::linestring<lateral_boost_point>> intersections;
                lpoly_bg::intersection(line, buffer, intersections);

                for (const auto &segment : intersections)
                    length += lpoly_bg::length(segment);
            }

            return length;
        }

        Projection project_to_polyline(const Point<dim> &p) const
        {
            Projection best;

            for (unsigned int i = 0; i + 1 < polyline_points.size(); ++i)
            {
                const Point<dim> &a = polyline_points[i];
                const Point<dim> &b = polyline_points[i + 1];

                const double dx = b[0] - a[0];
                const double dy = b[1] - a[1];

                const double length2 = dx * dx + dy * dy;

                if (length2 <= 0.0)
                    continue;

                double t =
                    ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length2;

                t = std::max(0.0, std::min(1.0, t));

                const double qx = a[0] + t * dx;
                const double qy = a[1] + t * dy;

                const double ex = p[0] - qx;
                const double ey = p[1] - qy;

                const double distance = std::sqrt(ex * ex + ey * ey);

                if (distance < best.distance)
                {
                    best.i0 = i;
                    best.i1 = i + 1;
                    best.t = t;
                    best.distance = distance;

                    best.station =
                        station[i] +
                        t * (station[i + 1] - station[i]);
                }
            }

            return best;
        }

        VerticalSample vertical_sample_at_column(const double z,
                                                 const unsigned int ip) const
        {
            if (ids.size() == 1)
                return VerticalSample{0, 0, 0.0};

            const unsigned int nlay = static_cast<unsigned int>(ids.size());

            if (z >= interfaces[0][ip])
                return VerticalSample{0, 0, 0.0};

            for (unsigned int k = 1; k < nlay - 1; ++k)
            {
                const double z_upper = interfaces[k - 1][ip];
                const double z_lower = interfaces[k][ip];

                if (z >= z_lower)
                {
                    if (vertical_interpolation == InterpolationType::Nearest)
                        return VerticalSample{k, k, 0.0};

                    const double dz = z_upper - z_lower;
                    const double weight =
                        dz > 0.0 ? (z_upper - z) / dz : 0.0;

                    return VerticalSample{k - 1, k,
                                          clamp_(weight, 0.0, 1.0)};
                }
            }

            return VerticalSample{nlay - 1, nlay - 1, 0.0};
        }

        VerticalSample vertical_sample_at_projection(const double z,
                                                     const Projection &projection) const
        {
            if (ids.size() == 1)
                return VerticalSample{0, 0, 0.0};

            const unsigned int nlay = static_cast<unsigned int>(ids.size());

            if (z >= interface_at_projection(0, projection))
                return VerticalSample{0, 0, 0.0};

            for (unsigned int k = 1; k < nlay - 1; ++k)
            {
                const double z_upper =
                    interface_at_projection(k - 1, projection);

                const double z_lower =
                    interface_at_projection(k, projection);

                if (z >= z_lower)
                {
                    if (vertical_interpolation == InterpolationType::Nearest)
                        return VerticalSample{k, k, 0.0};

                    const double dz = z_upper - z_lower;
                    const double weight =
                        dz > 0.0 ? (z_upper - z) / dz : 0.0;

                    return VerticalSample{k - 1, k,
                                          clamp_(weight, 0.0, 1.0)};
                }
            }

            return VerticalSample{nlay - 1, nlay - 1, 0.0};
        }

        double interface_at_projection(const unsigned int interface_index,
                                       const Projection &projection) const
        {
            return (1.0 - projection.t) *
                   interfaces[interface_index][projection.i0] +
                   projection.t *
                   interfaces[interface_index][projection.i1];
        }

        double value_at_sample(const unsigned int time_index,
                               const unsigned int ip,
                               const VerticalSample sample) const
        {
            const double v0 = value_at_id(time_index, ids[sample.lay0][ip]);

            if (sample.lay0 == sample.lay1)
                return v0;

            const double v1 = value_at_id(time_index, ids[sample.lay1][ip]);

            return (1.0 - sample.weight) * v0 + sample.weight * v1;
        }

        double value_at_projected_sample(const unsigned int time_index,
                                         const Projection &projection,
                                         const VerticalSample sample) const
        {
            const double v0 =
                value_at_projected_layer(time_index, projection, sample.lay0);

            if (sample.lay0 == sample.lay1)
                return v0;

            const double v1 =
                value_at_projected_layer(time_index, projection, sample.lay1);

            return (1.0 - sample.weight) * v0 + sample.weight * v1;
        }

        double value_at_projected_layer(const unsigned int time_index,
                                        const Projection &projection,
                                        const unsigned int lay) const
        {
            const double v0 =
                value_at_id(time_index, ids[lay][projection.i0]);

            const double v1 =
                value_at_id(time_index, ids[lay][projection.i1]);

            return (1.0 - projection.t) * v0 + projection.t * v1;
        }

        double value_at_id(const unsigned int time_index,
                           const int id) const
        {
            if (id < 0)
                return 0.0;

            return values.get_value_at_step(id,
                                            static_cast<int>(time_index));
        }
    };

}

#endif //LATERAL_POLYLINE_SPATIAL_INTERPOLANT_H
