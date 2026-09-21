//
// Created by giorgk on 6/23/26.
//

#ifndef INTERP_HELPERS_H
#define INTERP_HELPERS_H

#include <deal.II/base/point.h>

namespace npsat_flow{
    using namespace dealii;

    // ============================================================
    // 2D interpolation region
    //
    // Regions are ALWAYS 2D in NPSAT.
    // Even when dim=3, ownership is checked only in XY.
    // ============================================================
     template<int dim>
    class InterpRegion2D
    {
    public:
        enum class Shape
        {
            Rectangle,
            Polygon
        };

        Shape shape = Shape::Polygon;

        std::vector<Point<2>> vertices;

        void finalize()
        {
            update_bounding_box();
        }

        bool contains(const Point<dim> &p) const
        {
            if (shape == Shape::Rectangle)
                return contains_rectangle(p);

            return contains_polygon(p);
        }
    private:
        bool bbox_initialized = false;
        double xmin = 0.0;
        double xmax = 0.0;
        double ymin = 0.0;
        double ymax = 0.0;

        static constexpr double relative_boundary_tolerance = 1.0e-10;

        void update_bounding_box()
        {
            if (vertices.empty())
            {
                bbox_initialized = false;
                return;
            }

            xmin = xmax = vertices[0][0];
            ymin = ymax = vertices[0][1];

            for (const auto &v : vertices)
            {
                xmin = std::min(xmin, v[0]);
                xmax = std::max(xmax, v[0]);
                ymin = std::min(ymin, v[1]);
                ymax = std::max(ymax, v[1]);
            }

            bbox_initialized = true;
        }

        double tolerance() const
        {
            if (!bbox_initialized)
                return relative_boundary_tolerance;

            const double dx = xmax - xmin;
            const double dy = ymax - ymin;
            const double scale = std::max(1.0, std::sqrt(dx * dx + dy * dy));

            return relative_boundary_tolerance * scale;
        }

        bool bbox_contains(const Point<dim> &p, const double tol) const
        {
            if (!bbox_initialized)
                return true;

            return p[0] >= xmin - tol &&
                   p[0] <= xmax + tol &&
                   p[1] >= ymin - tol &&
                   p[1] <= ymax + tol;
        }

        bool contains_rectangle(const Point<dim> &p) const
        {
            const double tol = tolerance();

            if (bbox_initialized)
                return bbox_contains(p, tol);

            const auto &a = vertices[0];
            const auto &b = vertices[1];

            return p[0] >= std::min(a[0], b[0]) - tol &&
                   p[0] <= std::max(a[0], b[0]) + tol &&
                   p[1] >= std::min(a[1], b[1]) - tol &&
                   p[1] <= std::max(a[1], b[1]) + tol;
        }

        static bool point_near_segment(const Point<dim> &p,
                                       const Point<2> &a,
                                       const Point<2> &b,
                                       const double tol)
        {
            const double dx = b[0] - a[0];
            const double dy = b[1] - a[1];
            const double length2 = dx * dx + dy * dy;

            if (length2 <= std::numeric_limits<double>::epsilon())
            {
                const double px = p[0] - a[0];
                const double py = p[1] - a[1];
                return px * px + py * py <= tol * tol;
            }

            double t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length2;
            t = std::max(0.0, std::min(1.0, t));

            const double qx = a[0] + t * dx;
            const double qy = a[1] + t * dy;
            const double ex = p[0] - qx;
            const double ey = p[1] - qy;

            return ex * ex + ey * ey <= tol * tol;
        }

        bool contains_polygon(const Point<dim> &p) const
        {
            const double tol = tolerance();
            if (!bbox_contains(p, tol))
                return false;

            bool inside = false;
            const unsigned int n = vertices.size();

            for (unsigned int i = 0, j = n - 1; i < n; j = i++)
            {
                if (point_near_segment(p, vertices[j], vertices[i], tol))
                    return true;

                const double xi = vertices[i][0];
                const double yi = vertices[i][1];
                const double xj = vertices[j][0];
                const double yj = vertices[j][1];

                const bool intersect =
                    ((yi > p[1]) != (yj > p[1])) &&
                    (p[0] < (xj - xi) * (p[1] - yi) / (yj - yi) + xi);

                if (intersect)
                    inside = !inside;
            }

            return inside;
        }
    };

    // ============================================================
    // Base interpolation object
    //
    // One object =
    //     one polygon/rectangle
    //     one interpolation method
    // ============================================================
    template <int dim>
    class SpatialInterpolantBase
    {
    public:
        explicit SpatialInterpolantBase(InterpRegion2D<dim> region_in)
            :
            region(std::move(region_in))
        {}

        virtual ~SpatialInterpolantBase() = default;

        virtual bool contains(const Point<dim> &p) const
        {
            return region.contains(p);
        }

        virtual bool requires_lateral_face_overlap() const
        {
            return false;
        }

        virtual bool face_matches_lateral_overlap(const Point<dim> &,
                                                  const Point<dim> &) const
        {
            return true;
        }

        virtual void read_spatial_data(const std::string &filename, MPI_Comm comm = MPI_COMM_WORLD) = 0;
        virtual void read_values(const std::string &filename, double factor, MPI_Comm comm = MPI_COMM_WORLD) = 0;

        virtual double interpolate(const Point<dim> &p, unsigned int time_index) const = 0;
        virtual std::int64_t n_times() const = 0;

    protected:
        InterpRegion2D<dim> region;
    };

}

#endif //INTERP_HELPERS_H
