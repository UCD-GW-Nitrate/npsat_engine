//
// Created by giorgk on 6/24/26.
//

#ifndef BOUNDARY_HELPERS_H
#define BOUNDARY_HELPERS_H

#include <deal.II/base/point.h>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>

namespace npsat_flow {
    using namespace dealii;

    namespace bg = boost::geometry;

    using boost_point   = bg::model::d2::point_xy<double>;
    using boost_polygon = bg::model::polygon<boost_point>;

    /**
     * @brief Supported Dirichlet boundary geometry types.
     *
     * TOP and BOT are horizontal polygonal regions.
     * EDGE and EDGETOP are vertical lateral boundaries.
     */
    enum class BoundaryKind
    {
        TOP,
        BOT,
        EDGE,
        EDGETOP
    };

    /**
     * @brief Convert a user boundary keyword to a BoundaryKind.
     *
     * Accepted keywords are TOP, BOT, EDGE, and EDGETOP.
     */
    inline BoundaryKind boundary_kind_from_string(const std::string &s)
    {
        if (s == "TOP")     return BoundaryKind::TOP;
        if (s == "BOT")     return BoundaryKind::BOT;
        if (s == "EDGE")    return BoundaryKind::EDGE;
        if (s == "EDGETOP") return BoundaryKind::EDGETOP;

        throw std::runtime_error("Unknown boundary kind '" + s +
                                 "'. Expected one of TOP, BOT, EDGE, or EDGETOP.");
    }

    /**
     * @brief Add an id to a vector only if it is not already present.
     */
    inline void add_unique_id(std::vector<int> &ids, const int id)
    {
        if (std::find(ids.begin(), ids.end(), id) == ids.end())
            ids.push_back(id);
    }

    /**
     * @brief Lightweight geometry region used to identify boundary faces.
     *
     * All geometry is stored as Point<dim>. For boundary matching, only
     * coordinates 0 and 1 are used.
     */
    template <int dim>
    class BoundaryRegion
    {
    public:
        enum class Shape
        {
            Empty,
            Polygon,
            PolylineBuffer
        };

        Shape shape = Shape::Empty;
        std::vector<Point<dim>> points;
        boost_polygon polygon;
        std::vector<boost_polygon> buffered_polygons;

        BoundaryRegion() = default;

        bool empty() const
        {
            return shape == Shape::Empty || points.empty();
        }

        void set_polygon(const std::vector<Point<dim>> &pts)
        {
            if (pts.size() < 3)
                throw std::runtime_error("Polygon boundary requires at least 3 points.");

            polygon.clear();

            for (const auto &p : pts)
                bg::append(polygon.outer(), boost_point(p[0], p[1]));

            bg::append(polygon.outer(), boost_point(pts.front()[0], pts.front()[1]));

            bg::correct(polygon);

            shape = Shape::Polygon;
        }

        void set_rectangle(const std::vector<Point<dim>> &pts)
        {
            if (pts.size() != 2)
                throw std::runtime_error("Rectangle boundary requires exactly 2 points.");

            const double xmin = std::min(pts[0][0], pts[1][0]);
            const double xmax = std::max(pts[0][0], pts[1][0]);
            const double ymin = std::min(pts[0][1], pts[1][1]);
            const double ymax = std::max(pts[0][1], pts[1][1]);

            if (xmax <= xmin || ymax <= ymin)
                throw std::runtime_error("Rectangle boundary has zero area.");

            polygon.clear();
            bg::append(polygon.outer(), boost_point(xmin, ymin));
            bg::append(polygon.outer(), boost_point(xmax, ymin));
            bg::append(polygon.outer(), boost_point(xmax, ymax));
            bg::append(polygon.outer(), boost_point(xmin, ymax));
            bg::append(polygon.outer(), boost_point(xmin, ymin));

            bg::correct(polygon);

            shape = Shape::Polygon;
        }

        void set_lateral_polyline_buffer(const std::vector<Point<dim>> &pts,
                                     const double half_width)
        {
            if (pts.size() < 2)
                throw std::runtime_error("Lateral boundary requires at least 2 points.");

            if (half_width <= 0.0)
                throw std::runtime_error("Lateral buffer half-width must be positive.");

            buffered_polygons.clear();

            for (unsigned int i = 0; i + 1 < pts.size(); ++i)
            {
                const auto &a = pts[i];
                const auto &b = pts[i + 1];

                const double dx = b[0] - a[0];
                const double dy = b[1] - a[1];
                const double L = std::sqrt(dx * dx + dy * dy);

                if (L <= 0.0)
                    continue;

                const double nx = -dy / L;
                const double ny =  dx / L;

                boost_polygon poly;

                bg::append(poly.outer(), boost_point(a[0] + half_width * nx,
                                                     a[1] + half_width * ny));
                bg::append(poly.outer(), boost_point(b[0] + half_width * nx,
                                                     b[1] + half_width * ny));
                bg::append(poly.outer(), boost_point(b[0] - half_width * nx,
                                                     b[1] - half_width * ny));
                bg::append(poly.outer(), boost_point(a[0] - half_width * nx,
                                                     a[1] - half_width * ny));
                bg::append(poly.outer(), boost_point(a[0] + half_width * nx,
                                                     a[1] + half_width * ny));

                bg::correct(poly);
                buffered_polygons.push_back(std::move(poly));
            }

            shape = Shape::PolylineBuffer;
        }

        bool face_polygon_intersects(const std::vector<Point<dim>> &face_points) const
        {
            boost_polygon face_poly = make_polygon(face_points);

            if (shape == Shape::Polygon)
                return bg::intersects(face_poly, polygon) ||
                       bg::within(face_poly, polygon) ||
                       bg::within(polygon, face_poly);

            if (shape == Shape::PolylineBuffer)
            {
                for (const auto &buf : buffered_polygons)
                    if (bg::intersects(face_poly, buf) ||
                        bg::within(face_poly, buf) ||
                        bg::within(buf, face_poly))
                        return true;
            }

            return false;
        }

        double lateral_buffer_intersection_length(const Point<dim> &a,
                                              const Point<dim> &b) const
        {
            if (shape != Shape::PolylineBuffer)
                return 0.0;

            bg::model::linestring<boost_point> line;
            bg::append(line, boost_point(a[0], a[1]));
            bg::append(line, boost_point(b[0], b[1]));

            double length = 0.0;

            for (const auto &buf : buffered_polygons)
            {
                std::vector<bg::model::linestring<boost_point>> out;
                bg::intersection(line, buf, out);

                for (const auto &seg : out)
                    length += bg::length(seg);
            }

            return length;
        }

        double polygon_intersection_length(const Point<dim> &a,
                                           const Point<dim> &b) const
        {
            if (shape != Shape::Polygon)
                return 0.0;

            bg::model::linestring<boost_point> line;
            bg::append(line, boost_point(a[0], a[1]));
            bg::append(line, boost_point(b[0], b[1]));

            std::vector<bg::model::linestring<boost_point>> out;
            bg::intersection(line, polygon, out);

            double length = 0.0;
            for (const auto &seg : out)
                length += bg::length(seg);

            return length;
        }

        bool bbox_intersects(const std::vector<Point<dim>> &face_points) const
        {
            if (points.empty() || face_points.empty())
                return false;

            const auto b1 = bbox(points);
            const auto b2 = bbox(face_points);

            return !(b1.xmax < b2.xmin || b2.xmax < b1.xmin ||
                     b1.ymax < b2.ymin || b2.ymax < b1.ymin);
        }

        bool contains_point(const Point<dim> &p) const
        {
            if (shape != Shape::Polygon)
                return false;

            return bg::covered_by(boost_point(p[0], p[1]), polygon);
        }

    private:
        struct Box
        {
            double xmin, ymin, xmax, ymax;
        };

        static Box bbox(const std::vector<Point<dim>> &pts)
        {
            Box b{
                pts[0][0], pts[0][1],
                pts[0][0], pts[0][1]
            };

            for (const auto &p : pts)
            {
                b.xmin = std::min(b.xmin, p[0]);
                b.ymin = std::min(b.ymin, p[1]);
                b.xmax = std::max(b.xmax, p[0]);
                b.ymax = std::max(b.ymax, p[1]);
            }

            return b;
        }

        static boost_polygon make_polygon(const std::vector<Point<dim>> &pts)
        {
            boost_polygon poly;

            for (const auto &p : pts)
                bg::append(poly.outer(), boost_point(p[0], p[1]));

            bg::append(poly.outer(), boost_point(pts.front()[0], pts.front()[1]));

            bg::correct(poly);

            return poly;
        }

    };
}

#endif //BOUNDARY_HELPERS_H
