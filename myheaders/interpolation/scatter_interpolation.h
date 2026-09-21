//
// Created by giorgk on 6/23/26.
//

#ifndef SCATTER_INTERPOLATION_H
#define SCATTER_INTERPOLATION_H

#include <deal.II/base/mpi.h>
#include <map>
#include <array>
#include <memory>
#include "../nanoflann.hpp"

#include "interp_helpers.h"
#include "../reader_helper_func.h"
#include "../TimeSeries.h"

namespace npsat_flow{
    using namespace dealii;

    enum class ScatterMode
    {
        LINEAR,
        NEAREST
    };

    inline ScatterMode read_scatter_mode(const std::string &s)
    {
        if (s == "LINEAR")  return ScatterMode::LINEAR;
        if (s == "NEAREST") return ScatterMode::NEAREST;

        throw std::runtime_error("Unknown scatter interpolation mode: " + s);
    }

    struct ScatterKDPoint
    {
        double x = 0.0;
        double y = 0.0;
        int id = -1;
    };

    struct ScatterKDCloud
    {
        std::vector<ScatterKDPoint> pts;

        inline size_t kdtree_get_point_count() const
        {
            return pts.size();
        }

        inline double kdtree_get_pt(const size_t idx,
                                    const size_t dim) const
        {
            return dim == 0 ? pts[idx].x : pts[idx].y;
        }

        template <class BBOX>
        bool kdtree_get_bbox(BBOX &) const
        {
            return false;
        }
    };

    using ScatterKDTree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Adaptor<double, ScatterKDCloud>, ScatterKDCloud, 2>;

    template <int dim>
    class ScatteredSpatialInterpolant final : public SpatialInterpolantBase<dim>{
    public:
        explicit ScatteredSpatialInterpolant(InterpRegion2D<dim> region_in)
            : SpatialInterpolantBase<dim>(std::move(region_in))
        {}

        // In-memory input for the steady solver's reconstructed free surface.
        // The interpolation algorithm is the same as for file-based SCATTER data.
        void set_surface_data(const std::map<int, Point<dim>> &points,
                              const std::vector<std::vector<int>> &elements)
        {
            AssertThrow(dim == 3, ExcMessage("Triangulated free surfaces require dim == 3"));
            nodes.clear();
            triangles.clear();
            std::map<int, int> row_ids;
            std::vector<double> elevations;
            for (const auto &entry : points) {
                const int row = static_cast<int>(nodes.size());
                row_ids.emplace(entry.first, row);
                NodeData node;
                node.x = entry.second[0];
                node.y = entry.second[1];
                node.ids.push_back(row);
                nodes.push_back(std::move(node));
                elevations.push_back(entry.second[dim-1]);
            }
            for (const auto &element : elements) {
                AssertThrow(element.size() == 3, ExcMessage("Expected a surface triangle"));
                triangles.push_back({{row_ids.at(element[0]), row_ids.at(element[1]), row_ids.at(element[2])}});
            }
            n_points = nodes.size();
            n_triangles = triangles.size();
            n_layers = 1;
            mode_xy = ScatterMode::LINEAR;
            mode_z = ScatterMode::NEAREST;
            values.set_data_owned(std::move(elevations), n_points, 1);
            check_data_consistency();
            build_triangle_index();
        }

        void read_spatial_data(const std::string &filename, MPI_Comm comm) override
        {
            std::string file_text;
            read_text_file_mpi(filename, comm, file_text);
            std::istringstream in(file_text);

            in >> n_points >> n_triangles;
            in >> mode_xy_string >> mode_z_string;
            if (!in)
                throw std::runtime_error("Invalid scattered interpolation file: " + filename);

            mode_xy = read_scatter_mode(mode_xy_string);
            mode_z  = read_scatter_mode(mode_z_string);

            read_points(in);
            read_triangles(in);
            build_triangle_index();
        }

        void read_values(const std::string &filename, const double factor, MPI_Comm comm) override
        {
            values.read_data(filename, comm);
            values.multiply_by_factor(factor);
            check_data_consistency();
        }

        double interpolate(const Point<dim> &p,
                       unsigned int time_index) const override
        {
            TriangleQuery q = find_triangle(p[0], p[1]);

            if (mode_xy == ScatterMode::NEAREST || !q.inside)
                return interpolate_nearest_node(q.nearest_node,
                                                p,
                                                time_index);

            return interpolate_linear_triangle(q.triangle_id,
                                               q.w,
                                               p,
                                               time_index);
        }

        std::int64_t n_times() const override
        {
            return values.n_times();
        }

    private:
        struct NodeData
        {
            double x = 0.0;
            double y = 0.0;

            std::vector<int> ids;
            std::vector<double> interfaces;
        };

        struct TriangleQuery
        {
            int triangle_id = -1;
            int nearest_node = -1;
            bool inside = false;

            std::array<double, 3> w{{1.0, 0.0, 0.0}};
        };

        unsigned int n_points = 0;
        unsigned int n_triangles = 0;
        unsigned int n_layers = 1;

        std::string mode_xy_string;
        std::string mode_z_string;
        std::string values_file;

        ScatterMode mode_xy = ScatterMode::LINEAR;
        ScatterMode mode_z  = ScatterMode::NEAREST;

        std::vector<NodeData> nodes;
        std::vector<std::array<int, 3>> triangles;

        ScatterKDCloud triangle_cloud;
        std::shared_ptr<ScatterKDTree> triangle_index;

        TimeSeriesData<double> values;

        unsigned int n_search_triangles = 20;
        double bary_tol = 1.0e-10;


        void check_data_consistency() const
        {
            const std::int64_t n_value_rows = values.n_points();
            if (n_value_rows <= 0)
                throw std::runtime_error(
                    "Scattered interpolation data consistency error: values file has no rows.");

            for (unsigned int inode = 0; inode < nodes.size(); ++inode)
            {
                const NodeData &node = nodes[inode];

                if (node.ids.size() != n_layers)
                {
                    throw std::runtime_error(
                        "Scattered interpolation data consistency error: node " +
                        std::to_string(inode) + " has " +
                        std::to_string(node.ids.size()) + " value row IDs, but " +
                        std::to_string(n_layers) + " layers were read.");
                }

                for (unsigned int lay = 0; lay < node.ids.size(); ++lay)
                {
                    const int row_id = node.ids[lay];

                    if (row_id < 0)
                        continue;

                    if (static_cast<std::int64_t>(row_id) >= n_value_rows)
                    {
                        throw std::runtime_error(
                            "Scattered interpolation data consistency error: node " +
                            std::to_string(inode) + ", layer " +
                            std::to_string(lay) + " references value row " +
                            std::to_string(row_id) + ", but the values file has only " +
                            std::to_string(n_value_rows) + " rows.");
                    }
                }
            }
        }

        void read_points(std::istream &in)
        {
            nodes.resize(n_points);

            std::string line;
            std::getline(in, line);

            for (unsigned int i = 0; i < n_points; ++i)
            {
                std::getline(in, line);
                std::istringstream iss(line);

                std::vector<double> vals;
                double v;

                while (iss >> v)
                    vals.push_back(v);

                if (vals.size() < 3)
                    throw std::runtime_error("Invalid scattered interpolation point row.");

                nodes[i].x = vals[0];
                nodes[i].y = vals[1];

                const unsigned int remaining =
                    static_cast<unsigned int>(vals.size()) - 2;

                if (remaining == 1)
                {
                    n_layers = 1;
                    nodes[i].ids.resize(1);
                    nodes[i].ids[0] = static_cast<int>(vals[2]);
                }
                else
                {
                    if (remaining % 2 == 0)
                        throw std::runtime_error(
                            "Invalid scattered row. Expected: X Y ID1 elev12 ID2 ... IDn");

                    n_layers = (remaining + 1) / 2;

                    nodes[i].ids.resize(n_layers);
                    nodes[i].interfaces.resize(n_layers - 1);

                    unsigned int k = 2;

                    for (unsigned int lay = 0; lay < n_layers; ++lay)
                    {
                        nodes[i].ids[lay] = static_cast<int>(vals[k++]);

                        if (lay < n_layers - 1)
                            nodes[i].interfaces[lay] = vals[k++];
                    }
                }
            }
        }

        void read_triangles(std::istream &in)
        {
            triangles.resize(n_triangles);

            for (unsigned int i = 0; i < n_triangles; ++i)
            {
                int a, b, c;
                in >> a >> b >> c;

                triangles[i] = {{a, b, c}};
            }
        }

        void build_triangle_index()
        {
            triangle_cloud.pts.clear();
            triangle_cloud.pts.reserve(triangles.size());

            for (unsigned int itri = 0; itri < triangles.size(); ++itri)
            {
                const auto &tri = triangles[itri];

                const NodeData &a = nodes[tri[0]];
                const NodeData &b = nodes[tri[1]];
                const NodeData &c = nodes[tri[2]];

                ScatterKDPoint p;
                p.x = (a.x + b.x + c.x) / 3.0;
                p.y = (a.y + b.y + c.y) / 3.0;
                p.id = static_cast<int>(itri);

                triangle_cloud.pts.push_back(p);
            }

            triangle_index = std::make_shared<ScatterKDTree>(
                2,
                triangle_cloud,
                nanoflann::KDTreeSingleIndexAdaptorParams(10));

            triangle_index->buildIndex();
        }

        static double signed_area2(const NodeData &a, const NodeData &b,
                               const double x, const double y)
        {
            return (b.x - a.x) * (y - a.y)
                 - (b.y - a.y) * (x - a.x);
        }

        bool barycentric(const int tri_id, const double x, const double y,
                     std::array<double, 3> &w) const
        {
            const auto &tri = triangles[tri_id];

            const NodeData &a = nodes[tri[0]];
            const NodeData &b = nodes[tri[1]];
            const NodeData &c = nodes[tri[2]];

            const double det =
                (b.y - c.y) * (a.x - c.x)
              + (c.x - b.x) * (a.y - c.y);

            if (std::abs(det) < 1.0e-30)
                return false;

            w[0] =
                ((b.y - c.y) * (x - c.x)
               + (c.x - b.x) * (y - c.y)) / det;

            w[1] =
                ((c.y - a.y) * (x - c.x)
               + (a.x - c.x) * (y - c.y)) / det;

            w[2] = 1.0 - w[0] - w[1];

            return w[0] >= -bary_tol &&
                   w[1] >= -bary_tol &&
                   w[2] >= -bary_tol;
        }

        TriangleQuery find_triangle(const double x, const double y) const
        {
            TriangleQuery q;

            const double query_pt[2] = {x, y};

            const size_t nquery =
                std::min<size_t>(n_search_triangles, triangles.size());

            std::vector<typename ScatterKDTree::IndexType> ids(nquery);
            std::vector<double> d2(nquery);

            const size_t nfound = triangle_index->knnSearch(query_pt, nquery, ids.data(), d2.data());

            double best_node_d2 = std::numeric_limits<double>::max();

            for (size_t i = 0; i < nfound; ++i)
            {
                const int tri_id = static_cast<int>(ids[i]);

                std::array<double, 3> w;

                if (barycentric(tri_id, x, y, w))
                {
                    q.triangle_id = tri_id;
                    q.inside = true;
                    q.w = w;
                    return q;
                }

                const auto &tri = triangles[tri_id];

                for (unsigned int j = 0; j < 3; ++j)
                {
                    const int nid = tri[j];

                    const double dx = x - nodes[nid].x;
                    const double dy = y - nodes[nid].y;
                    const double dst2 = dx * dx + dy * dy;

                    if (dst2 < best_node_d2)
                    {
                        best_node_d2 = dst2;
                        q.nearest_node = nid;
                    }
                }
            }

            if (q.nearest_node < 0 && !nodes.empty())
                q.nearest_node = 0;

            return q;
        }

        double node_value(const int node_id, const unsigned int layer,
                      const unsigned int time_index) const
        {
            const int row_id = nodes[node_id].ids[layer];

            if (row_id < 0)
                return 0.0;

            return values.get_value_at_step(row_id, static_cast<int>(time_index));
        }

        double node_interface(const int node_id, const unsigned int interface_id) const
        {
            return nodes[node_id].interfaces[interface_id];
        }

        double interpolate_nearest_node(const int node_id,
                                    const Point<dim> &p,
                                    const unsigned int time_index) const
        {
            if (n_layers == 1)
                return node_value(node_id, 0, time_index);

            const double z = p[2];

            for (unsigned int lay = 0; lay < n_layers - 1; ++lay)
            {
                const double zint = node_interface(node_id, lay);

                if (z >= zint)
                    return node_value(node_id, lay, time_index);
            }

            return node_value(node_id, n_layers - 1, time_index);
        }

        double interpolate_linear_triangle(const int tri_id,
                                       const std::array<double, 3> &w,
                                       const Point<dim> &p,
                                       const unsigned int time_index) const
        {
            const auto &tri = triangles[tri_id];

            if (n_layers == 1)
            {
                return
                    w[0] * node_value(tri[0], 0, time_index) +
                    w[1] * node_value(tri[1], 0, time_index) +
                    w[2] * node_value(tri[2], 0, time_index);
            }

            const double z = p[2];

            std::vector<double> layer_values(n_layers);
            std::vector<double> interface_values(n_layers - 1);

            for (unsigned int lay = 0; lay < n_layers; ++lay)
            {
                layer_values[lay] =
                    w[0] * node_value(tri[0], lay, time_index) +
                    w[1] * node_value(tri[1], lay, time_index) +
                    w[2] * node_value(tri[2], lay, time_index);
            }

            for (unsigned int k = 0; k < n_layers - 1; ++k)
            {
                interface_values[k] =
                    w[0] * node_interface(tri[0], k) +
                    w[1] * node_interface(tri[1], k) +
                    w[2] * node_interface(tri[2], k);
            }

            if (mode_z == ScatterMode::NEAREST)
                return vertical_nearest(z, layer_values, interface_values);

            return vertical_linear(z, layer_values, interface_values);
        }

        double vertical_nearest(const double z,
                            const std::vector<double> &v,
                            const std::vector<double> &zi) const
        {
            for (unsigned int lay = 0; lay < n_layers - 1; ++lay)
                if (z >= zi[lay])
                    return v[lay];

            return v[n_layers - 1];
        }

        double vertical_linear(const double z,
                           const std::vector<double> &v,
                           const std::vector<double> &zi) const
        {
            if (n_layers == 1)
                return v[0];

            if (z >= zi[0])
                return v[0];

            for (unsigned int lay = 0; lay < n_layers - 2; ++lay)
            {
                const double ztop = zi[lay];
                const double zbot = zi[lay + 1];

                if (z <= ztop && z >= zbot)
                {
                    const double a = (z - zbot) / (ztop - zbot);
                    return a * v[lay] + (1.0 - a) * v[lay + 1];
                }
            }

            if (z <= zi[n_layers - 2])
                return v[n_layers - 1];

            return v[n_layers - 1];
        }


    };

}

#endif //SCATTER_INTERPOLATION_H
