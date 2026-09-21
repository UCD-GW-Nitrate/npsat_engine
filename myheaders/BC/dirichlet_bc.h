//
// Created by giorgk on 6/24/26.
//

#ifndef DIRICHLET_BC_H
#define DIRICHLET_BC_H

#include <deal.II/distributed/tria.h>

#include "boundary_base.h"
#include "../interpolation/interpolation_function.h"

namespace npsat_flow {
    using namespace dealii;

    /**
     * @brief One user-defined Dirichlet boundary condition.
     *
     * A DirichletPart stores:
     * - the boundary type,
     * - the user-defined geometry,
     * - the custom deal.II boundary id,
     * - the interpolation interface,
     * - the deal.II Function wrapper used by VectorTools.
     *
     * The object supports empty construction so it can be owned by the main
     * model class before any input file is read.
     */
    template <int dim>
    class DirichletPart {
    public:
        BoundaryKind kind;
        BoundaryRegion<dim> region;

        types::boundary_id boundary_id = numbers::invalid_boundary_id;

        std::shared_ptr<InterpInterface<dim>> interp;
        std::unique_ptr<InterpolationFunction<dim>> function;

        DirichletPart() = default;

        void set(const BoundaryKind kind_in,
                 BoundaryRegion<dim> region_in,
                 const types::boundary_id boundary_id_in,
                 std::shared_ptr<InterpInterface<dim>> interp_in)
        {
            kind = kind_in;
            region = std::move(region_in);
            boundary_id = boundary_id_in;
            interp = std::move(interp_in);

            function = std::make_unique<InterpolationFunction<dim>>(interp);
        }

        bool is_initialized() const
        {
            return boundary_id != numbers::invalid_boundary_id &&
                   interp != nullptr &&
                   function != nullptr;
        }

        void set_time_index(const unsigned int time_index)
        {
            if (function)
                function->set_time_index(time_index);
        }
    };

    /**
      * @brief Manager for all Dirichlet boundary conditions.
      *
      * This class reads the main Dirichlet file, creates interpolation
      * functions, assigns custom boundary ids to the triangulation, and exposes
      * the map required by deal.II:
      *
      * @code
      * std::map<types::boundary_id, const Function<dim> *>
      * @endcode
      *
      * The class is default-constructible and initially empty. This allows the
      * main NPSAT model object to own it before the input file is known.
      */
    template <int dim>
    class DirichletBoundary
    {
    public:
        using Part = DirichletPart<dim>;
        using TriaType = parallel::distributed::Triangulation<dim>;
        using CellIterator = typename TriaType::active_cell_iterator;

        DirichletBoundary() = default;

        /**
         * @brief Remove all Dirichlet parts and clear the function map.
         */
        void clear()
        {
            parts.clear();
            function_map.clear();
        }

        void set_lateral_matching_tolerances(const double half_width,
                                        const double min_overlap_length)
        {
            lateral_half_width = half_width;
            min_lateral_overlap = min_overlap_length;
        }

        /**
         * @brief Read the main Dirichlet boundary file and create all interpolation functions.
         *
         * File format:
         *
         * @code
         * N
         * TYPE nv bc_data_file
         * x1 y1
         * ...
         * xnv ynv
         * @endcode
         *
         * TYPE must be TOP, BOT, EDGE, or EDGETOP.
         * nv must always be greater than zero.
         *
         * TOP/BOT entries are stored as Boost polygons.
         * EDGE/EDGETOP entries are stored as buffered Boost polygons around
         * the input lateral polyline.
         */
        void read_data(const std::string &filename,
                       const double factor = 1.0,
                       MPI_Comm comm = MPI_COMM_WORLD,
                       const types::boundary_id first_custom_id = 17,
                       const std::string &input_path = "")
        {
            clear();

            const std::string boundary_file = resolve_relative_path(input_path, filename);
            std::string boundary_text;
            read_text_file_mpi(boundary_file, comm, boundary_text);
            std::istringstream in(boundary_text);

            unsigned int n_parts = 0;
            in >> n_parts;
            if (!in)
                throw std::runtime_error("Invalid Dirichlet file: " + boundary_file);
            parts.resize(n_parts);

            for (unsigned int i = 0; i < n_parts; ++i) {
                std::string type_string;
                int nv = 0;
                std::string bc_data_file;

                in >> type_string >> nv >> bc_data_file;

                const BoundaryKind kind = boundary_kind_from_string(type_string);
                BoundaryRegion<dim> region = read_boundary_region(in, kind, nv);

                auto interp = std::make_shared<InterpInterface<dim>>();
                interp->read_master_file(bc_data_file,
                                         factor,
                                         comm,
                                         input_path,
                                         lateral_half_width,
                                         min_lateral_overlap);

                const auto bid = static_cast<types::boundary_id>(first_custom_id + i);
                parts[i].set(kind, std::move(region), bid, std::move(interp));
            }
            build_function_map();
        }

        /**
         * @brief Rebuild the boundary-id to Function pointer map.
         *
         * This is called automatically by read_data().
         */
        void build_function_map()
        {
            function_map.clear();
            for (const auto &part : parts)
            {
                if (!part.is_initialized())
                    continue;

                function_map[part.boundary_id] = part.function.get();
            }
        }

        /**
         * @brief Return the map used by deal.II boundary utilities.
         */
        const std::map<types::boundary_id, const Function<dim> *> &
        get_function_map() const
        {
            return function_map;
        }

        /**
         * @brief Return mutable access to the map if needed by legacy code.
         */
        std::map<types::boundary_id, const Function<dim> *> &
        get_function_map()
        {
            return function_map;
        }

        /**
         * @brief Set the active time index for all Dirichlet functions.
         */
        void set_time_index(const unsigned int time_index)
        {
            for (auto &part : parts)
                part.set_time_index(time_index);
        }

        /**
         * @brief Assign custom Dirichlet ids to triangulation boundary faces.
         *
         * This method should be called after mesh generation and after each
         * refinement/coarsening cycle.
         *
         * @param triangulation Distributed triangulation.
         * @param top_boundary_ids Output list of top Dirichlet ids.
         * @param bottom_boundary_ids Output list of bottom Dirichlet ids.
         */
        void assign_to_triangulation(parallel::distributed::Triangulation<dim> &triangulation,
            std::map<types::boundary_id, const Function<dim> *> &function_map,
            const types::boundary_id first_custom_id = 17)
        {
            (void)first_custom_id;
            (void)function_map;
            // {//TODO Maybe the top and bottom boundary ids are not needed anymore
            //     top_boundary_ids.clear();
            //     bottom_boundary_ids.clear();
            //
            //     const auto top_id = static_cast<types::boundary_id>(GeometryInfo<dim>::faces_per_cell - 1);
            //     const auto bot_id = static_cast<types::boundary_id>(GeometryInfo<dim>::faces_per_cell - 2);
            //
            //     top_boundary_ids.push_back(top_id);
            //     bottom_boundary_ids.push_back(bot_id);
            // }

            for (auto cell = triangulation.begin_active(); cell != triangulation.end(); ++cell)
            {
                if (!(cell->is_locally_owned() || cell->is_ghost()))
                    continue;

                for (unsigned int iface = 0;iface < GeometryInfo<dim>::faces_per_cell; ++iface)
                {
                    auto face = cell->face(iface);

                    if (!face->at_boundary())
                        continue;

                    face->set_all_boundary_ids(iface);
                    face->set_user_index(0);

                    const int ipart = get_dirichlet_part_index_for_tria_face(cell, iface);
                    if (ipart  < 0)
                        continue;

                    const Part &part = parts[static_cast<unsigned int>(ipart)];

                    face->set_all_boundary_ids(part.boundary_id);
                    face->set_user_index(1);

                    // {// TODO if top and bottom boundary ids are not needed this should be removed
                    //     if (part.kind == BoundaryKind::TOP)
                    //         add_unique_id(top_boundary_ids, part.boundary_id);
                    //
                    //     if (part.kind == BoundaryKind::BOT)
                    //         add_unique_id(bottom_boundary_ids, part.boundary_id);
                    // }
                }
            }
        }

        /**
         * @brief Return the custom Dirichlet boundary id for a face.
         *
         * Returns -1 if the face is not part of a Dirichlet boundary.
         * This is the method to use for refinement marking.
         */
        int get_dirichlet_id_for_face(const CellIterator &cell, const unsigned int iface) const {
            const int ipart = get_dirichlet_part_index_for_tria_face(cell, iface);
            if (ipart < 0)
                return -1;

            return static_cast<int>(parts[static_cast<unsigned int>(ipart)].boundary_id);
        }

        bool face_is_dirichlet(const CellIterator &cell, const unsigned int iface) const
        {
            return get_dirichlet_part_index_for_tria_face(cell, iface) >= 0;
        }

        const std::vector<Part> &get_parts() const
        {
            return parts;
        }

        std::vector<Part> &get_parts()
        {
            return parts;
        }
    private:
        std::vector<Part> parts;
        std::map<types::boundary_id, const Function<dim> *> function_map;

        double lateral_half_width = 20.0;
        double min_lateral_overlap = 5.0;

        /**
         * @brief Read one geometry block and convert it to Boost geometry.
         */
        BoundaryRegion<dim> read_boundary_region(std::istream &in, const BoundaryKind, const int nv) const {
            if (nv <= 0)
                throw std::runtime_error("Dirichlet boundary entries must have nv > 0.");

            std::vector<Point<dim>> pts(static_cast<unsigned int>(nv));

            for (int i = 0; i < nv; ++i)
            {
                double x, y;
                in >> x >> y;

                Point<dim> p;
                p[0] = x;
                p[1] = y;

                pts[static_cast<unsigned int>(i)] = p;
            }

            BoundaryRegion<dim> region;
            region.points = std::move(pts);

            if (region.points.size() == 2)
                region.set_rectangle(region.points);
            else
                region.set_polygon(region.points);

            return region;
        }

        template <typename CellIterator>
        int get_dirichlet_part_index_for_face(const CellIterator &cell, const unsigned int iface) const
        {
            for (unsigned int ipart = 0; ipart < parts.size(); ++ipart)
            {
                const Part &part = parts[ipart];

                if (!part.is_initialized())
                    continue;

                if (!face_matches_kind(cell, iface, part.kind))
                    continue;

                if (!face_matches_region(cell, iface, part.region, part.kind))
                    continue;

                if (!face_matches_lateral_interpolant(cell, iface, part))
                    continue;

                return static_cast<int>(ipart);
            }

            return -1;
        }

        template <typename CellIterator>
        bool face_matches_kind(const CellIterator &cell, const unsigned int iface, const BoundaryKind kind) const
        {
            const unsigned int top_face =
                GeometryInfo<dim>::faces_per_cell - 1;

            const unsigned int bot_face =
                GeometryInfo<dim>::faces_per_cell - 2;

            if (kind == BoundaryKind::TOP)
                return iface == top_face ||
                       cell->face(iface)->boundary_id() == top_face;

            if (kind == BoundaryKind::BOT)
                return iface == bot_face ||
                       cell->face(iface)->boundary_id() == bot_face;

            if (kind == BoundaryKind::EDGE ||
                kind == BoundaryKind::EDGETOP)
            {
                const bool vertical_face =
                    iface != top_face && iface != bot_face;

                if (!vertical_face)
                    return false;

                if (kind == BoundaryKind::EDGETOP &&
                    !cell->face(top_face)->at_boundary())
                    return false;

                return true;
            }

            return false;
        }

        template <typename CellIterator>
        bool face_matches_region(const CellIterator &cell, const unsigned int iface,
                                 const BoundaryRegion<dim> &region, const BoundaryKind kind) const
        {
            std::vector<Point<dim>> face_points = get_face_points(cell, iface);

            if (kind == BoundaryKind::TOP || kind == BoundaryKind::BOT)
            {
                const boost_polygon face_polygon = make_face_polygon(face_points);

                return bg::intersects(face_polygon, region.polygon) ||
                       bg::within(face_polygon, region.polygon) ||
                       bg::within(region.polygon, face_polygon);
            }

            if (kind == BoundaryKind::EDGE || kind == BoundaryKind::EDGETOP)
            {
                for (const auto &p : face_points)
                    if (region.contains_point(p))
                        return true;

                return false;
            }

            return false;
        }

        template <typename CellIterator>
        static std::vector<Point<dim>> get_face_points(const CellIterator &cell, const unsigned int iface)
        {
            std::vector<Point<dim>> face_points;

            for (unsigned int iv = 0; iv < GeometryInfo<dim>::vertices_per_face; ++iv)
            {
                face_points.push_back(cell->face(iface)->vertex(iv));
            }

            return face_points;
        }

        static boost_polygon make_face_polygon(const std::vector<Point<dim>> &face_points)
        {
            boost_polygon poly;

            for (const auto &p : face_points)
                bg::append(poly.outer(), boost_point(p[0], p[1]));

            bg::append(poly.outer(),
                       boost_point(face_points.front()[0],
                                   face_points.front()[1]));

            bg::correct(poly);

            return poly;
        }

        static std::pair<Point<dim>, Point<dim>> representative_face_edge_xy(const std::vector<Point<dim>> &face_points)
        {
            constexpr double xy_tolerance = 1.0e-10;

            std::vector<Point<dim>> unique_xy_points;
            unique_xy_points.reserve(2);

            for (const auto &candidate : face_points)
            {
                bool already_seen = false;

                for (const auto &existing : unique_xy_points)
                {
                    const double dx = candidate[0] - existing[0];
                    const double dy = candidate[1] - existing[1];

                    if (std::sqrt(dx * dx + dy * dy) <= xy_tolerance)
                    {
                        already_seen = true;
                        break;
                    }
                }

                if (!already_seen)
                    unique_xy_points.push_back(candidate);
            }

            AssertThrow(unique_xy_points.size() == 2,
                        ExcMessage("Expected an extruded lateral face to have exactly two unique XY vertices."));

            const double dx = unique_xy_points[1][0] - unique_xy_points[0][0];
            const double dy = unique_xy_points[1][1] - unique_xy_points[0][1];
            const double length = std::sqrt(dx * dx + dy * dy);

            AssertThrow(length > xy_tolerance,
                        ExcMessage("Extruded lateral face has degenerate projected XY edge."));

            return {unique_xy_points[0], unique_xy_points[1]};
        }

        template <typename CellIterator>
        bool face_matches_lateral_interpolant(const CellIterator &cell,
                                              const unsigned int iface,
                                              const Part &part) const
        {
            if (part.kind != BoundaryKind::EDGE &&
                part.kind != BoundaryKind::EDGETOP)
                return true;

            const auto edge =
                representative_face_edge_xy(get_face_points(cell, iface));

            return part.interp->face_matches_lateral_overlap(edge.first,
                                                             edge.second);
        }

        int get_dirichlet_part_index_for_tria_face(const CellIterator &cell,
                                           const unsigned int iface) const
        {
            for (unsigned int ipart = 0; ipart < parts.size(); ++ipart)
            {
                const Part &part = parts[ipart];

                if (!part.is_initialized())
                    continue;

                if (!face_matches_kind(cell, iface, part.kind))
                    continue;

                if (!face_matches_region(cell, iface, part.region, part.kind))
                    continue;

                if (!face_matches_lateral_interpolant(cell, iface, part))
                    continue;

                return static_cast<int>(ipart);
            }

            return -1;
        }

    };


}

#endif //DIRICHLET_BC_H
