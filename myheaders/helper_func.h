//
// Created by giorgk on 6/23/26.
//

#ifndef HELPER_FUNC_H
#define HELPER_FUNC_H

#include <algorithm>
#include <sstream>
#include <vector>
#include <fstream>
#include <cassert>
#include <cctype>
#include <cmath>
#include <sys/stat.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include "flow_structures.h"

namespace npsat_flow{
    using namespace dealii;

    inline bool parse_double_or_file(const std::string& input, double& value, std::string& filename)
    {
        std::stringstream ss(input);

        double tmp;
        // Try parsing entire string as double
        if ((ss >> tmp) && (ss >> std::ws).eof())
        {
            value = tmp;
            filename.clear();
            return true;
        }
        // Otherwise assume filename
        filename = input;
        return false;
    }

    inline std::string trim(const std::string& s)
    {
        auto start = std::find_if_not(s.begin(), s.end(),
            [](unsigned char c) { return std::isspace(c); });

        auto end = std::find_if_not(s.rbegin(), s.rend(),
            [](unsigned char c) { return std::isspace(c); }).base();

        if (start >= end)
            return "";

        return std::string(start, end);
    }

    inline bool path_is_absolute(const std::string &path)
    {
        if (path.empty())
            return false;
        if (path.front() == '/' || path.front() == '\\')
            return true;
        return path.size() >= 3 && std::isalpha(static_cast<unsigned char>(path[0])) &&
               path[1] == ':' && (path[2] == '/' || path[2] == '\\');
    }

    inline std::string join_paths(const std::string &base, const std::string &path)
    {
        if (base.empty() || path.empty() || path_is_absolute(path))
            return path;

        if (base.back() == '/')
            return base + path;

        return base + "/" + path;
    }

    inline std::string resolve_relative_path(const std::string &base, const std::string &path)
    {
        const std::string clean_path = trim(path);
        if (clean_path.empty() || path_is_absolute(clean_path))
            return clean_path;

        return join_paths(base, clean_path);
    }

    template<typename T>
    std::vector<T> parse_list(const std::string& str, char delimiter = ',')
    {
        std::vector<T> values;

        std::stringstream ss(str);
        std::string token;

        while (std::getline(ss, token, delimiter))
        {
            token = trim(token);

            std::stringstream converter(token);

            T value;
            converter >> value;

            if (converter.fail())
            {
                throw std::runtime_error(
                    "Failed parsing value \"" + token + "\"");
            }

            values.push_back(value);
        }

        return values;
    }

    template<typename T>
    bool try_parse_list(const std::string& str, std::vector<T>& values, char delimiter = ',')
    {
        std::vector<T> parsed_values;

        std::stringstream ss(str);
        std::string token;

        while (std::getline(ss, token, delimiter))
        {
            token = trim(token);
            if (token.empty())
                return false;

            std::stringstream converter(token);

            T value;
            converter >> value;

            if (converter.fail() || !(converter >> std::ws).eof())
                return false;

            parsed_values.push_back(value);
        }

        if (parsed_values.empty())
            return false;

        values = parsed_values;
        return true;
    }

    inline bool path_exists(const std::string &path)
    {
        struct stat path_info;
        return stat(path.c_str(), &path_info) == 0;
    }
    inline bool path_is_directory(const std::string &path)
    {
        struct stat path_info;
        return stat(path.c_str(), &path_info) == 0 && S_ISDIR(path_info.st_mode);
    }

    template <typename T>
    inline T clamp_(T value, T lo, T hi)
    {
        assert(lo <= hi);
        return std::max(lo, std::min(value, hi));
    }

    inline ScreenClip well_length_inside_cell(double w_bottom, double w_top, double p_bot, double p_top)
    {
        ScreenClip c;
        c.z_low  = std::max(w_bottom, p_bot);
        c.z_high = std::min(w_top,   p_top);
        c.L      = c.z_high - c.z_low;
        if (c.L <= 0.0)
            c = ScreenClip{};
        return c;
    }

    inline void interpolate_top_bottom(
        const std::vector<double> &X,
        const std::vector<double> &Y,
        const std::vector<double> &top,
        const std::vector<double> &bot,
        double x, double y,
        double &p_top, double &p_bot)
    {
        // Solve for (u, v) using Newton iteration on bilinear mapping
        double u = 0.5, v = 0.5; // good initial guess

        for (int iter = 0; iter < 5; ++iter)
        {
            // Shape functions
            double N0 = (1-u)*(1-v);
            double N1 = u*(1-v);
            double N2 = u*v;
            double N3 = (1-u)*v;

            // Mapped x,y
            double xx = N0*X[0] + N1*X[1] + N2*X[2] + N3*X[3];
            double yy = N0*Y[0] + N1*Y[1] + N2*Y[2] + N3*Y[3];

            // Residuals
            double rx = xx - x;
            double ry = yy - y;

            // Derivatives wrt u
            double dN0du = -(1-v);
            double dN1du = (1-v);
            double dN2du = v;
            double dN3du = -v;

            // Derivatives wrt v
            double dN0dv = -(1-u);
            double dN1dv = -u;
            double dN2dv = u;
            double dN3dv = (1-u);
            double dxdu = dN0du*X[0] + dN1du*X[1] + dN2du*X[2] + dN3du*X[3];
            double dydu = dN0du*Y[0] + dN1du*Y[1] + dN2du*Y[2] + dN3du*Y[3];
            double dxdv = dN0dv*X[0] + dN1dv*X[1] + dN2dv*X[2] + dN3dv*X[3];
            double dydv = dN0dv*Y[0] + dN1dv*Y[1] + dN2dv*Y[2] + dN3dv*Y[3];

            // Solve 2x2 system for du, dv
            double det = dxdu*dydv - dydu*dxdv;
            double du = (-rx*dydv + ry*dxdv) / det;
            double dv = (-dxdu*ry + dydu*rx) / det; u += du; v += dv;
        }

        // Final shape functions
        double N0 = (1-u)*(1-v);
        double N1 = u*(1-v);
        double N2 = u*v;
        double N3 = (1-u)*v;

        // Interpolate top and bottom
        p_top = N0*top[0] + N1*top[1] + N2*top[2] + N3*top[3];
        p_bot = N0*bot[0] + N1*bot[1] + N2*bot[2] + N3*bot[3];
    }

    static void expand_trace_by_constraints(
        const types::global_dof_index        i,
        const AffineConstraints<double>     &constraints,
        std::vector<types::global_dof_index> &out)
    {
        out.clear();

        if (!constraints.is_constrained(i))
        {
            out.push_back(i);
            return;
        }

        const auto *entries = constraints.get_constraint_entries(i);
        if (entries == nullptr || entries->empty())
            return; // no structural coupling to add

        out.reserve(entries->size());
        for (const auto &e : *entries)
            out.push_back(e.first); // master dof
    }

    template<int dim>
    inline void get_face_xy_polygon(
        const typename DoFHandler<dim>::active_cell_iterator &cell,
        const unsigned int face_no,
        std::vector<double> &x,
        std::vector<double> &y)
    {
        AssertThrow(GeometryInfo<dim>::vertices_per_face == 4, ExcMessage("Face XY polygon helper expects quadrilateral faces."));

        x.clear();
        y.clear();
        x.reserve(GeometryInfo<dim>::vertices_per_face);
        y.reserve(GeometryInfo<dim>::vertices_per_face);

        for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_face; ++i)
        {
            const Point<dim> vertex = cell->face(face_no)->vertex(face_order[i]);
            x.push_back(vertex[0]);
            y.push_back(vertex[1]);
        }
    }

    template<int dim>
    inline bool map_xy_to_unit_face(
        const typename DoFHandler<dim>::active_cell_iterator &cell,
        const unsigned int face_no,
        const double x,
        const double y,
        Point<dim - 1> &unit_point)
    {
        AssertThrow(GeometryInfo<dim>::vertices_per_face == 4,
                    ExcMessage("Face point mapping helper expects quadrilateral faces."));

        double xv[4];
        double yv[4];
        for (unsigned int i = 0; i < 4; ++i)
        {
            const Point<dim> vertex = cell->face(face_no)->vertex(face_order[i]);
            xv[i] = vertex[0];
            yv[i] = vertex[1];
        }

        double u = 0.5;
        double v = 0.5;
        for (unsigned int iter = 0; iter < 12; ++iter)
        {
            const double N0 = (1.0 - u) * (1.0 - v);
            const double N1 = u * (1.0 - v);
            const double N2 = u * v;
            const double N3 = (1.0 - u) * v;

            const double xm = N0 * xv[0] + N1 * xv[1] + N2 * xv[2] + N3 * xv[3];
            const double ym = N0 * yv[0] + N1 * yv[1] + N2 * yv[2] + N3 * yv[3];

            const double rx = xm - x;
            const double ry = ym - y;
            if (std::sqrt(rx * rx + ry * ry) < 1.0e-10)
                break;

            const double dN0du = -(1.0 - v);
            const double dN1du =  (1.0 - v);
            const double dN2du =  v;
            const double dN3du = -v;

            const double dN0dv = -(1.0 - u);
            const double dN1dv = -u;
            const double dN2dv =  u;
            const double dN3dv =  (1.0 - u);

            const double dxdu = dN0du * xv[0] + dN1du * xv[1] + dN2du * xv[2] + dN3du * xv[3];
            const double dydu = dN0du * yv[0] + dN1du * yv[1] + dN2du * yv[2] + dN3du * yv[3];
            const double dxdv = dN0dv * xv[0] + dN1dv * xv[1] + dN2dv * xv[2] + dN3dv * xv[3];
            const double dydv = dN0dv * yv[0] + dN1dv * yv[1] + dN2dv * yv[2] + dN3dv * yv[3];

            const double det = dxdu * dydv - dydu * dxdv;
            if (std::abs(det) < 1.0e-24)
                return false;

            const double du = (-rx * dydv + ry * dxdv) / det;
            const double dv = (-dxdu * ry + dydu * rx) / det;

            u += du;
            v += dv;
        }

        constexpr double tol = 1.0e-8;
        if (u < -tol || u > 1.0 + tol || v < -tol || v > 1.0 + tol)
            return false;

        unit_point[0] = std::min(1.0, std::max(0.0, u));
        unit_point[1] = std::min(1.0, std::max(0.0, v));
        return true;
    }


    template<int dim>
    class SchurComplement : public Subscriptor
    {
    public:
        SchurComplement(const TrilinosWrappers::BlockSparseMatrix &system_matrix,
                        const IndexSet                            &lambda_owned,
                        const IndexSet                            &well_owned,
                        const AffineConstraints<double>           &lambda_constraints,
                        const MPI_Comm                            &mpi_comm)
        : system_matrix(system_matrix),
          lambda_constraints(lambda_constraints),
            well_owned(well_owned)
        {
            // tmp_w matches the Well space (Block 1)
            tmp_w.reinit(well_owned, mpi_comm);

            // coupling_contribution matches the Lambda/Face space (Block 0)
            Rw_vector.reinit(lambda_owned, mpi_comm);
        }

        void vmult(TrilinosWrappers::MPI::Vector &dst,
             const TrilinosWrappers::MPI::Vector &src) const
        {
            // 1. dst = S * src
            system_matrix.block(0, 0).vmult(dst, src);

            // 2. tmp_w = R^T * src  (Well-space temporary)
            system_matrix.block(0, 1).Tvmult(tmp_w, src);

            // 3. tmp_w = W^{-1} * tmp_w   (W is diagonal)
            const auto &W = system_matrix.block(1, 1);
            for (const auto i : well_owned)
            {
                double diag = W.diag_element(i);
                if (std::abs(diag) > 1e-16)
                    tmp_w(i) /= diag;
                else
                    tmp_w(i) = 0.0;
            }

            // 4. Rw_vector = R * tmp_w
            system_matrix.block(0, 1).vmult(Rw_vector, tmp_w);

            // 5. dst -= Rw_vector
            dst.add(-1.0, Rw_vector);

            // 6. Apply λ-constraints to operator (important!)
            //lambda_constraints.distribute(dst);
        }

    private:
        const TrilinosWrappers::BlockSparseMatrix &system_matrix;
        const AffineConstraints<double> &lambda_constraints;
        const IndexSet                            well_owned;   // <-- added
        mutable TrilinosWrappers::MPI::Vector      tmp_w;
        mutable TrilinosWrappers::MPI::Vector      Rw_vector;
    };

    template <typename T>
    void write_pod(std::ofstream &out, const T &v)
    {
        static_assert(std::is_trivially_copyable<T>::value,"T must be trivially copyable.");
        out.write(reinterpret_cast<const char*>(&v), sizeof(T));
    }

    inline void write_string(std::ofstream &out, const std::string &s)
    {
        const std::uint32_t n = static_cast<std::uint32_t>(s.size());
        write_pod(out, n);
        out.write(s.data(), n);
    }




}

#endif //HELPER_FUNC_H
