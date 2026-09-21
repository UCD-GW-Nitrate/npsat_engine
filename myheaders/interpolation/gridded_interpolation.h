//
// Created by giorgk on 6/23/26.
//

#ifndef GRIDDED_INTERPOLATION_H
#define GRIDDED_INTERPOLATION_H

#include <deal.II/base/mpi.h>

#include "interp_helpers.h"
#include "../TimeSeries.h"
#include "../reader_helper_func.h"
#include "../helper_func.h"

namespace npsat_flow{
    using namespace dealii;

    template <int dim>
    class GriddedSpatialInterpolant final
        : public SpatialInterpolantBase<dim>
    {
    public:
        explicit GriddedSpatialInterpolant(InterpRegion2D<dim> region_in)
            : SpatialInterpolantBase<dim>(std::move(region_in))
        {}

        /**
         * @brief Read the gridded spatial interpolation definition.
         *
         * This function reads only the grid geometry / layer-definition file.
         * The interpolation values are read separately by read_values().
         *
         * The grid file stores integer IDs that map each grid cell/layer to a row
         * in the TimeSeriesData object. The TimeSeriesData file itself is no longer
         * referenced here; it is supplied in the interpolation master file as the
         * separate values file for this region.
         *
         * Grid definition file format:
         *
         * @code
         * xorig ncol
         * yorig nrow
         * dx dy
         * nlay
         *
         * IDs_layer_1              (nrow x ncol integers)
         *
         * Interface_1_2            (nrow x ncol doubles)
         *
         * IDs_layer_2              (nrow x ncol integers)
         *
         * Interface_2_3            (nrow x ncol doubles)
         *
         * IDs_layer_3              (nrow x ncol integers)
         * @endcode
         *
         * If nlay == 1, only one ID block is expected and no interface block is read.
         *
         * @param filename Grid-definition file name.
         * @param comm MPI communicator. Unused here, but kept for a common interface.
         */

        void read_spatial_data(const std::string &filename, MPI_Comm comm) override
        {
            read_grid_definition(filename, comm);
        }

        void read_values(const std::string &filename,
                     const double factor,
                     MPI_Comm comm = MPI_COMM_WORLD) override
        {
            values.read_data(filename, comm);
            values.multiply_by_factor(factor);
            check_data_consistency();
        }

        double interpolate(const Point<dim> &p, unsigned int time_index) const override
        {
            const int id = find_id(p);

            if (id < 0)
                return default_value;

            return values.get_value_at_step(id, static_cast<int>(time_index));
        }

        std::int64_t n_times() const override
        {
            return values.n_times();
        }

    private:
        struct Grid2D
        {
            double xorig = 0.0;
            double yorig = 0.0;
            double dx = 1.0;
            double dy = 1.0;

            unsigned int ncol = 0;
            unsigned int nrow = 0;
            unsigned int nlay = 0;
        };

        Grid2D grid;

        /*
         * ids[lay][row*ncol + col]
         *
         * Layer 0 is the top layer.
         */
        std::vector<std::vector<int>> ids;

        /*
         * interfaces[k][row*ncol + col]
         *
         * interfaces[0] = elevation between layer 0 and 1
         * interfaces[1] = elevation between layer 1 and 2
         *
         * Size = nlay - 1.
         */
        std::vector<std::vector<double>> interfaces;

        TimeSeriesData<double> values;

        double default_value = 0.0;

    private:
      void check_data_consistency() const
        {
            const std::int64_t n_value_rows = values.n_points();
            if (n_value_rows <= 0)
                throw std::runtime_error(
                    "Gridded interpolation data consistency error: values file has no rows.");

            if (ids.size() != grid.nlay)
            {
                throw std::runtime_error(
                    "Gridded interpolation data consistency error: grid has " +
                    std::to_string(grid.nlay) + " layers, but " +
                    std::to_string(ids.size()) + " ID blocks were read.");
            }

            const std::size_t nxy =
                static_cast<std::size_t>(grid.nrow) * grid.ncol;

            for (unsigned int lay = 0; lay < ids.size(); ++lay)
            {
                if (ids[lay].size() != nxy)
                {
                    throw std::runtime_error(
                        "Gridded interpolation data consistency error: layer " +
                        std::to_string(lay) + " has " +
                        std::to_string(ids[lay].size()) +
                        " IDs, but nrow*ncol is " + std::to_string(nxy) + ".");
                }

                for (std::size_t ij = 0; ij < ids[lay].size(); ++ij)
                {
                    const int row_id = ids[lay][ij];

                    if (row_id < 0)
                        continue;

                    if (static_cast<std::int64_t>(row_id) >= n_value_rows)
                    {
                        throw std::runtime_error(
                            "Gridded interpolation data consistency error: layer " +
                            std::to_string(lay) + ", cell index " +
                            std::to_string(ij) + " references value row " +
                            std::to_string(row_id) + ", but the values file has only " +
                            std::to_string(n_value_rows) + " rows.");
                    }
                }
            }
        }

        std::size_t flat_index(const unsigned int row, const unsigned int col) const
      {
          return static_cast<std::size_t>(row) * grid.ncol + col;
      }

        bool row_col_from_xy(const double x, const double y,
                           unsigned int &row, unsigned int &col) const
      {
          const double fx = (x - grid.xorig) / grid.dx;
          const double fy = (y - grid.yorig) / grid.dy;

          int col_i = static_cast<int>(std::floor(fx));
          int row_i = static_cast<int>(std::floor(fy));

          col_i = clamp_(col_i, 0, static_cast<int>(grid.ncol) - 1);
          row_i = clamp_(row_i, 0, static_cast<int>(grid.nrow) - 1);

          col = static_cast<unsigned int>(col_i);
          row = static_cast<unsigned int>(row_i);

          return true;
      }

        int find_layer(const double z, const std::size_t ij) const
      {
          if (grid.nlay == 1)
              return 0;
          for (unsigned int k = 0; k < grid.nlay - 1; ++k)
          {
              const double z_interface = interfaces[k][ij];

              if (z >= z_interface)
                  return static_cast<int>(k);
          }
          return static_cast<int>(grid.nlay - 1);
      }

        int find_id(const Point<dim> &p) const
      {
          unsigned int row = 0;
          unsigned int col = 0;

          row_col_from_xy(p[0], p[1], row, col);

          const std::size_t ij = flat_index(row, col);

          int lay = 0;
          if (grid.nlay > 1)
              lay = find_layer(p[2], ij);

          return ids[static_cast<unsigned int>(lay)][ij];
      }

        void read_grid_definition(const std::string &filename, MPI_Comm comm)
      {
          std::string file_text;
          read_text_file_mpi(filename, comm, file_text);
          std::istringstream in(file_text);

          in >> grid.xorig >> grid.ncol;
          in >> grid.yorig >> grid.nrow;
          in >> grid.dx >> grid.dy;
          in >> grid.nlay;
          if (!in)
              throw std::runtime_error("Invalid grid definition file: " + filename);

          if (grid.ncol == 0 || grid.nrow == 0 || grid.nlay == 0)
              throw std::runtime_error("Invalid grid dimensions in: " + filename);

          const std::size_t nxy =
              static_cast<std::size_t>(grid.nrow) * grid.ncol;

          ids.resize(grid.nlay);
          interfaces.resize(grid.nlay > 0 ? grid.nlay - 1 : 0);

          for (unsigned int lay = 0; lay < grid.nlay; ++lay)
          {
              ids[lay].resize(nxy);

              for (unsigned int r = 0; r < grid.nrow; ++r)
                  for (unsigned int c = 0; c < grid.ncol; ++c)
                      in >> ids[lay][flat_index(r, c)];

              if (lay < grid.nlay - 1)
              {
                  interfaces[lay].resize(nxy);

                  for (unsigned int r = 0; r < grid.nrow; ++r)
                      for (unsigned int c = 0; c < grid.ncol; ++c)
                          in >> interfaces[lay][flat_index(r, c)];
              }
          }
      }


    };



}

#endif //GRIDDED_INTERPOLATION_H
