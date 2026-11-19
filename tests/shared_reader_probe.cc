// The reference headers rely on these includes from npsat_v2.cpp.
#include <sys/stat.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/numerics/data_out.h>
#include "interpolation/interp_interface.h"
#include "BC/dirichlet_bc.h"
#ifdef NPSAT_V2_REFERENCE
#include "mnwells.h"
#include "streams.h"
#else
#include "shared_sources.h"
#endif
#include <iomanip>
#include <iostream>

// Build once against v1, optionally again against an unmodified trace_dbg tree.
int main(int argc, char **argv) {
    dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
    AssertThrow(argc == 2, dealii::ExcMessage("Pass shared_flow/input directory"));
    const std::string root = argv[1];
    std::cout << std::setprecision(17);
    for (const char *name : {"top.master", "bottom.master", "conductivity.master", "recharge.master",
                             "porosity.master", "grid.master", "scatter.master", "lateral.master"}) {
        npsat_flow::InterpInterface<3> interp;
        interp.read_master_file(name, 1.0, MPI_COMM_WORLD, root);
        const dealii::Point<3> p(std::string(name) == "lateral.master" ? 0 : 25, 25, 5);
        for (unsigned step = 0; step < 2; ++step)
            std::cout << name << ':' << step << '=' << interp.interpolate(p, step) << '\n';
    }
    npsat_flow::DirichletBoundary<3> bc;
    bc.read_data("dirichlet.dat", 1.0, MPI_COMM_WORLD, 17, root);
    for (unsigned step = 0; step < 2; ++step) {
        bc.set_time_index(step);
        std::cout << "dirichlet:" << step << '='
                  << bc.get_parts()[0].function->value(dealii::Point<3>(25,25,0)) << '\n';
#ifdef NPSAT_V2_REFERENCE
        npsat_flow::MNWellCollection wells;
        wells.read_well_data(root + "/wells.csv");
        wells.Q_ts.read_data(root + "/well_rates.dat", MPI_COMM_WORLD);
        wells.set_time_step_number(step);
        for (const auto &w : wells.wells)
            std::cout << "well:" << step << '=' << w.Eid << ',' << w.x << ',' << w.y << ','
                      << w.top << ',' << w.bottom << ',' << w.q_row << ',' << wells.pumping_rate(w.q_row) << '\n';
        npsat_flow::StreamCollection<3> streams;
        streams.read_streams(root + "/streams.dat", root + "/stream_rates.dat", MPI_COMM_WORLD);
        npsat_flow::TimeSeriesData<double> rates;
        rates.read_data(root + "/stream_rates.dat", MPI_COMM_WORLD);
        // Query a containing polygon via the public API (the storage is private).
        std::vector<const npsat_flow::StreamCollection<3>::Stream *> found;
        streams.find_streams_in_polygon(found, {0,100,100,0}, {0,0,100,100});
        for (const auto *s : found)
            std::cout << "stream:" << step << '=' << s->row_id << ',' << rates.get_value_at_step(s->row_id,step) << '\n';
#else
        npsat_input::SharedSources sources;
        sources.read_wells(root + "/wells.csv", root + "/well_rates.dat", step, MPI_COMM_WORLD);
        sources.read_streams(root + "/streams.dat", root + "/stream_rates.dat", step, MPI_COMM_WORLD);
        for (const auto &w : sources.wells)
            std::cout << "well:" << step << '=' << w.id << ',' << w.x << ',' << w.y << ','
                      << w.top << ',' << w.bottom << ',' << w.row << ',' << w.rate << '\n';
        for (const auto &s : sources.streams)
            std::cout << "stream:" << step << '=' << s.row << ',' << s.rate << '\n';
#endif
    }
}
