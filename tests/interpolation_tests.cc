#include "../myheaders/interpolation/interpolation_function.h"
#include "../myheaders/conductivity_function.h"
#include "../myheaders/BC/dirichlet_bc.h"
#include "../myheaders/neumann_boundary.h"
#include <deal.II/grid/grid_generator.h>
#include <cmath>
#include <iostream>

void check(double actual, double expected) {
    AssertThrow(std::abs(actual - expected) < 1e-10,
                dealii::ExcMessage("Expected " + std::to_string(expected) + ", got " + std::to_string(actual)));
}

int main(int argc, char **argv) {
    dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
    AssertThrow(argc == 2, dealii::ExcMessage("Pass the interpolation_v2 fixture directory"));
    const std::string root = argv[1];
    auto read = [&](const std::string &name) {
        auto interp = std::make_shared<npsat_flow::InterpInterface<3>>();
        interp->read_master_file(name, 1.0, MPI_COMM_WORLD, root);
        return interp;
    };
    const dealii::Point<3> p(25, 25, 5);
    auto constant = read("constant.master");
    check(constant->n_times(), 2);
    npsat_flow::InterpolationFunction<3> function(constant);
    check(function.value(p), 10);
    function.set_time_index(1);
    auto copied_function = function;
    constant.reset();
    check(copied_function.value(p), 20);
    check(function.value(dealii::Point<3>(200, 200, 0)), 0);
    auto grid = read("grid.master");
    check(grid->interpolate(p, 0), 10);
    check(grid->interpolate(p, 1), 20);
    auto scatter = read("scatter.master");
    check(scatter->interpolate(p, 0), 17.5);
    check(scatter->interpolate(p, 1), 35);
    npsat_flow::ScatteredSpatialInterpolant<3> surface{npsat_flow::InterpRegion2D<3>()};
    std::map<int, dealii::Point<3>> surface_points = {{4,{0,0,10}}, {9,{100,0,20}}, {15,{0,100,30}}};
    surface.set_surface_data(surface_points, {{4,9,15}});
    check(surface.interpolate(p, 0), 17.5);
    auto lateral = read("lateral.master");
    check(lateral->interpolate(dealii::Point<3>(0, 25, 5), 0), 12.5);
    AssertThrow(lateral->face_matches_lateral_overlap(dealii::Point<3>(0, 0, 5), dealii::Point<3>(0, 100, 5)), dealii::ExcMessage("Lateral face not matched"));
    ConductivityFunction<3> tensor(grid, scatter, grid);
    check(tensor.value(p)[0][0], 10);
    check(tensor.value(p)[1][1], 17.5);
    check(tensor.value(p)[2][2], 10);
    tensor.set_time_index(1);
    check(tensor.value(p)[0][0], 20);
    ConductivityFunction<3> isotropic(grid);
    check(isotropic.value(p)[2][2], 10);
    ConductivityFunction<3> vertical(grid, scatter);
    check(vertical.value(p)[1][1], 10);
    check(vertical.value(p)[2][2], 17.5);
    npsat_flow::InterpInterface<2> flat;
    flat.read_master_file("constant.master", 1.0, MPI_COMM_WORLD, root);
    check(flat.interpolate(dealii::Point<2>(25,25), 1), 20);
    dealii::parallel::distributed::Triangulation<3> tria(MPI_COMM_WORLD);
    dealii::GridGenerator::hyper_rectangle(tria, dealii::Point<3>(0,0,0), dealii::Point<3>(100,100,10), true);
    npsat_flow::DirichletBoundary<3> bc;
    bc.read_data("dirichlet.dat", 1.0, MPI_COMM_WORLD, 17, root);
    std::map<dealii::types::boundary_id,const dealii::Function<3> *> functions;
    bc.assign_to_triangulation(tria, functions);
    functions = bc.get_function_map();
    check(functions.at(17)->value(p), 10);
    unsigned int matched = 0;
    for (auto cell : tria.active_cell_iterators())
        if (cell->is_locally_owned())
            matched += cell->face(5)->boundary_id() == 17;
    check(dealii::Utilities::MPI::sum(matched, MPI_COMM_WORLD), 1);
    BoundaryConditions::Neumann<3> neumann;
    AssertThrow(neumann.getData(root + "/neumann.dat", root, MPI_COMM_WORLD), dealii::ExcMessage("Neumann read failed"));
    auto copied_neumann = neumann;
    check(copied_neumann.interpolate(p, 0), 20);
    AssertThrow(copied_neumann.isPartofBoundary(p, 0), dealii::ExcMessage("Bottom face not matched"));
    check(copied_neumann.interpolate(dealii::Point<3>(0,25,5), 1), 12.5);
    std::cout << "Interpolation integration checks passed\n";
}
