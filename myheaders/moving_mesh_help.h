#ifndef MOVING_MESH_HELP_H
#define MOVING_MESH_HELP_H

#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/constraint_matrix.h>


#include "mesh_struct.h"

template<int dim>
void calculate_mesh_vertices(Mesh_struct<dim>& mesh_struct,
                             TrilinosWrappers::MPI::Vector& distributed_mesh_vertices,
                             TrilinosWrappers::MPI::Vector& mesh_vertices,
                             ConstraintMatrix& mesh_constrains,
                             unsigned int n_levels,
                             MPI_Comm& communicator,
                             ConditionalOStream pcout){

    unsigned int my_rank = Utilities::MPI::this_mpi_process(mpi_communicator);
    unsigned int n_proc = Utilities::MPI::n_mpi_processes(mpi_communicator);

    pcout << "calculate_mesh_vertices" << std::endl;
    pcout << "Nlevels " << n_levels << std::endl;

    int iter = 0; int change_line = 0;

    while (true){
        int count_changes = 0;
        for (unsigned int i_level = 0; i_level < n_levels; ++i_level){

        }
    }


}

#endif // MOVING_MESH_HELP_H
