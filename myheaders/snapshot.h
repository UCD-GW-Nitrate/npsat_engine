#ifndef SNAPSHOT_H
#define SNAPSHOT_H


template <int dim>
class Snapshot
{
public:
    Snapshot(MPI_Comm&          mpi_communicator_in,
             DoFHandler<dim>&   dof_handler_in,
             const FE_Q<dim>&   fe_in,
             TrilinosWrappers::MPI::Vector& locally_relevant_solution_in);

    void save(std::string basename, int iter,
                parallel::distributed::Triangulation<dim>& triangulation);
    void load(std::string base_name, int iter,
              parallel::distributed::Triangulation<dim>& triangulation);

private:
    MPI_Comm                                mpi_communicator;
    DoFHandler<dim>&                        dof_handler;
    const FE_Q<dim>&                        fe;
    TrilinosWrappers::MPI::Vector&          locally_relevant_solution;
    ConditionalOStream                      pcout;
};


template <int dim>
Snapshot<dim>::Snapshot(MPI_Comm&                          mpi_communicator_in,
                             DoFHandler<dim>&                   dof_handler_in,
                             const FE_Q<dim>&                   fe_in,
                             TrilinosWrappers::MPI::Vector&     locally_relevant_solution_in)
    :
    mpi_communicator(mpi_communicator_in),
    dof_handler(dof_handler_in),
    fe(fe_in),
    locally_relevant_solution(locally_relevant_solution_in),
    pcout(std::cout,(Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
{}


template <int dim>
void Snapshot<dim>::save(std::string basename, int iter,
                         parallel::distributed::Triangulation<dim> &triangulation){

    pcout << "Saving snapshot..." << std::endl << std::flush;

    std::vector<const TrilinosWrappers::MPI::Vector *> x_system(1);
    x_system[0] = &locally_relevant_solution;

    parallel::distributed::SolutionTransfer<dim, TrilinosWrappers::MPI::Vector> sol_trans(dof_handler);
    sol_trans.prepare_serialization (x_system);

    const std::string filename = (basename + "_sol_" +
                                  Utilities::int_to_string(iter, 3) + ".npsatsol");

    triangulation.save(filename.c_str());
}


template <int dim>
void Snapshot<dim>::load(std::string base_name, int iter,
                         parallel::distributed::Triangulation<dim> &triangulation){

    const std::string filename = (base_name + "_sol_" +
                                  Utilities::int_to_string(iter, 3) + ".npsatsol");
     std::ifstream  datafile(filename.c_str());
     if (!datafile.good()){
         pcout << "Can't load the snapshot " << filename << std::endl;
         return;
     }
     else{
         pcout << "Loading snapshot..." << std::endl << std::flush;
         triangulation.load(filename.c_str());
         dof_handler.distribute_dofs (fe);
         pcout << "number of locally owned cells: "
               << triangulation.n_locally_owned_active_cells() << std::endl;

         pcout << " Number of active cells: "
               << triangulation.n_global_active_cells()
               << std::endl
               << " Number of degrees of freedom: "
               << dof_handler.n_dofs()
               << std::endl << std::flush;

         IndexSet    locally_owned_dofs;
         IndexSet    locally_relevant_dofs;
         locally_owned_dofs = dof_handler.locally_owned_dofs ();
         DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
         locally_relevant_solution.reinit (locally_owned_dofs,
                                           locally_relevant_dofs, mpi_communicator);

         TrilinosWrappers::MPI::Vector distributed_system(locally_owned_dofs, mpi_communicator);
         std::vector<TrilinosWrappers::MPI::Vector *> x_system(1);
         x_system[0] = & (distributed_system);

         parallel::distributed::SolutionTransfer<dim, TrilinosWrappers::MPI::Vector> sol_trans(dof_handler);
         sol_trans.deserialize (x_system);
         locally_relevant_solution = distributed_system;
     }
}

#endif // SNAPSHOT_H
