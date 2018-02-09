#ifndef STEADY_STATE_H
#define STEADY_STATE_H

#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/dofs/function_map.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/numerics/vector_tools.h>

#include "my_functions.h"

using namespace dealii;

template<int dim>
class GWFLOW{
public:
    GWFLOW(MPI_Comm&                            mpi_communicator_in,
           DoFHandler<dim>&                     dof_handler,
           const FE_Q<dim>&                     fe,
           TrilinosWrappers::MPI::Vector&       locally_relevant_solution,
           typename FunctionMap<dim>::type&     dirichlet_boundary,
           MyTensorFunction<dim>&               HK_function,
           MyFunction<dim,dim-1>&               groundwater_recharge,
           std::vector<int>&                    top_boundary_ids);


    void Simulate(int iter,                                     std::string output_file,
                  parallel::distributed::Triangulation<dim>& 	triangulation/*,
                    SourceSinks::WELLS&                         wells,
                    SourceSinks::Streams&                      streams*/);

private:
    MPI_Comm                                    mpi_communicator;
    DoFHandler<dim>&                            dof_handler;
    const FE_Q<dim>&                            fe;
    IndexSet                                  	locally_owned_dofs;
    IndexSet                                  	locally_relevant_dofs;
    TrilinosWrappers::MPI::Vector& 				locally_relevant_solution;
    TrilinosWrappers::SparseMatrix 			  	system_matrix;
    TrilinosWrappers::MPI::Vector       	  	system_rhs;
    ConstraintMatrix                          	constraints;
    typename FunctionMap<dim>::type				dirichlet_boundary;
    MyTensorFunction<dim>	 					HK;
    MyFunction<dim,dim-1> 						GWRCH;
    std::vector<int>                            top_boundary_ids;
    ConditionalOStream                        	pcout;
    TimerOutput                               	computing_timer;

    void setup_system();
    void assemble();
    void solve();
    void output(int iter, std::string output_file,
                parallel::distributed::Triangulation<dim>& 	triangulation);
};

template <int dim>
GWFLOW<dim>::GWFLOW(MPI_Comm&                            mpi_communicator_in,
                    DoFHandler<dim>&                     dof_handler_in,
                    const FE_Q<dim>&                     fe_in,
                    TrilinosWrappers::MPI::Vector&       locally_relevant_solution_in,
                    typename FunctionMap<dim>::type&     dirichlet_boundary_in,
                    MyTensorFunction<dim>&               HK_function,
                    MyFunction<dim,dim-1>&               groundwater_recharge,
                    std::vector<int>&                    top_boundary_ids_in)
    :
      mpi_communicator(mpi_communicator_in),
      dof_handler(dof_handler_in),
      fe(fe_in),
      locally_relevant_solution(locally_relevant_solution_in),
      dirichlet_boundary(dirichlet_boundary_in),
      HK(HK_function),
      GWRCH(groundwater_recharge),
      top_boundary_ids(top_boundary_ids_in),
      pcout(std::cout,(Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
      computing_timer(pcout, TimerOutput::summary, TimerOutput::wall_times)
{}

template <int dim>
void GWFLOW<dim>::setup_system(){
    TimerOutput::Scope t(computing_timer, "setup");
    pcout << "Setting up system..." << std::endl << std::flush;
    dof_handler.distribute_dofs (fe);
    pcout   << " Number of degrees of freedom: "
            << dof_handler.n_dofs()
            << std::endl << std::flush;

    locally_owned_dofs = dof_handler.locally_owned_dofs ();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    locally_relevant_solution.reinit (locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

    system_rhs.reinit (locally_owned_dofs, mpi_communicator);
    system_rhs = 0;

    constraints.clear ();
    constraints.reinit (locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints (dof_handler, constraints);
    VectorTools::interpolate_boundary_values(dof_handler, dirichlet_boundary, constraints);
    constraints.close ();

}

#endif // STEADY_STATE_H
