#ifndef NPSAT_H
#define NPSAT_H

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/constraint_matrix.h>

#include "user_input.h"
#include "make_grid.h"
#include "dsimstructs.h"
#include "mesh_struct.h"
#include "dirichlet_boundary.h"
#include "steady_state.h"

using namespace dealii;

/*!
 * \brief The NPSAT class is the main class that makes use of every other functionality of this programm
 */
template <int dim>
class NPSAT{
public:
    //! The contructor requires the user inputs structure. Besides initializing all default deall parameters,
    //! the construcotr generates the initial mesh and reads the dirichlet boundary conditions.
    NPSAT(AquiferProperties<dim> AQP);

    //! The destructor frees the dof handles
    ~NPSAT();

    //! The solve method starts the simulation
    //void solve();

    //! The solve_refine method every iteration first solve and the refines the mesh according to the
    //! error criteria
    void solve_refine();



private:
    MPI_Comm                                  	mpi_communicator;
    parallel::distributed::Triangulation<dim> 	triangulation;
    DoFHandler<dim>                             dof_handler;
    FE_Q<dim>                                 	fe;

    TrilinosWrappers::MPI::Vector               locally_relevant_solution;

    // Moving mesh data
    DoFHandler<dim>                             mesh_dof_handler;
    FESystem<dim>                              	mesh_fe;
    TrilinosWrappers::MPI::Vector               mesh_vertices;
    TrilinosWrappers::MPI::Vector               distributed_mesh_vertices;
    IndexSet                                    mesh_locally_owned;
    IndexSet                                    mesh_locally_relevant;
    ConstraintMatrix                            mesh_constraints;


    void make_grid();

    AquiferProperties<dim>                      AQProps;

    Mesh_struct<dim>                            mesh_struct;

    // Boundary Conditions
    typename FunctionMap<dim>::type             dirichlet_boundary;
    BoundaryConditions::Dirichlet<dim>          DirBC;
    std::vector<int>                            top_boundary_ids;
    std::vector<int>                            bottom_boundary_ids;

    ConditionalOStream                        	pcout;
    int                                         my_rank;



};

template <int dim>
NPSAT<dim>::NPSAT(AquiferProperties<dim> AQP)
    :
    mpi_communicator (MPI_COMM_WORLD),
    triangulation (mpi_communicator,
                    typename Triangulation<dim>::MeshSmoothing
                    (Triangulation<dim>::smoothing_on_refinement |
                    Triangulation<dim>::limit_level_difference_at_vertices)),
    dof_handler (triangulation),
    fe (1),
    mesh_dof_handler (triangulation),
    mesh_fe (FE_Q<dim>(1),dim),
    AQProps(AQP),
    mesh_struct(AQP.xy_thres, AQP.z_thres),
    pcout(std::cout,(Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
{
    //user_input = CLI;
    my_rank = Utilities::MPI::this_mpi_process(mpi_communicator);
    make_grid();
    DirBC.get_from_file(AQProps.dirichlet_file_names, AQProps.Dirs.input);
}

template <int dim>
NPSAT<dim>::~NPSAT(){
    dof_handler.clear();
    mesh_dof_handler.clear();
}

template <int dim>
void NPSAT<dim>::make_grid(){
    //int my_rank = Utilities::MPI::this_mpi_process(mpi_communicator);
    AquiferGrid::GridGenerator<dim> gg(AQProps);
    gg.make_grid(triangulation);

    // set display scales only during debuging
    mesh_struct.dbg_set_scales(AQProps.dbg_scale_x, AQProps.dbg_scale_z);

    mesh_struct.updateMeshStruct(mesh_dof_handler,
                                 mesh_fe,
                                 mesh_constraints,
                                 mesh_locally_owned,
                                 mesh_locally_relevant,
                                 mesh_vertices,
                                 distributed_mesh_vertices,
                                 mpi_communicator, pcout);

    const MyFunction<dim, dim-1> top_function(AQProps.top_elevation);
    const MyFunction<dim, dim-1> bottom_function(AQProps.bottom_elevation);

    mesh_struct.compute_initial_elevations(top_function,bottom_function, AQProps.vert_discr);

    mesh_struct.updateMeshElevation(mesh_dof_handler,
                                    mesh_constraints,
                                    mesh_vertices,
                                    distributed_mesh_vertices,
                                    mpi_communicator,
                                    pcout);
    mesh_struct.printMesh(AQProps.Dirs.output, AQProps.sim_prefix + "0", my_rank, mesh_dof_handler);
}

template <int dim>
void NPSAT<dim>::solve_refine(){

    // Create a hydraulic conductivity function
    MyTensorFunction<dim>* HK_function;
    if (dim == 2){
        if (AQProps.HKuse[1] == false)
            HK_function = new MyTensorFunction<dim>(AQProps.HydraulicConductivity[0]);
        else
            HK_function = new MyTensorFunction<dim>(AQProps.HydraulicConductivity[0],
                                                    AQProps.HydraulicConductivity[1]);
    }
    else if (dim == 3){
        if (AQProps.HKuse[1] == false && AQProps.HKuse[2] == false)
            HK_function = new MyTensorFunction<dim>(AQProps.HydraulicConductivity[0]);
        else if(AQProps.HKuse[1] == false && AQProps.HKuse[2] == true)
            HK_function = new MyTensorFunction<dim>(AQProps.HydraulicConductivity[0],
                                                    AQProps.HydraulicConductivity[2]);
        else if(AQProps.HKuse[1] == true && AQProps.HKuse[2] == true)
            HK_function = new MyTensorFunction<dim>(AQProps.HydraulicConductivity[0],
                                                    AQProps.HydraulicConductivity[1],
                                                    AQProps.HydraulicConductivity[2]);
    }

    MyFunction<dim, dim-1> GR_funct(AQProps.GroundwaterRecharge);


    for (unsigned int iter = 0; iter < AQProps.solver_param.NonLinearIter ; ++iter){
        pcout << "|----------- Iteration : " << iter << " -------------|" << std::endl;

        DirBC.assign_dirichlet_to_triangulation(triangulation,
                                                dirichlet_boundary,
                                                top_boundary_ids,
                                                bottom_boundary_ids);
        GWFLOW<dim> gw(mpi_communicator,
                       dof_handler,
                       fe,
                       locally_relevant_solution,
                       dirichlet_boundary,
                       HK_function[0],
                       GR_funct,
                       top_boundary_ids);

        gw.Simulate(iter, AQProps.sim_prefix, triangulation, AQProps.wells);

    }




}


#endif // NPSAT_H
