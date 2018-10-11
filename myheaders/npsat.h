#ifndef NPSAT_H
#define NPSAT_H

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/numerics/error_estimator.h>

#include "user_input.h"
#include "make_grid.h"
#include "dsimstructs.h"
#include "mesh_struct.h"
#include "dirichlet_boundary.h"
#include "steady_state.h"
#include "mix_mesh.h"
#include "particle_tracking.h"
#include "streamlines.h"

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

    //! This method does the actuall refinement and all the required actions that are associated with
    //! such as transfer the refinement to other processors etc.
    void do_refinement();

    void do_refinement1();

    void particle_tracking();



private:
    MPI_Comm                                  	mpi_communicator;
    parallel::distributed::Triangulation<dim> 	triangulation;
    DoFHandler<dim>                             dof_handler;
    FE_Q<dim>                                 	fe;
    ConstraintMatrix                          	Headconstraints;

    TrilinosWrappers::MPI::Vector               locally_relevant_solution;

    // Moving mesh data
    DoFHandler<dim>                             mesh_dof_handler;
    FESystem<dim>                              	mesh_fe;
    TrilinosWrappers::MPI::Vector               mesh_vertices;
    TrilinosWrappers::MPI::Vector               distributed_mesh_vertices;
    TrilinosWrappers::MPI::Vector               mesh_Offset_vertices;
    TrilinosWrappers::MPI::Vector               distributed_mesh_Offset_vertices;
    IndexSet                                    mesh_locally_owned;
    IndexSet                                    mesh_locally_relevant;
    ConstraintMatrix                            mesh_constraints;
    mix_mesh<dim-1>                             top_grid;
    mix_mesh<dim-1>                             bottom_grid;


    AquiferProperties<dim>                      AQProps;

    Mesh_struct<dim>                            mesh_struct;

    // Boundary Conditions
    typename FunctionMap<dim>::type             dirichlet_boundary;
    BoundaryConditions::Dirichlet<dim>          DirBC;
    std::vector<int>                            top_boundary_ids;
    std::vector<int>                            bottom_boundary_ids;

    ConditionalOStream                        	pcout;
    int                                         my_rank;

    void make_grid();
    void create_dim_1_grids();
    void flag_cells_for_refinement();
    void print_mesh();
    void save_solution();
    void load_solution();


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
    mesh_struct.folder_Path = AQProps.Dirs.output;
}

template <int dim>
void NPSAT<dim>::make_grid(){
    //int my_rank = Utilities::MPI::this_mpi_process(mpi_communicator);
    AquiferGrid::GridGenerator<dim> gg(AQProps);
    gg.make_grid(triangulation);

    // set display scales only during debuging
    mesh_struct.dbg_set_scales(AQProps.dbg_scale_x, AQProps.dbg_scale_z);
    mesh_struct.prefix = "iter0";
    mesh_struct.updateMeshStruct(mesh_dof_handler,
                                 mesh_fe,
                                 mesh_constraints,
                                 mesh_locally_owned,
                                 mesh_locally_relevant,
                                 mesh_vertices,
                                 distributed_mesh_vertices,
                                 mesh_Offset_vertices,
                                 distributed_mesh_Offset_vertices,
                                 mpi_communicator, pcout);

    const MyFunction<dim, dim> top_function(AQProps.top_elevation);
    const MyFunction<dim, dim> bottom_function(AQProps.bottom_elevation);

    mesh_struct.compute_initial_elevations(top_function,bottom_function);

    mesh_struct.updateMeshElevation(mesh_dof_handler,
                                    triangulation,
                                    mesh_constraints,
                                    mesh_vertices,
                                    distributed_mesh_vertices,
                                    mesh_Offset_vertices,
                                    distributed_mesh_Offset_vertices,
                                    mpi_communicator,
                                    pcout);

    unsigned int count_refinements = 0;
    while (true){
        bool done_wells = false;
        bool done_streams = false;


        if (count_refinements < AQProps.N_well_refinement){
            AQProps.wells.flag_cells_for_refinement(triangulation);
        }
        else
            done_wells = true;

        if(count_refinements < AQProps.N_streams_refinement){
            AQProps.streams.flag_cells_for_refinement(triangulation);
        }
        else
            done_streams = true;

        if (done_wells && done_streams)
            break;

        do_refinement1();

        mesh_struct.updateMeshStruct(mesh_dof_handler,
                                     mesh_fe,
                                     mesh_constraints,
                                     mesh_locally_owned,
                                     mesh_locally_relevant,
                                     mesh_vertices,
                                     distributed_mesh_vertices,
                                     mesh_Offset_vertices,
                                     distributed_mesh_Offset_vertices,
                                     mpi_communicator, pcout);
        mesh_struct.compute_initial_elevations(top_function,bottom_function);

        mesh_struct.updateMeshElevation(mesh_dof_handler,
                                        triangulation,
                                        mesh_constraints,
                                        mesh_vertices,
                                        distributed_mesh_vertices,
                                        mesh_Offset_vertices,
                                        distributed_mesh_Offset_vertices,
                                        mpi_communicator,
                                        pcout);

        count_refinements++;
    }
}

template <int dim>
void NPSAT<dim>::solve_refine(){

    // Create a hydraulic conductivity function
    MyTensorFunction<dim>* HK_function = 0;
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

    MyFunction<dim, dim> GR_funct(AQProps.GroundwaterRecharge);


    for (int iter = 0; iter < AQProps.solver_param.NonLinearIter ; ++iter){
        pcout << "|----------- Iteration : " << iter << " -------------|" << std::endl;

        DirBC.assign_dirichlet_to_triangulation(triangulation,
                                                dirichlet_boundary,
                                                top_boundary_ids,
                                                bottom_boundary_ids);
        GWFLOW<dim> gw(mpi_communicator,
                       dof_handler,
                       fe,
                       Headconstraints,
                       locally_relevant_solution,
                       dirichlet_boundary,
                       HK_function[0],
                       GR_funct,
                       top_boundary_ids);

        gw.Simulate(iter,
                    AQProps.Dirs.output + AQProps.sim_prefix,
                    triangulation, AQProps.wells, AQProps.streams);

        //gw.Simulate_refine(iter,
        //                AQProps.Dirs.output + AQProps.sim_prefix,
        //                triangulation, AQProps.wells /*,
        //                SourceSinks::Streams&                      streams*/,
        //                AQProps.refine_param.TopFraction, AQProps.refine_param.BottomFraction);


        if (iter < AQProps.solver_param.NonLinearIter - 1){
            create_dim_1_grids();
            if (iter < AQProps.refine_param.MaxRefinement)
                flag_cells_for_refinement();
            do_refinement1();

            mesh_struct.prefix = "iter" + std::to_string(iter);
            mesh_struct.updateMeshStruct(mesh_dof_handler,
                                         mesh_fe,
                                         mesh_constraints,
                                         mesh_locally_owned,
                                         mesh_locally_relevant,
                                         mesh_vertices,
                                         distributed_mesh_vertices,
                                         mesh_Offset_vertices,
                                         distributed_mesh_Offset_vertices,
                                         mpi_communicator, pcout);

            mesh_struct.assign_top_bottom(top_grid, bottom_grid, pcout, mpi_communicator);
            mesh_struct.updateMeshElevation(mesh_dof_handler,
                                            triangulation,
                                            mesh_constraints,
                                            mesh_vertices,
                                            distributed_mesh_vertices,
                                            mesh_Offset_vertices,
                                            distributed_mesh_Offset_vertices,
                                            mpi_communicator,
                                            pcout);
            //print_mesh();

        }
    }
    //save_solution();
}

template <int dim>
void NPSAT<dim>::create_dim_1_grids(){
    pcout << "Create 2D grids..." << std::endl << std::flush;
    top_grid.reset();
    bottom_grid.reset();
    std::vector<double> new_old_elev(2);

    int point_counter_top = 0;
    int point_counter_bottom = 0;
    std::vector<int>	tempcell;
    std::vector<int>ind;
    tempcell.clear(); ind.clear();
    if (dim == 2){
        tempcell.resize(2);
        ind.resize(2);
        ind.resize(2);
        ind[0]=0; ind[1]=1;
    }
    else if (dim == 3){
        tempcell.resize(4);
        ind.resize(4);
        ind[0]=0; ind[1]=1; ind[2]=3; ind[3]=2;
    }

    QTrapez<dim-1> face_trapez_formula; // In trapezoid quadrature the quadrature points coincide with the cell vertices
    FEFaceValues<dim> fe_face_values(fe, face_trapez_formula, update_values);
    std::vector< double > values(face_trapez_formula.size());
    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell){
        if (cell->is_locally_owned()){// || cell->is_ghost()
            for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face){
                if(cell->face(face)->at_boundary()){
                    bool is_top = false;
                    bool is_bottom = false;
                    for (unsigned int ibnd = 0; ibnd < top_boundary_ids.size(); ++ibnd){
                        if (static_cast<int>(cell->face(face)->boundary_id()) == top_boundary_ids[ibnd]){
                            is_top = true;
                            break;
                        }
                    }
                    if (!is_top){
                        for (unsigned int ibnd = 0; ibnd < bottom_boundary_ids.size(); ++ibnd){
                            if (static_cast<int>(cell->face(face)->boundary_id()) == bottom_boundary_ids[ibnd]){
                                is_bottom = true;
                                break;
                            }
                        }
                    }

                    if (is_top){
                        fe_face_values.reinit (cell, face);
                        fe_face_values.get_function_values(locally_relevant_solution, values);
                        for (unsigned int ii = 0; ii < face_trapez_formula.size(); ++ii){
                            Point<dim> temp_point_dim = cell->face(face)->vertex(ii);
                            Point<dim-1> temp_point_dim_1;// this is the point projected in dim-1 dimension
                            for (unsigned int kk = 0; kk < dim-1; ++kk){
                                temp_point_dim_1[kk] =temp_point_dim[kk];
                            }

                            int id = is_point_in_list<dim-1>(temp_point_dim_1, top_grid.P, 1e-3);
                            if (id < 0){
                                top_grid.add_point(temp_point_dim_1);
                                new_old_elev[0] = values[ii];
                                new_old_elev[1] = temp_point_dim[dim-1];
                                top_grid.data_point.push_back(new_old_elev);
                                tempcell[static_cast<unsigned int>(ind[ii])] = point_counter_top;
                                point_counter_top++;
                            }
                            else{
                                tempcell[static_cast<unsigned int>(ind[ii])] = id;
                            }
                        }
                        top_grid.add_element(tempcell);

                    }
                    else if (is_bottom){
                        for (unsigned int ii = 0; ii < face_trapez_formula.size(); ++ii){
                            Point<dim> temp_point_dim = cell->face(face)->vertex(ii);
                            Point<dim-1> temp_point_dim_1;// this is the point projected in dim-1 dimension
                            for (unsigned int kk = 0; kk < dim-1; ++kk){
                                temp_point_dim_1[kk] =temp_point_dim[kk];
                            }

                            int id = is_point_in_list<dim-1>(temp_point_dim_1, bottom_grid.P, 1e-3);
                            if (id < 0){
                                //if (std::abs(temp_point_dim_1[0]-4250.0) < 0.1 && std::abs(temp_point_dim_1[1]-2250.0) < 0.1){
                                //    std::cout << "@$#%&$#%&^@%$&@#%$&#@%$ Rank " << my_rank << " has found point 4250,2250" << std::endl;
                                //}
                                bottom_grid.add_point(temp_point_dim_1);
                                bottom_grid.data_point.push_back(std::vector<double>(1,temp_point_dim[dim-1]));
                                tempcell[static_cast<unsigned int>(ind[ii])] = point_counter_bottom;
                                point_counter_bottom++;
                            }
                            else{
                                tempcell[static_cast<unsigned int>(ind[ii])] = id;
                            }
                        }
                        bottom_grid.add_element(tempcell);
                    }
                }
            }
        }
    }

    top_grid.Np = top_grid.P.size();
    top_grid.Nel = top_grid.MSH.size();
    bottom_grid.Np = bottom_grid.P.size();
    bottom_grid.Nel = bottom_grid.MSH.size();
    //std::cout << "Rank " << my_rank << " has (" << top_grid.Np << "," << top_grid.Nel << ") top and (" << bottom_grid.Np << "," << bottom_grid.Nel << ") bottom" << std::endl;

    //for (unsigned int i = 0; i < top_grid.Np; ++i){
    //    std::cout << "R( " << my_rank << "): " << top_grid.P[i] << " -> " << top_grid.data_point[i][0] << std::endl;
    //}
}

template <int dim>
void NPSAT<dim>::flag_cells_for_refinement(){
    Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate(dof_handler,
                                     QGauss<dim-1>(fe.degree+2),
                                     typename FunctionMap<dim>::type(),
                                     locally_relevant_solution,
                                     estimated_error_per_cell);

    parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                             estimated_error_per_cell,
                                             AQProps.refine_param.TopFraction,
                                             AQProps.refine_param.BottomFraction);

    typename parallel::distributed::Triangulation<dim>::active_cell_iterator
    cell = triangulation.begin_active(),
    endc = triangulation.end();
    for (; cell!=endc; ++cell){
        if (cell->is_locally_owned()){
            double min_dim = cell->minimum_vertex_distance();
            if (min_dim/2 < AQProps.refine_param.MinElementSize)
                cell->clear_refine_flag();
        }
    }

}

template <int dim>
void NPSAT<dim>::do_refinement(){
    // first prepare the triangulation
    triangulation.prepare_coarsening_and_refinement();

    //prepare vertices for transfering
    parallel::distributed::SolutionTransfer<dim, TrilinosWrappers::MPI::Vector>mesh_trans(mesh_dof_handler);
    std::vector<const TrilinosWrappers::MPI::Vector *> x_fs_system (1);

    x_fs_system[0] = &mesh_vertices;
    mesh_trans.prepare_for_coarsening_and_refinement(x_fs_system);

    std::cout << "Number of active cells Before: "
              << triangulation.n_active_cells()
              << std::endl;
    // execute the actual refinement
    triangulation.execute_coarsening_and_refinement ();

    std::cout << "Number of active cells After: "
              << triangulation.n_active_cells()
              << std::endl;

    //For the mesh
    mesh_dof_handler.distribute_dofs(mesh_fe); // distribute the dofs again
    mesh_locally_owned = mesh_dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs (mesh_dof_handler, mesh_locally_relevant);

    distributed_mesh_vertices.reinit(mesh_locally_owned, mpi_communicator);
    distributed_mesh_vertices.compress(VectorOperation::insert);

    std::vector<TrilinosWrappers::MPI::Vector *> mesh_tmp (1);
    mesh_tmp[0] = &(distributed_mesh_vertices);

    mesh_trans.interpolate (mesh_tmp);
    //mesh_constraints.distribute(mesh_tmp[0]);
    mesh_vertices.reinit (mesh_locally_owned, mesh_locally_relevant, mpi_communicator);
    mesh_vertices = distributed_mesh_vertices;
    pcout << "moving vertices " << std::endl << std::flush;
    mesh_struct.move_vertices(mesh_dof_handler,
                             mesh_vertices);

}

template <int dim>
void NPSAT<dim>::do_refinement1(){

    std::vector<bool> locally_owned_vertices = triangulation.get_used_vertices();
    {
        // Create the boolean input of communicate_locally_moved_vertices method
        // see implementation of GridTools::get_locally_owned_vertices in grid_tools.cc line 2172 (8.5.0)
        typename parallel::distributed::Triangulation<dim>::active_cell_iterator
        cell = triangulation.begin_active(),
        endc = triangulation.end();
        for (; cell!=endc; ++cell){
            if (cell->is_artificial() ||
                    (cell->is_ghost() && cell->subdomain_id() < triangulation.locally_owned_subdomain() )){
                for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
                    locally_owned_vertices[cell->vertex_index(v)] = false;

            }
        }
    }

    // Call the method before
    triangulation.communicate_locally_moved_vertices(locally_owned_vertices);
    //{
    //    std::ofstream out ("test_triaD" + std::to_string(my_rank) + ".vtk");
    //    GridOut grid_out;
    //    grid_out.write_ucd(triangulation, out);
    //}

    {// Apply the opposite displacement
        std::map<types::global_dof_index, bool> set_dof;
        std::map<types::global_dof_index, bool>::iterator it_set;
        typename DoFHandler<dim>::active_cell_iterator
        cell = mesh_dof_handler.begin_active(),
        endc = mesh_dof_handler.end();
        for (; cell != endc; ++cell){
            if (cell->is_locally_owned()){
                for (unsigned int vertex_no = 0; vertex_no < GeometryInfo<dim>::vertices_per_cell; ++vertex_no){
                    Point<dim> &v=cell->vertex(vertex_no);
                    for (unsigned int dir=0; dir < dim; ++dir){
                        types::global_dof_index dof = cell->vertex_dof_index(vertex_no, dir);
                        it_set = set_dof.find(dof);
                        if (it_set == set_dof.end()){
                            //std::cout << dof << ": " << v(dir) << ", " << mesh_Offset_vertices(dof) << std::endl;
                            v(dir) = v(dir) - mesh_Offset_vertices(dof);
                            //std::cout << v(dir) << std::endl;
                            set_dof[dof] = true;
                        }
                    }
                }
            }
        }
    }
    //{
    //    std::ofstream out ("test_triaE" + std::to_string(my_rank) + ".vtk");
    //    GridOut grid_out;
    //    grid_out.write_ucd(triangulation, out);
    //}
    triangulation.communicate_locally_moved_vertices(locally_owned_vertices);


    // now the mesh should consistent as when it was first created
    // so we can hopefully refine it
    triangulation.execute_coarsening_and_refinement ();
    //{
    //    std::ofstream out ("test_triaE" + std::to_string(my_rank) + ".vtk");
    //    GridOut grid_out;
    //    grid_out.write_ucd(triangulation, out);
    //}
}

template <int dim>
void NPSAT<dim>::particle_tracking(){
    pcout << "PARTICLE TRACKING..." << std::endl;
    unsigned int n_proc = Utilities::MPI::n_mpi_processes(mpi_communicator);

    MyFunction<dim, dim> porosity_fnc(AQProps.Porosity);

    Particle_Tracking<dim> pt(mpi_communicator,
                         dof_handler, fe,
                         Headconstraints,
                         locally_relevant_solution,
                         AQProps.HK_function[0],
                         porosity_fnc,
                         AQProps.part_param);

    //pt.average_velocity_field(velocity_dof_handler,velocity_fe);
    pt.average_velocity_field();

    std::vector<Streamline<dim>> All_streamlines;
    std::vector<std::vector<Streamline<dim>>> part_of_streamlines(n_proc);

    MPI_Barrier(mpi_communicator);
    if (my_rank == 0){
        AQProps.wells.distribute_particles(All_streamlines,
                                           AQProps.part_param.Wells_N_per_layer,
                                           AQProps.part_param.Wells_N_Layers,
                                           AQProps.part_param.radius);
    }
    MPI_Barrier(mpi_communicator);


    int part_done[1];
    part_done[0] = 0;
    int particle_iter = 0;
    while (true){
        part_of_streamlines[my_rank].clear();
        MPI_Barrier(mpi_communicator);
        if (my_rank == 0){
            std::default_random_engine generator;
            // create a subvector of streamlines
            unsigned int N = AQProps.part_param.Nparallel_particles;
            if (All_streamlines.size() < N + 10){ // This should be 1000, we set it to 10 for debug
                N = All_streamlines.size();
            }
            for (unsigned int i = 0 ; i < N; ++i){
                if (All_streamlines.size() == 0)
                    break;

                std::uniform_int_distribution<int> distribution(0, All_streamlines.size()-1);
                int ii = distribution(generator);
                part_of_streamlines[my_rank].push_back(All_streamlines[ii]);
                All_streamlines.erase(All_streamlines.begin() + ii);
            }
            if (All_streamlines.size() == 0)
                part_done[0] = 1;
        }

        MPI_Barrier(mpi_communicator);
        Sent_receive_streamlines_all_to_all(part_of_streamlines, my_rank, n_proc, mpi_communicator);
        std::cout << "I'm proc " << my_rank << " and have " << part_of_streamlines[my_rank].size() << " to trace" << std::endl;
        MPI_Barrier(mpi_communicator);

        pt.trace_particles(part_of_streamlines[my_rank], particle_iter++, AQProps.Dirs.output + AQProps.sim_prefix);

        // Processor 0 which is responsible to send out the streamlines will
        // broadcast if there are more streamlines to trace
        MPI_Bcast( part_done, 1, MPI_INT, 0, mpi_communicator);
        if (part_done[0] == 1)
            break;
    }
}

template <int dim>
void NPSAT<dim>::print_mesh(){
    pcout << "\t Printing mesh only..." << std::endl << std::flush;
    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    Vector<float> subdomain (triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i){
        subdomain(i) = triangulation.locally_owned_subdomain();
    }
    data_out.add_data_vector (subdomain, "subdomain");
    data_out.build_patches ();

    const std::string filename = ("Current_Mesh_" +
                                  Utilities::int_to_string
                                  (triangulation.locally_owned_subdomain(), 4));

    std::ofstream output ((filename + ".vtu").c_str());
    data_out.write_vtu (output);
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0){
        std::vector<std::string> filenames;
        for (unsigned int i=0; i < Utilities::MPI::n_mpi_processes(mpi_communicator); ++i){
            filenames.push_back ("Current_Mesh_" +
                                 Utilities::int_to_string
                                 (triangulation.locally_owned_subdomain(), 4) +
                                 ".vtu");
        }
        const std::string pvtu_master_filename = ("Current_Mesh_.pvtu");
        std::ofstream pvtu_master (pvtu_master_filename.c_str());
        data_out.write_pvtu_record(pvtu_master, filenames);
    }
}

template <int dim>
void NPSAT<dim>::save_solution(){
    std::vector<const TrilinosWrappers::MPI::Vector *> x_system(1);
    x_system[0] = &locally_relevant_solution;

    parallel::distributed::SolutionTransfer<dim, TrilinosWrappers::MPI::Vector> solution_tranfser(dof_handler);
    solution_tranfser.prepare_serialization(x_system);
    std::string outfile = AQProps.Dirs.output + AQProps.sim_prefix + ".sol";
    triangulation.save(outfile.c_str());
}

#endif // NPSAT_H
