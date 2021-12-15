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
//#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/numerics/error_estimator.h>

#include <random>
#include <ostream>

#include "user_input.h"
#include "make_grid.h"
#include "dsimstructs.h"
#include "mesh_struct.h"
#include "dirichlet_boundary.h"
#include "steady_state.h"
//#include "mix_mesh.h"
#include "nanoflann_structures.h"
#include "particle_tracking.h"
#include "streamlines.h"

//#include "helper_functions.h"


using namespace dealii;

/*!
 * \brief The NPSAT class is the main class that makes use of every other functionality of this programm
 */
template <int dim>
class NPSAT{
public:
    //! The contructor requires the user inputs structure. Besides initializing all default deall parameters,
    //! the construcotr generates the initial mesh and reads the dirichlet boundary conditions.
    NPSAT(AquiferProperties<dim>& AQP);

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

    void printVelocityField(MyTensorFunction<dim> &HK_function);

    void printGWrecharge(MyFunction<dim, dim> &RCH_function);



private:
    MPI_Comm                                  	mpi_communicator;
    parallel::distributed::Triangulation<dim> 	triangulation;
    DoFHandler<dim>                             dof_handler;
    FE_Q<dim>                                 	fe;
    AffineConstraints<double>                   Headconstraints;

    TrilinosWrappers::MPI::Vector               locally_relevant_solution;
    TrilinosWrappers::MPI::Vector               system_rhs;

    // Moving mesh data
    DoFHandler<dim>                             mesh_dof_handler;
    FESystem<dim>                              	mesh_fe;
    TrilinosWrappers::MPI::Vector               mesh_vertices;
    TrilinosWrappers::MPI::Vector               distributed_mesh_vertices;
    TrilinosWrappers::MPI::Vector               mesh_Offset_vertices;
    TrilinosWrappers::MPI::Vector               distributed_mesh_Offset_vertices;
    IndexSet                                    mesh_locally_owned;
    IndexSet                                    mesh_locally_relevant;
    AffineConstraints<double>                   mesh_constraints;
    //mix_mesh<dim-1>                             top_grid;
    //mix_mesh<dim-1>                             bottom_grid;
    PointVectorCloud                            topCloud;
    std::shared_ptr<pointVector_kd_tree>        topCloudIndex;
    PointVectorCloud                            botCloud;
    std::shared_ptr<pointVector_kd_tree>        botCloudIndex;
    std::vector<ScatterInterp<dim>>             newTopFnc;





    AquiferProperties<dim>                      AQProps;

    Mesh_struct<dim>                            mesh_struct;

    // Boundary Conditions
    std::map<types::boundary_id, const Function<dim>* >             dirichlet_boundary;
    BoundaryConditions::Dirichlet<dim>          DirBC;
    std::vector<int>                            top_boundary_ids;
    std::vector<int>                            bottom_boundary_ids;

    ConditionalOStream                        	pcout;
    int                                         my_rank;

    void make_grid();
    //void create_dim_1_grids();
    void create_top_bot_functions();
    void create_top_scatter_func();
    void flag_cells_for_refinement();
    void print_mesh();
    void save();
    void load();
    void read_Particles_from_file(std::vector<Streamline<dim> > &Streamlines);
    bool meshConform = false;

    void UpdateNonConformingElevation();
    bool useSolution2UpdateMesh = false;
};

template <int dim>
NPSAT<dim>::NPSAT(AquiferProperties<dim>& AQP)
    :
    mpi_communicator (MPI_COMM_WORLD),
    triangulation (mpi_communicator,
                    typename Triangulation<dim>::MeshSmoothing
                    (Triangulation<dim>::smoothing_on_refinement |
                    Triangulation<dim>::limit_level_difference_at_vertices)),
    dof_handler (triangulation),
    fe (1),
    mesh_dof_handler (triangulation),
    mesh_fe (FE_Q<dim>(1),1),
    AQProps(AQP),
    mesh_struct(AQP.xy_thres, AQP.z_thres),
    pcout(std::cout,(Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
{
    //user_input = CLI;
    my_rank = Utilities::MPI::this_mpi_process(mpi_communicator);
    pcout << "Simulation started at \n" << print_current_time() << std::endl;
    make_grid();
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

    // Load solution
    if (AQProps.solver_param.load_solution > 0){

        load();
        mesh_struct.move_vertices(mesh_dof_handler, mesh_vertices);

        if (AQProps.print_init_mesh >0){
            std::ofstream out (AQProps.Dirs.output + AQProps.sim_prefix + "_init_mesh_" + Utilities::int_to_string(my_rank,4) + ".vtk");
            GridOut grid_out;
            grid_out.write_ucd(triangulation, out);
        }
        return;
    }

    // Refine uniformly if it is requested
    if (AQProps.N_init_refinement > 0){
        triangulation.refine_global(AQProps.N_init_refinement);
    }

    if (!meshConform){
        UpdateNonConformingElevation();
    }


    if (meshConform){
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

        //MyFunction<dim, dim> top_function(AQProps.top_elevation);
        //MyFunction<dim, dim> bottom_function(AQProps.bottom_elevation);
        //double test = top_function.value(Point<dim>(550,0));
        //double tmp = AQProps.top_elevation.interpolate(Point<dim>(550,0));

        mesh_struct.compute_initial_elevations(AQProps.top_elevation, AQProps.bottom_elevation);

        mesh_struct.updateMeshElevation(mesh_dof_handler,
                                        triangulation,
                                        mesh_constraints,
                                        mesh_vertices,
                                        distributed_mesh_vertices,
                                        mesh_Offset_vertices,
                                        distributed_mesh_Offset_vertices,
                                        mpi_communicator,
                                        pcout);
    }




    unsigned int count_refinements = 0;
    while (true){
        pcout << "-------Refinement Iteration : " << count_refinements << std::endl;
        bool done_wells = false;
        bool done_streams = false;
        bool done_top = false;
        bool done_bnds = false;


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

        if (count_refinements >= AQProps.N_top_refinements){
            done_top = true;
        }
        if (count_refinements >= AQProps.N_Bnd_refinements){
            done_bnds = true;
        }
        if (!done_top || !done_bnds){
            refineTop<dim>(triangulation, !done_top, !done_bnds);
        }



        if (done_wells && done_streams && done_top && done_bnds)
            break;

        do_refinement1();

        if (!meshConform){
            UpdateNonConformingElevation();
        }
        else{
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
            mesh_struct.compute_initial_elevations(AQProps.top_elevation, AQProps.bottom_elevation);

            mesh_struct.updateMeshElevation(mesh_dof_handler,
                                            triangulation,
                                            mesh_constraints,
                                            mesh_vertices,
                                            distributed_mesh_vertices,
                                            mesh_Offset_vertices,
                                            distributed_mesh_Offset_vertices,
                                            mpi_communicator,
                                            pcout);
        }


        count_refinements++;
    }

    // Prepare dirichlet Boudnary Conditions
    DirBC.get_from_file(AQProps.dirichlet_file_names, AQProps.Dirs.input);

    if (AQProps.print_init_mesh >0){
        std::ofstream out (AQProps.Dirs.output + AQProps.sim_prefix + "_init_mesh_" + Utilities::int_to_string(my_rank,4) + ".vtk");
        GridOut grid_out;
        grid_out.write_ucd(triangulation, out);
    }
}

template<int dim>
void NPSAT<dim>::UpdateNonConformingElevation() {
    //{
    //    std::ofstream out("Pre_init_mesh.vtk");
    //    GridOut grid_out;
    //    grid_out.write_ucd(triangulation, out);
    //}

    const MappingQ1<dim> mapping;
    mesh_dof_handler.distribute_dofs(mesh_fe);
    pcout << "Distribute mesh dofs..." << mesh_dof_handler.n_dofs() << std::endl << std::flush;
    MPI_Barrier(mpi_communicator);
    mesh_locally_owned = mesh_dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs (mesh_dof_handler, mesh_locally_relevant);
    //std::cout << "IsHere 1" << std::endl;

    mesh_vertices.reinit (mesh_locally_owned, mesh_locally_relevant, mpi_communicator);
    distributed_mesh_vertices.reinit(mesh_locally_owned, mpi_communicator);
    mesh_Offset_vertices.reinit (mesh_locally_owned, mesh_locally_relevant, mpi_communicator);
    distributed_mesh_Offset_vertices.reinit(mesh_locally_owned, mpi_communicator);

    //std::cout << "IsHere 2" << std::endl;

    const std::vector<Point<dim> > mesh_support_points
            = mesh_fe.base_element(0).get_unit_support_points();

    FEValues<dim> fe_mesh_points (mapping,
                                  mesh_fe,
                                  mesh_support_points,
                                  update_quadrature_points);

    mesh_constraints.clear();
    mesh_constraints.reinit(mesh_locally_relevant);
    DoFTools::make_hanging_node_constraints(mesh_dof_handler, mesh_constraints);
    mesh_constraints.close();
    //std::cout << "IsHere 3" << std::endl;

    std::map<int, PZ<dim>> dofZnew;
    typename std::map<int, PZ<dim>>::iterator itpz;
    std::vector<unsigned int> cell_dof_indices (mesh_fe.dofs_per_cell);

    for (const auto &cell : mesh_dof_handler.active_cell_iterators()){
        if (cell->is_locally_owned()){
            bool print_this_cell = false;
            //if (Point<dim>(322356, 3.98326e+06, 100).distance(cell->center()) < 10) {
            //    print_this_cell = true;
            //    std::cout << my_rank << "----------------" << std::endl;
            //}
            fe_mesh_points.reinit(cell);
            cell->get_dof_indices (cell_dof_indices);
            for (unsigned int idof = 0; idof < mesh_fe.base_element(0).dofs_per_cell; ++idof){
                PZ<dim> pz;

                unsigned int support_point_index = mesh_fe.component_to_system_index(0, idof);
                int current_dof = static_cast<int>(cell_dof_indices[support_point_index]);
                itpz = dofZnew.find(current_dof);
                if (itpz != dofZnew.end()){
                    //std::cout << "DOF:" << itpz->first << ": " << itpz->second.p << " | " << itpz->second.Znew << std::endl;
                    continue;
                }

                pz.p = fe_mesh_points.quadrature_point(idof);
                double PointTop;

                if (useSolution2UpdateMesh){
                    PointTop = newTopFnc[0].interpolate(pz.p);
                }
                else{
                    PointTop = AQProps.top_elevation.interpolate(pz.p);
                }

                double PointBot = AQProps.bottom_elevation.interpolate(pz.p);
                double newElev = PointTop * pz.p[dim-1]/100.0 + PointBot*(1 - pz.p[dim-1]/100.0);
                double dz = newElev - pz.p[dim-1];

                distributed_mesh_Offset_vertices[cell_dof_indices[support_point_index]] = dz;
                distributed_mesh_vertices[cell_dof_indices[support_point_index]] = newElev;
                pz.Znew = newElev;
                dofZnew.insert(std::pair<int, PZ<dim>> (current_dof, pz) );
            }
        }
    }

    MPI_Barrier(mpi_communicator);
    // The compress sends the data to the processors that owns the data
//distributed_mesh_vertices.compress(VectorOperation::add);
//distributed_mesh_Offset_vertices.compress(VectorOperation::insert);
    //std::cout << "IsHere 4" << std::endl;

    //mesh_constraints.distribute(distributed_mesh_Offset_vertices);
    mesh_Offset_vertices = distributed_mesh_Offset_vertices;

    //mesh_constraints.distribute(distributed_mesh_vertices);
    mesh_vertices = distributed_mesh_vertices;
    //std::cout << "IsHere 5" << std::endl;

    for (const auto &cell : mesh_dof_handler.active_cell_iterators()){
        if (cell->is_locally_owned()){
            for (unsigned int vertex_no = 0; vertex_no < GeometryInfo<dim>::vertices_per_cell; ++vertex_no){
                Point<dim> &v=cell->vertex(vertex_no);
                for (unsigned int dir=0; dir < dim; ++dir){
                    v(dir) = mesh_vertices(cell->vertex_dof_index(vertex_no, dir));
                    //if (Point<dim-1>(322119.700000, 3983005.750000).distance(Point<dim-1>(v[0], v[1])) < 10){
                    //    std::cout << "Assigned: " << v(dir) << std::endl;
                    //}
                }
            }
        }
    }

    //std::cout << "IsHere 6" << std::endl;
    //{
    //    std::ofstream out("Post_init_mesh.vtk");
    //    GridOut grid_out;
    //    grid_out.write_ucd(triangulation, out);
    //}
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

    SimPrintFlags printing_flags;
    printing_flags.top_point_cloud = AQProps.print_point_top_cloud;
    printing_flags.print_vtk = AQProps.print_solution_vtk;
    printing_flags.print_velocity = AQProps.print_velocity_cloud;
    printing_flags.print_BND = AQProps.print_bnd_cond;
    useSolution2UpdateMesh = true;



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
                       system_rhs,
                       dirichlet_boundary,
                       HK_function[0],
                       GR_funct,
                       AQProps.NeumanBoundaries,
                       top_boundary_ids,
                        AQProps.solver_param);

        gw.Simulate(iter,
                    AQProps.Dirs.output + AQProps.sim_prefix,
                    triangulation, AQProps.wells, AQProps.streams, printing_flags);

        //gw.Simulate_refine(iter,
        //                AQProps.Dirs.output + AQProps.sim_prefix,
        //                triangulation, AQProps.wells /*,
        //                SourceSinks::Streams&                      streams*/,
        //                AQProps.refine_param.TopFraction, AQProps.refine_param.BottomFraction);


        if (iter < AQProps.solver_param.NonLinearIter - 1){
            pcout << "      Updating Mesh ..." <<std::endl;
            newTopFnc.clear();
            newTopFnc.resize(1);
            if (!meshConform) {
                create_top_scatter_func();
            }
            else
                create_top_bot_functions();

            //create_dim_1_grids();
            if (iter < AQProps.refine_param.MaxRefinement)
                flag_cells_for_refinement();
            do_refinement1();

            if (!meshConform){
                UpdateNonConformingElevation();
            }
            else{
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
                MPI_Barrier(mpi_communicator);
                //mesh_struct.assign_top_bottom(top_grid, bottom_grid, pcout, mpi_communicator);
                mesh_struct.assign_top_bottom(topCloud, topCloudIndex,
                                              AQProps.top_cloud_param.Power,
                                              AQProps.top_cloud_param.Radius,
                                              botCloud, botCloudIndex,
                                              AQProps.bot_cloud_param.Power,
                                              AQProps.bot_cloud_param.Radius,
                                              AQProps.top_cloud_param.Threshold,
                                              pcout, mpi_communicator);
                MPI_Barrier(mpi_communicator);
                mesh_struct.updateMeshElevation(mesh_dof_handler,
                                                triangulation,
                                                mesh_constraints,
                                                mesh_vertices,
                                                distributed_mesh_vertices,
                                                mesh_Offset_vertices,
                                                distributed_mesh_Offset_vertices,
                                                mpi_communicator,
                                                pcout);
            }

            MPI_Barrier(mpi_communicator);
            //print_mesh();

        }
    }

    if (AQProps.solver_param.save_solution > 0)
        save();
    if (AQProps.print_velocity_cloud > 0)
        printVelocityField(HK_function[0]);

    if (AQProps.print_GW_rch > 0)
        printGWrecharge(GR_funct);


    pcout << "Simulation ended at \n" << print_current_time() << std::endl;

}

/*
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
*/

template<int dim>
void NPSAT<dim>::create_top_scatter_func(){
    unsigned int my_rank = Utilities::MPI::this_mpi_process(mpi_communicator);
    unsigned int n_proc = Utilities::MPI::n_mpi_processes(mpi_communicator);
    std::map<int,Point<dim> > topPoints;
    typename std::map<int,Point<dim> >::iterator itTop;
    std::vector<std::vector<int>> Elements;

    QTrapez<dim-1> face_trapez_formula;
    FEFaceValues<dim> fe_face_values(fe, face_trapez_formula, update_values);
    std::vector< double > values(face_trapez_formula.size());
    std::vector<unsigned int> face_dof_indices(fe.dofs_per_face);

    unsigned int iTopface = GeometryInfo<dim>::faces_per_cell-1;

    for(const auto &cell : dof_handler.active_cell_iterators()){
        if (cell->is_locally_owned()){
            if(cell->face(iTopface)->at_boundary()){
                fe_face_values.reinit (cell, iTopface);
                cell->face(iTopface)->get_dof_indices(face_dof_indices);
                fe_face_values.get_function_values(locally_relevant_solution, values);
                for (unsigned int ii = 0; ii < face_trapez_formula.size(); ++ii){
                    itTop = topPoints.find(face_dof_indices[ii]);
                    if (itTop == topPoints.end()){
                        Point<dim> temp_point_dim = cell->face(iTopface)->vertex(ii);
                        temp_point_dim[dim-1] = values[ii];
                        //std::cout << temp_point_dim << std::endl;
                        topPoints.insert(std::pair<int, Point<dim> >(face_dof_indices[ii], temp_point_dim));
                    }
                }
                if (dim == 2){
                    std::vector<int>tmp;
                    tmp.push_back(face_dof_indices[0]);
                    tmp.push_back(face_dof_indices[1]);
                    Elements.push_back(tmp);
                }
                else if (dim == 3){
                    std::vector<int>tmp;
                    tmp.push_back(face_dof_indices[0]);
                    tmp.push_back(face_dof_indices[1]);
                    tmp.push_back(face_dof_indices[2]);
                    Elements.push_back(tmp);
                    std::vector<int>tmp1;
                    tmp1.push_back(face_dof_indices[1]);
                    tmp1.push_back(face_dof_indices[3]);
                    tmp1.push_back(face_dof_indices[2]);
                    Elements.push_back(tmp1);
                }
            }
        }
    }
    std::cout << "Rank: " << my_rank << " has: " << topPoints.size()
              << " top and: " << Elements.size() << " bottom" << std::endl;

    if (n_proc > 1){
        MPI_Barrier(mpi_communicator);
        std::vector<int> Number_of_Points;
        std::vector<int> Number_of_Elements;
        std::vector<std::vector<int>> doftop(n_proc);
        std::vector<std::vector<double>> xpnt(n_proc);
        std::vector<std::vector<double>> ypnt(n_proc);
        std::vector<std::vector<double>> zpnt(n_proc);
        std::vector<std::vector<int>> iA(n_proc);
        std::vector<std::vector<int>> iB(n_proc);
        std::vector<std::vector<int>> iC(n_proc);

        for (itTop = topPoints.begin(); itTop != topPoints.end(); ++ itTop){
            doftop[my_rank].push_back(itTop->first);
            xpnt[my_rank].push_back(itTop->second[0]);
            if (dim == 2){
                zpnt[my_rank].push_back(itTop->second[1]);
            }
            else{
                ypnt[my_rank].push_back(itTop->second[1]);
                zpnt[my_rank].push_back(itTop->second[2]);
            }
        }

        for (unsigned int i = 0; i < Elements.size(); ++i){
            iA[my_rank].push_back(Elements[i][0]);
            iB[my_rank].push_back(Elements[i][1]);
            if (dim == 3)
                iC[my_rank].push_back(Elements[i][2]);
        }

        MPI_Barrier(mpi_communicator);
        // Send Receive points
        Send_receive_size(static_cast<unsigned int>(doftop[my_rank].size()), n_proc, Number_of_Points, mpi_communicator);
        Sent_receive_data<int>(doftop, Number_of_Points, my_rank, mpi_communicator, MPI_INT);
        Sent_receive_data<double>(xpnt, Number_of_Points, my_rank, mpi_communicator, MPI_DOUBLE);
        if (dim == 3)
            Sent_receive_data<double>(ypnt, Number_of_Points, my_rank, mpi_communicator, MPI_DOUBLE);
        Sent_receive_data<double>(zpnt, Number_of_Points, my_rank, mpi_communicator, MPI_DOUBLE);

        // Send receive the elements
        Send_receive_size(static_cast<unsigned int>(iA[my_rank].size()), n_proc, Number_of_Elements, mpi_communicator);
        Sent_receive_data<int>(iA, Number_of_Elements, my_rank, mpi_communicator, MPI_INT);
        Sent_receive_data<int>(iB, Number_of_Elements, my_rank, mpi_communicator, MPI_INT);
        if (dim == 3)
            Sent_receive_data<int>(iC, Number_of_Elements, my_rank, mpi_communicator, MPI_INT);

        // Each processors will loop through the data sent from the other processors and get the ones
        // that is missing
        for (unsigned int i_proc = 0; i_proc < n_proc; ++i_proc){
            if (i_proc == my_rank)
                continue;
            // Get the points
            for (unsigned int i = 0; i < doftop[i_proc].size(); ++i){
                itTop = topPoints.find(doftop[i_proc][i]);
                if (itTop == topPoints.end()){
                    Point<dim> temp_point_dim;
                    temp_point_dim[0] = xpnt[i_proc][i];
                    if (dim == 2){
                        temp_point_dim[1] = zpnt[i_proc][i];
                    }
                    else if (dim == 3){
                        temp_point_dim[1] = ypnt[i_proc][i];
                        temp_point_dim[2] = zpnt[i_proc][i];
                    }
                    //std::cout << "sent " << temp_point_dim << std::endl;
                    topPoints.insert(std::pair<int, Point<dim> >(doftop[i_proc][i], temp_point_dim));
                }
            }

            // Get the elements
            for (unsigned int i = 0; i < iA[i_proc].size(); ++i){
                std::vector<int> elem;
                elem.push_back(iA[i_proc][i]);
                elem.push_back(iB[i_proc][i]);
                if (dim == 3)
                    elem.push_back(iC[i_proc][i]);
                Elements.push_back(elem);
            }
        }
    }

    std::cout << "Rank: " << my_rank << " has: " << topPoints.size()
              << " top and: " << Elements.size() << " bottom" << std::endl;

    newTopFnc[0].SetScalarData(topPoints,Elements,SCI_METHOD::LINEAR);
}

template<int dim>
void NPSAT<dim>::create_top_bot_functions(){
    unsigned int my_rank = Utilities::MPI::this_mpi_process(mpi_communicator);
    unsigned int n_proc = Utilities::MPI::n_mpi_processes(mpi_communicator);

    std::map<int,std::pair<Point<dim>, double> > topPoints;
    std::map<int,Point<dim> > botPoints;
    typename std::map<int,std::pair<Point<dim>, double> >::iterator itTop;
    typename std::map<int,Point<dim> >::iterator itBot;

    QTrapez<dim-1> face_trapez_formula;
    FEFaceValues<dim> fe_face_values(fe, face_trapez_formula, update_values);
    std::vector< double > values(face_trapez_formula.size());
    std::vector<unsigned int> face_dof_indices(fe.dofs_per_face);
    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell){
        if (cell->is_locally_owned()){
            unsigned int iface = GeometryInfo<dim>::faces_per_cell-1;
            if(cell->face(iface)->at_boundary()){
                // This is top face
                fe_face_values.reinit (cell, iface);
                cell->face(iface)->get_dof_indices(face_dof_indices);
                fe_face_values.get_function_values(locally_relevant_solution, values);
                for (unsigned int ii = 0; ii < face_trapez_formula.size(); ++ii){
                    itTop = topPoints.find(face_dof_indices[ii]);
                    if (itTop == topPoints.end()){
                        Point<dim> temp_point_dim = cell->face(iface)->vertex(ii);
                        topPoints.insert(std::pair<int, std::pair<Point<dim>, double> >
                                                 (face_dof_indices[ii],
                                                 std::pair<Point<dim>, double>(temp_point_dim, values[ii])));
                    }
                }
            }
            iface = GeometryInfo<dim>::faces_per_cell-2;
            if(cell->face(iface)->at_boundary()){
                // This is bottom face
                cell->face(iface)->get_dof_indices(face_dof_indices);
                fe_face_values.reinit (cell, iface);
                fe_face_values.get_function_values(locally_relevant_solution, values);
                for (unsigned int ii = 0; ii < face_trapez_formula.size(); ++ii){
                    itBot = botPoints.find(face_dof_indices[ii]);
                    if (itBot == botPoints.end()){
                        Point<dim> temp_point_dim = cell->face(iface)->vertex(ii);
                        botPoints.insert(std::pair<int, Point<dim>>(face_dof_indices[ii], temp_point_dim));
                    }
                }
            }
        }
    }
    std::cout << "Rank: " << my_rank << " has: " << topPoints.size() << " top and: " << botPoints.size() << " bottom" << std::endl;

    if (n_proc > 1){
        MPI_Barrier(mpi_communicator);
        // Next we have to send the points to every other processor
        std::vector<int> top_Number_Points;
        std::vector<std::vector<int>> doftop(n_proc);
        std::vector<std::vector<double>> xtop(n_proc);
        std::vector<std::vector<double>> ytop(n_proc);
        std::vector<std::vector<double>> zold(n_proc);
        std::vector<std::vector<double>> znew(n_proc);
        for (itTop = topPoints.begin(); itTop != topPoints.end(); ++itTop){
            doftop[my_rank].push_back(itTop->first);
            xtop[my_rank].push_back(itTop->second.first[0]);
            if (dim == 3){
                ytop[my_rank].push_back(itTop->second.first[1]);
                zold[my_rank].push_back(itTop->second.first[2]);
            }
            else if (dim == 2){
                zold[my_rank].push_back(itTop->second.first[1]);
            }
            znew[my_rank].push_back(itTop->second.second);
        }

        std::vector<int> bot_Number_Points;
        std::vector<std::vector<int>> dofbot(n_proc);
        std::vector<std::vector<double>> xbot(n_proc);
        std::vector<std::vector<double>> ybot(n_proc);
        std::vector<std::vector<double>> zbot(n_proc);
        for(itBot = botPoints.begin(); itBot != botPoints.end(); ++itBot){
            dofbot[my_rank].push_back(itBot->first);
            xbot[my_rank].push_back(itBot->second[0]);
            if (dim == 3){
                ybot[my_rank].push_back(itBot->second[1]);
                zbot[my_rank].push_back(itBot->second[2]);
            }
            else if (dim == 2){
                zbot[my_rank].push_back(itBot->second[1]);
            }
        }
        MPI_Barrier(mpi_communicator);

        // Send receive data for the top
        Send_receive_size(static_cast<unsigned int>(doftop[my_rank].size()), n_proc, top_Number_Points, mpi_communicator);
        Sent_receive_data<int>(doftop, top_Number_Points, my_rank, mpi_communicator, MPI_INT);
        Sent_receive_data<double>(xtop, top_Number_Points, my_rank, mpi_communicator, MPI_DOUBLE);
        if (dim == 3)
            Sent_receive_data<double>(ytop, top_Number_Points, my_rank, mpi_communicator, MPI_DOUBLE);
        Sent_receive_data<double>(zold, top_Number_Points, my_rank, mpi_communicator, MPI_DOUBLE);
        Sent_receive_data<double>(znew, top_Number_Points, my_rank, mpi_communicator, MPI_DOUBLE);

        // Send receive data for the bot
        Send_receive_size(static_cast<unsigned int>(dofbot[my_rank].size()), n_proc, bot_Number_Points, mpi_communicator);
        Sent_receive_data<int>(dofbot, bot_Number_Points, my_rank, mpi_communicator, MPI_INT);
        Sent_receive_data<double>(xbot, bot_Number_Points, my_rank, mpi_communicator, MPI_DOUBLE);
        if (dim == 3)
            Sent_receive_data<double>(ybot, bot_Number_Points, my_rank, mpi_communicator, MPI_DOUBLE);
        Sent_receive_data<double>(zbot, bot_Number_Points, my_rank, mpi_communicator, MPI_DOUBLE);

        // Each processors will loop through the data sent from the other processors and get the ones
        // that is missing
        for (unsigned int i_proc = 0; i_proc < n_proc; ++i_proc){
            if (i_proc == my_rank)
                continue;
            for (unsigned int i = 0; i < doftop[i_proc].size(); ++i){
                itTop = topPoints.find(doftop[i_proc][i]);
                if (itTop == topPoints.end()){
                    Point<dim> temp_point_dim;
                    temp_point_dim[0] = xtop[i_proc][i];
                    if (dim == 3){
                        temp_point_dim[1] = ytop[i_proc][i];
                        temp_point_dim[2] = zold[i_proc][i];
                    }
                    else if (dim == 2){
                        temp_point_dim[1] = zold[i_proc][i];
                    }
                    topPoints.insert(std::pair<int, std::pair<Point<dim>, double> >
                                             (doftop[i_proc][i],
                                              std::pair<Point<dim>, double>(
                                                      temp_point_dim, znew[i_proc][i])));
                }
            }

            for (unsigned int i = 0; i < dofbot[i_proc].size(); ++i){
                itBot = botPoints.find(dofbot[i_proc][i]);
                if (itBot == botPoints.end()){
                    Point<dim> temp_point_dim;
                    temp_point_dim[0] = xbot[i_proc][i];
                    if (dim == 3){
                        temp_point_dim[1] = ybot[i_proc][i];
                        temp_point_dim[2] = zbot[i_proc][i];
                    }
                    else if (dim == 2){
                        temp_point_dim[1] = zbot[i_proc][i];
                    }
                    botPoints.insert(std::pair<int, Point<dim>>(dofbot[i_proc][i], temp_point_dim));
                }
            }
        }
    }
    MPI_Barrier(mpi_communicator);
    std::cout << "Rank: " << my_rank << " has: " << topPoints.size() << " top and: " << botPoints.size() << " bottom" << std::endl;
    // ALl processors must have a complete set of top and bottom points to create
    // the top and bottom interpolation function
    topCloud.pts.clear();
    botCloud.pts.clear();
    topCloudIndex.reset(new pointVector_kd_tree(
            2, topCloud,
            nanoflann::KDTreeSingleIndexAdaptorParams(10)) );
    botCloudIndex.reset(new pointVector_kd_tree(
            2, botCloud,
            nanoflann::KDTreeSingleIndexAdaptorParams(10)) );


    for (itTop = topPoints.begin(); itTop != topPoints.end(); ++itTop){
        PointVector pid;
        pid.x = itTop->second.first[0];
        if (dim == 3){
            pid.y = itTop->second.first[1];
            pid.values.push_back(itTop->second.first[2]);
        }
        else if (dim == 2){
            pid.y = 0;
            pid.values.push_back(itTop->second.first[1]);
        }
        pid.values.push_back(itTop->second.second);
        topCloud.pts.push_back(pid);
    }
    topCloudIndex->buildIndex();

    for (itBot = botPoints.begin(); itBot != botPoints.end(); ++itBot){
        PointVector pid;
        pid.x = itBot->second[0];
        if (dim == 3){
            pid.y = itBot->second[1];
            pid.values.push_back(itBot->second[2]);
        }
        else if (dim == 2){
            pid.y = 0;
            pid.values.push_back(itBot->second[1]);
        }
        botCloud.pts.push_back(pid);
    }
    botCloudIndex->buildIndex();
    std::cout << "Rank " << my_rank << "done building trees" << std::endl;
}


template <int dim>
void NPSAT<dim>::flag_cells_for_refinement(){
    pcout << "Flag cells for refinement" << std::endl << std::flush;
    MPI_Barrier(mpi_communicator);
    Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate(dof_handler,
                                     QGauss<dim-1>(fe.degree+2),
                                     {},
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
    pcout << "Executing refinement... " << std::endl << std::flush;
    MPI_Barrier(mpi_communicator);
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
    //    std::ofstream out ("RefA" + std::to_string(my_rank) + ".vtk");
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
    //    std::ofstream out ("RefB" + std::to_string(my_rank) + ".vtk");
    //    GridOut grid_out;
    //    grid_out.write_ucd(triangulation, out);
    //}
    triangulation.communicate_locally_moved_vertices(locally_owned_vertices);


    // now the mesh should consistent as when it was first created
    // so we can hopefully refine it
    triangulation.execute_coarsening_and_refinement();
    //{
    //    std::ofstream out ("RefC" + std::to_string(my_rank) + ".vtk");
    //    GridOut grid_out;
    //    grid_out.write_ucd(triangulation, out);
    //}
}

template <int dim>
void NPSAT<dim>::particle_tracking(){
    pcout << "PARTICLE TRACKING..." << std::endl;
    pcout << "Started at \n" << print_current_time() << std::endl;
    unsigned int n_proc = Utilities::MPI::n_mpi_processes(mpi_communicator);

    MyFunction<dim, dim> porosity_fnc(AQProps.Porosity);

    Particle_Tracking<dim> pt(mpi_communicator,
                         dof_handler, fe,
                         Headconstraints,
                         locally_relevant_solution,
                         system_rhs,
                         AQProps.HK_function[0],
                         porosity_fnc,
                         AQProps.part_param);

    //pt.average_velocity_field(velocity_dof_handler,velocity_fe);
    //pt.calculate_RT0_velocity_field();
    pt.average_velocity_field1(AQProps.print_Average_Velocity);

    std::vector<Streamline<dim>> All_streamlines;
    std::vector<std::vector<Streamline<dim>>> part_of_streamlines(n_proc);

    MPI_Barrier(mpi_communicator);
    if (my_rank == 0){
        if (AQProps.part_param.trace_wells > 0)
            AQProps.wells.distribute_particles(All_streamlines,
                                               AQProps.part_param.Wells_N_per_layer,
                                               AQProps.part_param.Wells_N_Layers,
                                               AQProps.part_param.radius,
                                               AQProps.part_param.PartDistribType);

        read_Particles_from_file(All_streamlines);
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
            if (All_streamlines.size() < N + 1000){ // This should be 1000 an maybe over, but set it to 10 for debug
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
            pcout << "      There are " << All_streamlines.size()  << " particles to trace" << std::endl;
        }

        MPI_Barrier(mpi_communicator);
        Sent_receive_streamlines_all_to_all(part_of_streamlines, my_rank, n_proc, mpi_communicator);
        //std::cout << "I'm proc " << my_rank << " and have " << part_of_streamlines[my_rank].size() << " to trace" << std::endl;
        MPI_Barrier(mpi_communicator);

        pt.trace_particles(part_of_streamlines[my_rank], particle_iter++);

        // Processor 0 which is responsible to send out the streamlines will
        // broadcast if there are more streamlines to trace
        MPI_Bcast( part_done, 1, MPI_INT, 0, mpi_communicator);
        if (part_done[0] == 1)
            break;
    }
    pcout << "Particle tracking ended at \n" << print_current_time() << std::endl;
    pcout << "To gather the streamlines use the following command: \n"
          << "npsat -p " << AQProps.main_param_file
          << " -g " << n_proc << " " << particle_iter << " -e # of expected entities" << std::endl;

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
void NPSAT<dim>::save(){
    pcout << "Saving snapshot..." << std::endl << std::flush;
    // preparing the solution
    std::vector<const TrilinosWrappers::MPI::Vector *> x_system(1);
    x_system[0] = &locally_relevant_solution;
    parallel::distributed::SolutionTransfer<dim, TrilinosWrappers::MPI::Vector> sol_trans(dof_handler);
    sol_trans.prepare_for_serialization (x_system);

    // preparing the mesh vertices
    std::vector<const TrilinosWrappers::MPI::Vector *> x_fs_system (1);
    x_fs_system[0] = &mesh_vertices;
    parallel::distributed::SolutionTransfer<dim, TrilinosWrappers::MPI::Vector> msh_trans(mesh_dof_handler);
    msh_trans.prepare_for_serialization(x_fs_system);

    const std::string filename = (AQProps.Dirs.output + AQProps.sim_prefix + AQProps.solution_suffix + ".npsat");
    triangulation.save(filename.c_str());
}

template <int dim>
void NPSAT<dim>::load(){
    const std::string filename = (AQProps.Dirs.output + AQProps.sim_prefix + AQProps.solution_suffix + ".npsat");
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

        Headconstraints.clear();
        Headconstraints.reinit(locally_relevant_dofs);
        DoFTools::make_hanging_node_constraints (dof_handler, Headconstraints);
        Headconstraints.close();


        mesh_dof_handler.distribute_dofs(mesh_fe);
        mesh_locally_owned = mesh_dof_handler.locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs (mesh_dof_handler, mesh_locally_relevant);
        mesh_vertices.reinit (mesh_locally_owned,
                              mesh_locally_relevant, mpi_communicator);
        TrilinosWrappers::MPI::Vector distributed_mesh_system(mesh_locally_owned, mpi_communicator);
        std::vector<TrilinosWrappers::MPI::Vector *> fs_system(1);

        parallel::distributed::SolutionTransfer<dim, TrilinosWrappers::MPI::Vector> msh_trans(mesh_dof_handler);
        fs_system[0] = & (distributed_mesh_system);
        msh_trans.deserialize(fs_system);
        mesh_vertices = distributed_mesh_system;
    }
}

template <int dim>
void NPSAT<dim>::read_Particles_from_file(std::vector<Streamline<dim>> &Streamlines){
    if (AQProps.part_param.Particles_in_file.empty())
        return;
    const std::string filename = (AQProps.Dirs.input + AQProps.part_param.Particles_in_file);
    std::ifstream  datafile(filename.c_str());
    if (!datafile.good()){
        pcout << "Can't load the particle file " << filename << std::endl;
        return;
    }
    else{
        char buffer[512];
        unsigned int Nparticles, IDE, IDS;
        double Xcoord, Ycoord, Zcoord;
        datafile.getline(buffer,512);
        std::istringstream inp1(buffer);
        inp1 >> Nparticles;

        for (unsigned int i = 0; i < Nparticles; i++){
            datafile.getline(buffer,512);
            std::istringstream inp(buffer);
            Point<dim> p;
            inp >> IDE;
            inp >> IDS;
            inp >> Xcoord;
            p[0] = Xcoord;
            inp >> Ycoord;
            p[1] = Ycoord;
            if (dim == 3){
                inp >> Zcoord;
                p[2] = Zcoord;
            }
            Streamlines.push_back(Streamline<dim>(IDE, IDS, p));
        }
    }
}


template <int dim>
void NPSAT<dim>::printVelocityField(MyTensorFunction<dim>& HK_function){
    const std::string vel_filename = (AQProps.Dirs.output + AQProps.sim_prefix + "_" +
                                      Utilities::int_to_string(my_rank,4) + ".vel");

    pcout << "Printing velocity field in: " <<  (AQProps.Dirs.output + AQProps.sim_prefix + "xxxx.vel") << std::endl;

    std::ofstream vel_stream_file;
    vel_stream_file.open(vel_filename.c_str());

    int n_quad_points = 1;

    const QGauss<dim> quadrature_formula1(1);
    const QGauss<dim> quadrature_formula2(2);
    const QGauss<dim-1> face_quadrature_formula(2);
    const unsigned int   n_q_points1 = quadrature_formula1.size();
    const unsigned int   n_q_points2 = quadrature_formula2.size();
    const unsigned int   n_face_q_points = face_quadrature_formula.size();
    FEValues<dim> fe_values1 (fe, quadrature_formula1,
                             update_values    |  update_gradients |
                             update_quadrature_points |
                             update_JxW_values);
    FEValues<dim> fe_values2 (fe, quadrature_formula2,
                              update_values    |  update_gradients |
                              update_quadrature_points |
                              update_JxW_values);
    FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                      update_values  | update_gradients |
                                      update_quadrature_points |
                                      update_JxW_values);

    std::vector<Tensor<2,dim> > hydraulic_conductivity_values1(n_q_points1);
    std::vector<Tensor<1, dim> > hgrad1(n_q_points1);
    std::vector<Tensor<2,dim> > hydraulic_conductivity_values2(n_q_points2);
    std::vector<Tensor<1, dim> > hgrad2(n_q_points2);
    std::vector<Tensor<2,dim> > hydraulic_conductivity_values_face(n_face_q_points);
    std::vector<Tensor<1, dim> > hgrad_face(n_face_q_points);
    Tensor<1,dim> KdH;

    //std::vector<Point<dim>> cell_vertices(GeometryInfo<dim>::vertices_per_cell);
    //std::vector<Point<dim>> face_vertices(GeometryInfo<dim>::vertices_per_face);

    unsigned int topfaceid = GeometryInfo<dim>::faces_per_cell - 1;
    double m = AQProps.multiplier_velocity_print;

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    double shape_value;
    for (; cell!=endc; ++cell){
        if (cell->is_locally_owned()){
            double diameter = cell->diameter();
            BoundingBox<dim> BB = cell->bounding_box();
            // This pair containts the points of the Bounding Box. the first point of the pair is the lower corner
            // and the second point is the upper corner of the box
            std::pair<Point<dim>, Point<dim>> bb_points = BB.get_boundary_points();
            double cell_ratio = 0;
            if (dim == 3){
                double dx_cell = bb_points.second[0] - bb_points.first[0];
                double dy_cell = bb_points.second[1] - bb_points.first[1];
                double dz_cell = bb_points.second[2] - bb_points.first[2];
                cell_ratio = std::max(dx_cell, dy_cell)/dz_cell;
            }
            else if (dim == 2){
                double dx_cell = bb_points.second[0] - bb_points.first[0];
                double dy_cell = bb_points.second[1] - bb_points.first[1];
                cell_ratio = dx_cell/dy_cell;
            }

            bool is_cell_top = false;
            if (cell->face(topfaceid)->at_boundary())
                is_cell_top = true;
            else{
                if (!cell->neighbor(topfaceid)->has_children()){
                    if (cell->neighbor(topfaceid)->face(topfaceid)->at_boundary())
                        is_cell_top = true;
                }
            }


            if (is_cell_top){
                // For the top layer use higher order values
                fe_values2.reinit (cell);
                fe_values2.get_function_gradients(locally_relevant_solution, hgrad2);
                std::vector<Point<dim>> quad_points = fe_values2.get_quadrature_points();
                HK_function.value_list(quad_points, hydraulic_conductivity_values2);
                //for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell; ++i){
                //    cell_vertices[i] = cell->vertex(i);
                //}
                for (unsigned int q_point=0; q_point < n_q_points2; ++q_point){
                    KdH = hydraulic_conductivity_values2[q_point]*hgrad2[q_point];
                    //Point<dim> p;
                    //for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell; ++i){
                    //    shape_value = fe_values2.shape_value(i,q_point);
                    //    for (unsigned int idim = 0; idim < dim; ++idim){
                    //        p[idim] = p[idim] + shape_value*cell_vertices[i](idim);
                    //    }
                    //}
                    if (dim == 2){
                        vel_stream_file << std::setprecision(2) << std::fixed
                                        << quad_points[q_point][0] << " " << quad_points[q_point][1] << " "
                                        << std::setprecision(6) << std::fixed
                                        << -m*KdH[0] << " " << -m*KdH[1] << " ";
                    }
                    else if (dim == 3){
                        vel_stream_file << std::setprecision(2) << std::fixed
                                        << quad_points[q_point][0] << " " << quad_points[q_point][1] << " " << quad_points[q_point][2] << " "
                                        << std::setprecision(6) << std::fixed
                                        << -m*KdH[0] << " " << -m*KdH[1] << " " << -m*KdH[2] << " ";
                    }
                    vel_stream_file << cell->subdomain_id() << " " << std::setprecision(1) << std::fixed << diameter << " " << cell_ratio << std::endl;
                }

                // Print the velocity on the top face also if its on the top boundary
                if (cell->face(topfaceid)->at_boundary()){
                    fe_face_values.reinit (cell, topfaceid);
                    fe_face_values.get_function_gradients(locally_relevant_solution, hgrad_face);
                    std::vector<Point<dim>> quad_points = fe_face_values.get_quadrature_points();
                    HK_function.value_list(quad_points, hydraulic_conductivity_values_face);
                    for (unsigned int q_point=0; q_point < n_face_q_points; ++q_point) {
                        KdH = hydraulic_conductivity_values_face[q_point] * hgrad_face[q_point];
                        if (dim == 2){
                            vel_stream_file << std::setprecision(2) << std::fixed
                                            << quad_points[q_point][0] << " " << quad_points[q_point][1] << " "
                                            << std::setprecision(6) << std::fixed
                                            << -m*KdH[0] << " " << -m*KdH[1] << " ";
                        }
                        else if (dim == 3){
                            vel_stream_file << std::setprecision(2) << std::fixed
                                            << quad_points[q_point][0] << " " << quad_points[q_point][1] << " " << quad_points[q_point][2] << " "
                                            << std::setprecision(6) << std::fixed
                                            << -m*KdH[0] << " " << -m*KdH[1] << " " << -m*KdH[2] << " ";
                        }
                        vel_stream_file << cell->subdomain_id() << " " << std::setprecision(1) << std::fixed << diameter << " " << cell_ratio << std::endl;
                    }
                }
            }
            else{
                fe_values1.reinit (cell);
                fe_values1.get_function_gradients(locally_relevant_solution, hgrad1);
                std::vector<Point<dim>> quad_points = fe_values1.get_quadrature_points();
                HK_function.value_list(quad_points, hydraulic_conductivity_values1);
                //for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell; ++i){
                //    cell_vertices[i] = cell->vertex(i);
                //}
                for (unsigned int q_point=0; q_point < n_q_points1; ++q_point){
                    KdH = hydraulic_conductivity_values1[q_point]*hgrad1[q_point];
                    //Point<dim> p;
                    //for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell; ++i){
                    //    shape_value = fe_values1.shape_value(i,q_point);
                    //    for (unsigned int idim = 0; idim < dim; ++idim){
                    //        p[idim] = p[idim] + shape_value*cell_vertices[i](idim);
                    //    }
                    //}

                    if (dim == 2){
                        vel_stream_file << std::setprecision(2) << std::fixed
                                        << quad_points[q_point][0] << " " << quad_points[q_point][1] << " "
                                        << std::setprecision(6) << std::fixed
                                        << -m*KdH[0] << " " << -m*KdH[1] << " ";
                    }
                    else if (dim == 3){
                        vel_stream_file << std::setprecision(2) << std::fixed
                                        << quad_points[q_point][0] << " " << quad_points[q_point][1] << " " << quad_points[q_point][2] << " "
                                        << std::setprecision(6) << std::fixed
                                        << -m*KdH[0] << " " << -m*KdH[1] << " " << -m*KdH[2] << " ";
                    }
                    vel_stream_file << cell->subdomain_id() << " " << std::setprecision(1) << std::fixed << diameter << " " << cell_ratio << std::endl;
                }
            }
        }
    }
    vel_stream_file.close();
}

template <int dim>
void NPSAT<dim>::printGWrecharge(MyFunction<dim, dim> &RCH_function) {
    if (dim != 3 ){
        pcout << "Printing recharge in other than 3D is not supported yet" << std::endl;
        return;
    }
    const std::string rch_filename = (AQProps.Dirs.output + AQProps.sim_prefix + "_" +
                                      Utilities::int_to_string(my_rank,4) + ".rch");
    pcout << "Printing Groundwater recharge in: " <<  (AQProps.Dirs.output + AQProps.sim_prefix + "xxxx.vel") << std::endl;
    std::ofstream rch_stream;
    rch_stream.open(rch_filename);

    const QGauss<dim-1> face_quadrature_formula(2);
    const unsigned int n_face_q_points = face_quadrature_formula.size();
    FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                      update_values         | update_quadrature_points  | update_jacobians |
                                      update_normal_vectors | update_JxW_values);
    std::vector<double>			 		recharge_values(n_face_q_points);
    std::vector<Point<dim>> quad_points;

    unsigned int topfaceid = GeometryInfo<dim>::faces_per_cell-1;

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell){
        if (cell->is_locally_owned()){
            if(cell->face(topfaceid)->at_boundary()){
                fe_face_values.reinit (cell, topfaceid);
                quad_points = fe_face_values.get_quadrature_points();
                RCH_function.value_list(quad_points, recharge_values);
                for (unsigned int i = 0; i < recharge_values.size(); ++i){
                    rch_stream << quad_points[i] << " " << recharge_values[i]*1000 << std::endl;
                }
            }
        }
    }
    rch_stream.close();
}

#endif // NPSAT_H
