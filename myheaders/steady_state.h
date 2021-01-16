#ifndef STEADY_STATE_H
#define STEADY_STATE_H

#include <math.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/solver.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_solver.h>
//#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/dofs/function_map.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/lac/affine_constraints.h>

#include "my_functions.h"
#include "helper_functions.h"
#include "wells.h"
#include "streams.h"
#include "cgal_functions.h"
#include "dsimstructs.h"
#include "dirichlet_boundary.h"


struct SimPrintFlags{
public:
    SimPrintFlags()
        :
        top_point_cloud(0),
        top_mesh(0),
        bot_mesh(0),
        boundary_mesh(0),
        print_vtk(0),
        print_velocity(0),
        print_BND(0)
    {}
    int top_point_cloud;
    int top_mesh;
    int bot_mesh;
    int boundary_mesh;
    int print_vtk;
    int print_velocity;
    int print_BND;
};

using namespace dealii;

template<int dim>
class GWFLOW{
public:
    GWFLOW(MPI_Comm&                            mpi_communicator_in,
           DoFHandler<dim>&                     dof_handler,
           const FE_Q<dim>&                     fe,
           AffineConstraints<double>&           constraints,
           TrilinosWrappers::MPI::Vector&       locally_relevant_solution,
           TrilinosWrappers::MPI::Vector&       system_rhs,
           std::map<types::boundary_id, const Function<dim>* >&     dirichlet_boundary,
           MyTensorFunction<dim>&               HK_function,
           MyFunction<dim,dim>&                 groundwater_recharge,
           BoundaryConditions::Neumann<dim>&    Neumann_conditions,
           std::vector<int>&                    top_boundary_ids,
           SolverParameters&                    solver_param_in);



    void Simulate(int iter,                                     std::string output_file,
                  parallel::distributed::Triangulation<dim>& 	triangulation,
                  Well_Set<dim>&                                wells,
                  Streams<dim>&                                 streams,
                  SimPrintFlags                                 printflags);

    //void Simulate_refine(int iter,                                     std::string output_file,
    //                     parallel::distributed::Triangulation<dim>& 	triangulation,
    //                     Well_Set<dim>&                                     wells/*,
    //                     SourceSinks::Streams&                      streams*/,
    //                     double top_fraction, double bot_fraction);


private:
    MPI_Comm                                    mpi_communicator;
    DoFHandler<dim>&                            dof_handler;
    const FE_Q<dim>&                            fe;
    AffineConstraints<double>&                  constraints;
    IndexSet                                  	locally_owned_dofs;
    IndexSet                                  	locally_relevant_dofs;
    TrilinosWrappers::MPI::Vector& 				locally_relevant_solution;
    TrilinosWrappers::SparseMatrix 			  	system_matrix;
    TrilinosWrappers::MPI::Vector&       	  	system_rhs;
    std::map<types::boundary_id, const Function<dim>* >&				dirichlet_boundary;
    MyTensorFunction<dim>	 					HK;
    MyFunction<dim,dim> 						GWRCH;
    BoundaryConditions::Neumann<dim>            Neumann;
    std::vector<int>                            top_boundary_ids;
    SolverParameters                            solver_param;

    ConditionalOStream                        	pcout;
    TimerOutput                               	computing_timer;

    int                                         my_rank;
    int                                         n_proc;

    void setup_system();
    void assemble();
    void solve();
    void output(int iter, std::string output_file,
                parallel::distributed::Triangulation<dim>& 	triangulation,
                SimPrintFlags                                 printflags);
    void refine (parallel::distributed::Triangulation<dim>& 	triangulation,
                 double top_fraction, double bot_fraction);

    void output_xyz_top(int iter, std::string output_file);
    void output_DBC(int iter, std::string output_file);
    SolverControl::State writeSolverConverge(const unsigned int iteration, const double check_value, const TrilinosWrappers::MPI::Vector /*&current_iterate*/)const{
        pcout << "\tIter:" << iteration << ": " << check_value << std::endl;
        return SolverControl::success;
    };

};

template <int dim>
GWFLOW<dim>::GWFLOW(MPI_Comm&                           mpi_communicator_in,
                    DoFHandler<dim>&                    dof_handler_in,
                    const FE_Q<dim>&                    fe_in,
                    AffineConstraints<double>&          constraints_in,
                    TrilinosWrappers::MPI::Vector&      locally_relevant_solution_in,
                    TrilinosWrappers::MPI::Vector&      system_rhs_in,
                    std::map<types::boundary_id, const Function<dim>* >&    dirichlet_boundary_in,
                    MyTensorFunction<dim>&              HK_function,
                    MyFunction<dim, dim>&               groundwater_recharge,
                    BoundaryConditions::Neumann<dim>&   Neumann_conditions,
                    std::vector<int>&                   top_boundary_ids_in,
                    SolverParameters&                   solver_param_in)
    :
      mpi_communicator(mpi_communicator_in),
      dof_handler(dof_handler_in),
      fe(fe_in),
      constraints(constraints_in),
      locally_relevant_solution(locally_relevant_solution_in),
      system_rhs(system_rhs_in),
      dirichlet_boundary(dirichlet_boundary_in),
      HK(HK_function),
      GWRCH(groundwater_recharge),
      Neumann(Neumann_conditions),
      top_boundary_ids(top_boundary_ids_in),
      solver_param(solver_param_in),
      pcout(std::cout,(Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),
      computing_timer(pcout, TimerOutput::summary, TimerOutput::wall_times)
{
    my_rank = Utilities::MPI::this_mpi_process(mpi_communicator);
    n_proc = Utilities::MPI::n_mpi_processes(mpi_communicator);
}

template <int dim>
void GWFLOW<dim>::setup_system(){
    TimerOutput::Scope t(computing_timer, "setup");
    pcout << "\tSetting up system..." << std::endl << std::flush;
    dof_handler.distribute_dofs (fe);
    pcout   << "\t Number of degrees of freedom: "
            << dof_handler.n_dofs()
            << std::endl << std::flush;

    locally_owned_dofs = dof_handler.locally_owned_dofs ();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    locally_relevant_solution.reinit (locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

    system_rhs.reinit (locally_owned_dofs, mpi_communicator);
    //system_rhs = 0;

    constraints.clear ();
    constraints.reinit (locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints (dof_handler, constraints);
    VectorTools::interpolate_boundary_values(dof_handler, dirichlet_boundary, constraints);
    constraints.close ();

    DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs(),
                                                    dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern (dof_handler, dynamic_sparsity_pattern,constraints, false);
    SparsityTools::distribute_sparsity_pattern (dynamic_sparsity_pattern,
                                                dof_handler.n_locally_owned_dofs_per_processor(),
                                                mpi_communicator,
                                                locally_relevant_dofs);
    system_matrix.reinit (locally_owned_dofs,
                          locally_owned_dofs,
                          dynamic_sparsity_pattern,
                          mpi_communicator);
}

template <int dim>
void GWFLOW<dim>::assemble(){
    TimerOutput::Scope t(computing_timer, "assemble");
    pcout << "\t Assembling system..." << std::endl << std::flush;
    const QGauss<dim>  quadrature_formula(2);
    const QGauss<dim-1> face_quadrature_formula(2);

    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values    |  update_gradients |
                             update_quadrature_points |
                             update_JxW_values);
    FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                      update_values         | update_quadrature_points  | update_jacobians |
                                      update_normal_vectors | update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();

    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs (dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    std::vector<Tensor<2,dim> >	 		hydraulic_conductivity_values(n_q_points);
    std::vector<double>			 		recharge_values(n_face_q_points);


    if (std::abs(solver_param.rch_multiplier - 1) > 0.0001 ){
        pcout << "Recharge multiplier: " << solver_param.rch_multiplier << std::endl;
    }


    double QRCH_TOT = 0;
    std::vector<double> QFLOW_TOT(Neumann.Nbnd(),0);
    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell){
        if (cell->is_locally_owned()){
            cell_matrix = 0;
            cell_rhs = 0;
            fe_values.reinit (cell);

            HK.value_list(fe_values.get_quadrature_points(),
                          hydraulic_conductivity_values);
			
			bool print_this_cell = false;
            for (unsigned int q_point=0; q_point < n_q_points; ++q_point){
                for (unsigned int i=0; i < dofs_per_cell; ++i){
                    for (unsigned int j=0; j < dofs_per_cell; ++j){
                        cell_matrix(i,j) += (fe_values.shape_grad(i,q_point)*
                                             hydraulic_conductivity_values[q_point]*
                                             fe_values.shape_grad(j,q_point)*
                                             fe_values.JxW(q_point));
						if (std::isnan(fe_values.JxW(q_point)) == 1){
                            print_this_cell = true;
                        }
                    }
                }
            }

            if (print_this_cell)
                print_cell_coords<dim>(cell);

            for (unsigned int i_face=0; i_face < GeometryInfo<dim>::faces_per_cell; ++i_face){
                if(cell->face(i_face)->at_boundary()){
                    if ((cell->face(i_face)->boundary_id() == 5 && dim == 3) ||
                            (cell->face(i_face)->boundary_id() == 3 && dim == 2)){ // Top recharge
                        fe_face_values.reinit (cell, i_face);
                        double weight = recharge_weight<dim>(cell, i_face);
                        GWRCH.value_list(fe_face_values.get_quadrature_points(), recharge_values);

                        for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point){
                            for (unsigned int i = 0; i < dofs_per_cell; ++i){
                                double Q_rch = (recharge_values[q_point] * weight *
                                                fe_face_values.shape_value(i,q_point)*
                                                fe_face_values.JxW(q_point));
                                if (std::isnan(Q_rch) == 1){
                                    std::cout  << "Rank: " << my_rank << " Rval: " << recharge_values[q_point] << " | w: " << weight
                                              << " | ShVal: " << fe_face_values.shape_value(i,q_point)
                                              << " | JxW: " << fe_face_values.JxW(q_point) << std::endl;
								    //std::cout << "Bc: " << cell->barycenter() << std::endl;
								    print_cell_face_matlab<dim>(cell, i_face);
                                }

                                cell_rhs(i) += Q_rch * solver_param.rch_multiplier;
                                QRCH_TOT += Q_rch * solver_param.rch_multiplier;
                            }
                        }
                    }
                    if (Neumann.Nbnd() > 0){
                        for (int ibnd = 0; ibnd < Neumann.Nbnd(); ++ibnd){
                            if (((cell->face(i_face)->boundary_id() == 4 && dim == 3) ||
                                (cell->face(i_face)->boundary_id() == 2 && dim == 2)) &&
                                  Neumann.getType(ibnd) == BoundaryConditions::BoundaryType::BOT){ // Flows from bottom face
                                fe_face_values.reinit (cell, i_face);
                                std::vector<Point<dim> > q_pnts = fe_face_values.get_quadrature_points();
                                for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point){
                                    double flow_rate = Neumann.interpolate(q_pnts[q_point], ibnd);
                                    for (unsigned int i = 0; i < dofs_per_cell; ++i){
                                        double Q_flow = flow_rate *
                                                fe_face_values.shape_value(i,q_point)*
                                                fe_face_values.JxW(q_point);

                                        cell_rhs(i) += Q_flow;
                                        QFLOW_TOT[ibnd] += Q_flow;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            cell->get_dof_indices (local_dof_indices);
            constraints.distribute_local_to_global (cell_matrix,
                                                    cell_rhs,
                                                    local_dof_indices,
                                                    system_matrix,
                                                    system_rhs);

        }
    }

    MPI_Barrier(mpi_communicator);
    // Sum up Flow contributions for display
    sum_scalar<double>(QRCH_TOT, n_proc, mpi_communicator, MPI_DOUBLE);
    for (int ii = 0; ii < Neumann.Nbnd(); ++ii)
        sum_scalar<double>(QFLOW_TOT[ii], n_proc, mpi_communicator, MPI_DOUBLE);

    if (my_rank == 0){
        std::cout << "\t QRCH: [" << QRCH_TOT << "]" << std::endl;
        for (int ii = 0; ii < Neumann.Nbnd(); ++ii)
            std::cout << "\t QFLOW(" << ii << "): [" << QFLOW_TOT[ii] << "]" << std::endl;
    }
    MPI_Barrier(mpi_communicator);

    system_matrix.compress (VectorOperation::add);
    system_rhs.compress (VectorOperation::add);
}

template <int dim>
void GWFLOW<dim>::solve(){
    TimerOutput::Scope t(computing_timer, "solve");
    pcout << "\t Solving system..." << std::endl << std::flush;
    double solver_tolerance = solver_param.solver_tol*system_rhs.l2_norm();
    //double solver_tolerance = 0.000000001*system_rhs.l2_norm(); // use this for CVHM
    //pcout << "\t\t\t l2 norm: " << solver_tolerance << std::endl; // use this for CVHM
    //if (solver_tolerance < solver_param.solver_tol){ // use this for CVHM
    //    solver_tolerance = solver_param.solver_tol;
    //}

    pcout << "\t\t Relative Solver tolerance:" << solver_tolerance << std::endl;
    TrilinosWrappers::MPI::Vector completely_distributed_solution(locally_owned_dofs,mpi_communicator);
    SolverControl solver_control (dof_handler.n_dofs(), solver_tolerance, true, true);
    solver_control.log_result(true);
    solver_control.log_history(true);
    solver_control.log_frequency(1);
    //solver_control.enable_history_data();



    SolverCG<TrilinosWrappers::MPI::Vector>  solver (solver_control);
    //SolverGMRES<TrilinosWrappers::MPI::Vector> solver (solver_control); // use this for CVHM

    TrilinosWrappers::PreconditionAMG       preconditioner;
    TrilinosWrappers::PreconditionAMG::AdditionalData data;
    //TrilinosWrappers::PreconditionAMGMueLu  preconditioner; // use this for CVHM
    //TrilinosWrappers::PreconditionAMGMueLu::AdditionalData data; // use this for CVHM

    data.output_details = static_cast<bool>(solver_param.output_details);
    data.n_cycles = 1;
    data.w_cycle = false;
    data.aggregation_threshold = 0.0001;
    data.smoother_sweeps = 2;
    data.smoother_overlap = 0;
    data.smoother_type = "Chebyshev";
    data.coarse_type = "Amesos-KLU";

    //FEValuesExtractors::Scalar extractor(dim);
    //DoFTools::extract_constant_modes(dof_handler, dof_handler.get_fe_collection().component_mask(extractor),data.constant_modes);

    preconditioner.initialize (system_matrix, data);

    solver.connect(std::bind(&GWFLOW<dim>::writeSolverConverge, this, std::placeholders::_1,std::placeholders::_2,std::placeholders::_3));

    solver.solve (system_matrix,
                  completely_distributed_solution,
                  system_rhs,
                  preconditioner);


    pcout << "   Solved in " << solver_control.last_step()
          << " iterations." << std::endl << std::flush;

    constraints.distribute (completely_distributed_solution);
    locally_relevant_solution = completely_distributed_solution;
}

template <int dim>
void GWFLOW<dim>::output(int iter, std::string output_file,
                         parallel::distributed::Triangulation<dim>& 	triangulation,
                         SimPrintFlags                                 printflags){
    TimerOutput::Scope t(computing_timer, "output");
    pcout << "\t Printing results..." << std::endl << std::flush;

    if (printflags.print_vtk == 1){
        DataOut<dim> data_out;
        data_out.attach_dof_handler (dof_handler);
        data_out.add_data_vector (locally_relevant_solution, "Head");
        Vector<float> subdomain (triangulation.n_active_cells());
        for (unsigned int i = 0; i < subdomain.size(); ++i){
            subdomain(i) = triangulation.locally_owned_subdomain();
        }
        data_out.add_data_vector (subdomain, "subdomain");

        Vector<double> Conductivity (triangulation.n_active_cells());
        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();
        int cnt_cells = 0;
        for (; cell!=endc; ++cell){
            if (cell->is_locally_owned()){
                Tensor<2,dim> value = HK.value(cell->barycenter());
                Conductivity[cnt_cells] = value[0][0];
            }
            ++cnt_cells;
        }
        data_out.add_data_vector (Conductivity, "Conductivity",
                                  DataOut<dim>::type_cell_data);

        data_out.build_patches ();

        const std::string filename = (output_file +
                                      Utilities::int_to_string (iter, 3) +
                                      "." +
                                      Utilities::int_to_string
                                      (triangulation.locally_owned_subdomain(), 4));

        std::ofstream output ((filename + ".vtu").c_str());
        data_out.write_vtu (output);
        if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0){
            std::vector<std::string> filenames;
            for (unsigned int i=0; i < Utilities::MPI::n_mpi_processes(mpi_communicator); ++i){
                filenames.push_back (output_file +
                                     Utilities::int_to_string (iter, 3) +
                                     "." +
                                     Utilities::int_to_string (i, 4) +
                                     ".vtu");
            }
            const std::string pvtu_master_filename = (output_file +
                                                      Utilities::int_to_string(iter,3) + ".pvtu");
            std::ofstream pvtu_master (pvtu_master_filename.c_str());
            data_out.write_pvtu_record(pvtu_master, filenames);
            //const std::string visit_master_filename = (output_file +
            //                                            Utilities::int_to_string(iter, 3) + ".visit");
            //std::ofstream visit_master(visit_master_filename.c_str());
            //data_out.write_visit_record(visit_master, filenames);
        }
    }

    if (printflags.print_BND == 1)
        output_DBC(iter, output_file);

    // Write a point cloud of the Top surface only
    if (printflags.top_point_cloud > 0)
        output_xyz_top(iter, output_file);
}

template <int dim>
void GWFLOW<dim>::Simulate(int iter,                                     std::string output_file,
                           parallel::distributed::Triangulation<dim>& 	triangulation,
                           Well_Set<dim> &wells,
                           Streams<dim> &streams,
                           SimPrintFlags printflags){
    setup_system();


    wells.add_contributions(system_rhs,
                            dof_handler,
                            fe,
                            constraints,
                            HK,
                            mpi_communicator);

    streams.add_contributions(system_rhs,
                              dof_handler,
                              fe,
                              constraints,
                              top_boundary_ids, my_rank, n_proc, mpi_communicator);

    assemble();
    solve();
    output(iter, output_file, triangulation, printflags);
}

//template <int dim>
//void GWFLOW<dim>::Simulate_refine(int iter,                                     std::string output_file,
//                                  parallel::distributed::Triangulation<dim>& 	triangulation,
//                                  Well_Set<dim>&                                     wells/*,
//                                  SourceSinks::Streams&                      streams*/,
//                                  double top_fraction, double bot_fraction){

//    Simulate(iter,output_file,triangulation,wells);

//    refine(triangulation, top_fraction, bot_fraction);

//    int my_rank = Utilities::MPI::this_mpi_process(mpi_communicator);
//    std::ofstream out ("test_tria" + std::to_string(my_rank) + ".vtk");
//    GridOut grid_out;
//    grid_out.write_ucd(triangulation, out);

//}

template <int dim>
void GWFLOW<dim>::refine(parallel::distributed::Triangulation<dim>& 	triangulation,
                         double top_fraction, double bot_fraction){
    TimerOutput::Scope t(computing_timer, "refine");
    Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate (dof_handler,
                                          QGauss<dim-1>(fe.degree+2),
                                          {},
                                          locally_relevant_solution,
                                          estimated_error_per_cell);

    parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                             estimated_error_per_cell,
                                             top_fraction,
                                             bot_fraction);

    triangulation.execute_coarsening_and_refinement ();

}

template <int dim>
void GWFLOW<dim>::output_xyz_top(int iter, std::string output_file){

    const std::string top_filename = (output_file + "top_" +
                                      Utilities::int_to_string(iter,3) + "_" +
                                      Utilities::int_to_string(my_rank,4) +
                                      ".xyz");

    std::ofstream top_stream_file;
    top_stream_file.open(top_filename.c_str());

    QTrapez<dim-1> face_trapez_formula;
    FEFaceValues<dim> fe_face_values(fe, face_trapez_formula, update_values);
    std::vector< double > values(face_trapez_formula.size());
    // solution_points is a vector of vectors of DIM+1 (x, y, z, value) or (x, y, value)
    std::vector<std::vector<double>> solution_points;
    CGAL::Point_set_2<ine_Kernel, Tds> PointSet;

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell){
        if (cell->is_locally_owned()){
            for (unsigned int i_face = 0; i_face < GeometryInfo<dim>::faces_per_cell; ++i_face){
                if (cell->face(i_face)->at_boundary()){
                    for (unsigned int i = 0; i < top_boundary_ids.size(); ++i){
                        if (cell->face(i_face)->boundary_id() == static_cast<unsigned int>(top_boundary_ids[i])){
                            // This is a top boundary face
                            fe_face_values.reinit (cell, i_face);
                            fe_face_values.get_function_values(locally_relevant_solution, values);
                            for (unsigned int ii = 0; ii < face_trapez_formula.size(); ++ii){
                                Point<dim> current_point = cell->face(i_face)->vertex(ii);
                                std::vector<int> ids;
                                if (dim == 2)
                                    ids = circle_search_in_2DSet(PointSet, ine_Point3(current_point[0], 0, 0 ), 0.001);
                                else if (dim == 3)
                                    ids = circle_search_in_2DSet(PointSet, ine_Point3(current_point[0], current_point[1], 0 ), 0.001);
                                else
                                    std::cerr << "Seriously??" << std::endl;

                                if (ids.size() == 0){
                                    // Insert the point
                                    std::vector< std::pair<ine_Point2, unsigned> > newpoint;
                                    if (dim == 2)
                                        newpoint.push_back(std::make_pair(ine_Point2(current_point[0], 0), solution_points.size()));
                                    else if (dim == 3)
                                        newpoint.push_back(std::make_pair(ine_Point2(current_point[0], current_point[1]), solution_points.size()));
                                    else
                                        std::cerr << "I know you can do better" << std::endl;
                                    PointSet.insert(newpoint.begin(), newpoint.end());
                                    std::vector<double> v_temp;
                                    for (unsigned int idim = 0; idim < dim; idim++)
                                        v_temp.push_back(current_point[idim]);
                                    v_temp.push_back(values[ii]);
                                    solution_points.push_back(v_temp);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    top_stream_file << solution_points.size() << std::endl;
    for (unsigned int i = 0; i < solution_points.size(); i++){
        for (unsigned int idim = 0; idim < dim; idim++){
            top_stream_file <<  std::setprecision(15) << solution_points[i][idim] << " ";
        }
        top_stream_file <<  std::setprecision(15) << solution_points[i][dim] << std::endl;
    }
    top_stream_file.close();
}


template <int dim>
void GWFLOW<dim>::output_DBC(int iter, std::string output_file){
    const std::string dbc_filename = (output_file + "DBC_" +
                                      Utilities::int_to_string(iter,3) + "_" +
                                      Utilities::int_to_string(my_rank,4) +
                                      ".dat");
    std::ofstream dbc_stream_file;
    dbc_stream_file.open(dbc_filename.c_str());

    const QGauss<dim-1> face_quadrature_formula(2);
    FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                      update_values | update_gradients         | update_quadrature_points |
                                      update_normal_vectors | update_JxW_values);
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_face_q_points = face_quadrature_formula.size();
    std::vector<Tensor<1, dim> >        head_grad_values(n_face_q_points);
    std::vector<Tensor<1, dim> >        normal_vectors(n_face_q_points);
    std::vector<Tensor<2,dim> >	 		hydraulic_conductivity_values(n_face_q_points);

    typename std::map<types::boundary_id, const Function<dim>* >::iterator itbc;

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell){
        if (cell->is_locally_owned()){
            for (unsigned int i_face = 0; i_face < GeometryInfo<dim>::faces_per_cell; ++i_face){
                if (cell->face(i_face)->at_boundary()){

                    for (itbc = dirichlet_boundary.begin(); itbc != dirichlet_boundary.end(); ++itbc){
                        if (cell->face(i_face)->boundary_id() == itbc->first){
                            //std::cout << cell->face(i_face)->boundary_id() << std::endl;
                            fe_face_values.reinit (cell, i_face);
                            Point<dim> b = cell->face(i_face)->center();
                            //std::cout << b << std::endl;
                            //for (unsigned int ipnt = 0; ipnt <GeometryInfo<dim>::vertices_per_face; ++ipnt){
                            //   std::cout << cell->face(i_face)->vertex(ipnt) << std::endl;
                            //}

                            fe_face_values.get_function_gradients(locally_relevant_solution, head_grad_values);
                            normal_vectors = fe_face_values.get_normal_vectors();
                            HK.value_list(fe_face_values.get_quadrature_points(), hydraulic_conductivity_values);
                            Tensor<1, dim> q_darcy;
                            Tensor<1, dim> normal;
                            for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point){
                                for (unsigned int i = 0; i < dofs_per_cell; ++i){
                                    for (unsigned int idim = 0; idim < dim; ++idim){
                                        double shapexJxW = fe_face_values.shape_value(i,q_point)*
                                                           fe_face_values.JxW(q_point);

                                        q_darcy[idim] += -(hydraulic_conductivity_values[q_point][idim][idim]*
                                                           head_grad_values[q_point][idim]*shapexJxW);
                                        normal[idim] += normal_vectors[q_point][idim]*shapexJxW;
                                    }
                                }
                            }
                            normal = normal/normal.norm();

                            bool print_point = false;
                            if (print_point){
                                if (dim == 3)
                                    dbc_stream_file << std::setprecision(3) << std::fixed << b[0] << " " << b[1] << " " << b[2] << " "
                                                                            << q_darcy[0] << " " << q_darcy[1] << " " << q_darcy[2] << " "
                                                                            << normal[0] << " " << normal[1] << " " << normal[2] << std::endl;
                                else if (dim == 2)
                                    dbc_stream_file << std::setprecision(3) << std::fixed << b[0] << " " << b[1] << " "
                                                                            << q_darcy[0] << " " << q_darcy[1] << " "
                                                                            << normal[0] << " " << normal[1] << std::endl;
                            }
                            else{
                                if (dim == 3){
                                    dbc_stream_file << std::setprecision(3) << std::fixed
                                                    << q_darcy[0] << " " << q_darcy[1] << " " << q_darcy[2] << " ";
                                    for (unsigned int ivert = 0; ivert < GeometryInfo<dim>::vertices_per_face; ++ivert){
                                        dbc_stream_file << std::setprecision(3) << std::fixed
                                                        << cell->face(i_face)->vertex(ivert)[0] << " "
                                                        << cell->face(i_face)->vertex(ivert)[1] << " "
                                                        << cell->face(i_face)->vertex(ivert)[2] << " ";
                                    }
                                    dbc_stream_file << std::endl;
                                }
                            }
                            //std::cout << q_darcy << std::endl;
                            //std::cout << normal << std::endl;
                        }
                    }
                }
            }
        }
    }
    dbc_stream_file.close();
}

#endif // STEADY_STATE_H
