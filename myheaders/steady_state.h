#ifndef STEADY_STATE_H
#define STEADY_STATE_H

#include <math.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_solver.h>
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
#include "my_functions.h"
#include "helper_functions.h"
#include "wells.h"
#include "streams.h"
#include "cgal_functions.h"

using namespace dealii;

template<int dim>
class GWFLOW{
public:
    GWFLOW(MPI_Comm&                            mpi_communicator_in,
           DoFHandler<dim>&                     dof_handler,
           const FE_Q<dim>&                     fe,
           ConstraintMatrix&                    constraints,
           TrilinosWrappers::MPI::Vector&       locally_relevant_solution,
           typename FunctionMap<dim>::type&     dirichlet_boundary,
           MyTensorFunction<dim>&               HK_function,
           MyFunction<dim,dim>&               groundwater_recharge,
           std::vector<int>&                    top_boundary_ids);


    void Simulate(int iter,                                     std::string output_file,
                  parallel::distributed::Triangulation<dim>& 	triangulation,
                  Well_Set<dim>&                                wells,
                  Streams<dim>&                                 streams);

    //void Simulate_refine(int iter,                                     std::string output_file,
    //                     parallel::distributed::Triangulation<dim>& 	triangulation,
    //                     Well_Set<dim>&                                     wells/*,
    //                     SourceSinks::Streams&                      streams*/,
    //                     double top_fraction, double bot_fraction);


private:
    MPI_Comm                                    mpi_communicator;
    DoFHandler<dim>&                            dof_handler;
    const FE_Q<dim>&                            fe;
    ConstraintMatrix&                          	constraints;
    IndexSet                                  	locally_owned_dofs;
    IndexSet                                  	locally_relevant_dofs;
    TrilinosWrappers::MPI::Vector& 				locally_relevant_solution;
    TrilinosWrappers::SparseMatrix 			  	system_matrix;
    TrilinosWrappers::MPI::Vector       	  	system_rhs;
    typename FunctionMap<dim>::type				dirichlet_boundary;
    MyTensorFunction<dim>	 					HK;
    MyFunction<dim,dim> 						GWRCH;
    std::vector<int>                            top_boundary_ids;
    ConditionalOStream                        	pcout;
    TimerOutput                               	computing_timer;
    int                                         my_rank;
    int                                         n_proc;

    void setup_system();
    void assemble();
    void solve();
    void output(int iter, std::string output_file,
                parallel::distributed::Triangulation<dim>& 	triangulation);
    void refine (parallel::distributed::Triangulation<dim>& 	triangulation,
                 double top_fraction, double bot_fraction);

    void output_xyz_top(int iter, std::string output_file);
};

template <int dim>
GWFLOW<dim>::GWFLOW(MPI_Comm&                            mpi_communicator_in,
                    DoFHandler<dim>&                     dof_handler_in,
                    const FE_Q<dim>&                     fe_in,
                    ConstraintMatrix&                    constraints_in,
                    TrilinosWrappers::MPI::Vector&       locally_relevant_solution_in,
                    typename FunctionMap<dim>::type&     dirichlet_boundary_in,
                    MyTensorFunction<dim>&               HK_function,
                    MyFunction<dim, dim> &groundwater_recharge,
                    std::vector<int>&                    top_boundary_ids_in)
    :
      mpi_communicator(mpi_communicator_in),
      dof_handler(dof_handler_in),
      fe(fe_in),
      constraints(constraints_in),
      locally_relevant_solution(locally_relevant_solution_in),
      dirichlet_boundary(dirichlet_boundary_in),
      HK(HK_function),
      GWRCH(groundwater_recharge),
      top_boundary_ids(top_boundary_ids_in),
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

    double QRCH_TOT = 0;
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
            for (unsigned int q_point=0; q_point<n_q_points; ++q_point){
                for (unsigned int i=0; i<dofs_per_cell; ++i){
                    for (unsigned int j=0; j<dofs_per_cell; ++j){
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
                            (cell->face(i_face)->boundary_id() == 3 && dim == 2)){
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

                                cell_rhs(i) += Q_rch;
                                QRCH_TOT += Q_rch;
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
    sum_scalar<double>(QRCH_TOT, n_proc, mpi_communicator, MPI_DOUBLE);
    if (my_rank == 0)
        std::cout << "\t QRCH: [" << QRCH_TOT << "]" << std::endl;
    MPI_Barrier(mpi_communicator);

    system_matrix.compress (VectorOperation::add);
    system_rhs.compress (VectorOperation::add);
}

template <int dim>
void GWFLOW<dim>::solve(){
    TimerOutput::Scope t(computing_timer, "solve");
    pcout << "\t Solving system..." << std::endl << std::flush;
    TrilinosWrappers::MPI::Vector completely_distributed_solution(locally_owned_dofs,mpi_communicator);
    SolverControl solver_control (dof_handler.n_dofs(), 1e-8);
    solver_control.log_result(true);
    solver_control.log_history(true);
    solver_control.log_frequency(0);

    SolverCG<TrilinosWrappers::MPI::Vector>  solver (solver_control);
    TrilinosWrappers::PreconditionAMG       preconditioner;
    TrilinosWrappers::PreconditionAMG::AdditionalData data;
    data.output_details = false;
    data.n_cycles = 1;
    data.w_cycle = false;
    data.aggregation_threshold = 0.000001;
    data.smoother_sweeps = 4;
    data.smoother_overlap = 0;
    data.smoother_type = "Chebyshev";
    data.coarse_type = "Amesos-KLU";

    preconditioner.initialize (system_matrix, data);

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
                         parallel::distributed::Triangulation<dim>& 	triangulation){
    TimerOutput::Scope t(computing_timer, "output");
    pcout << "\t Printing results..." << std::endl << std::flush;

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

    // Write a point cloud of the Top surface only
    output_xyz_top(iter, output_file);
}

template <int dim>
void GWFLOW<dim>::Simulate(int iter,                                     std::string output_file,
                           parallel::distributed::Triangulation<dim>& 	triangulation,
                           Well_Set<dim> &wells,
                           Streams<dim> &streams){
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
    output(iter, output_file, triangulation);
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
                                          typename FunctionMap<dim>::type(),
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

    const std::string top_filename = (output_file + "_top_" +
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
                                    std::cerr << "That was a good one!!" << std::endl;

                                if (ids.size() == 0){
                                    // Insert the point
                                    std::vector< std::pair<ine_Point2, unsigned> > newpoint;
                                    if (dim == 2)
                                        newpoint.push_back(std::make_pair(ine_Point2(current_point[0], 0), solution_points.size()));
                                    else if (dim == 3)
                                        newpoint.push_back(std::make_pair(ine_Point2(current_point[0], current_point[1]), solution_points.size()));
                                    else
                                        std::cerr << "Don't give up your day job " << std::endl;
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

#endif // STEADY_STATE_H
