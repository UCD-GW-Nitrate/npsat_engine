#ifndef PARTICLE_TRACKING_H
#define PARTICLE_TRACKING_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/base/conditional_ostream.h>

#include "my_functions.h"
#include "dsimstructs.h"
#include "streamlines.h"
#include "cgal_functions.h"


using namespace dealii;

template <int dim>
class Particle_Tracking{
public:
    Particle_Tracking(MPI_Comm& mpi_communicator_in,
                      DoFHandler<dim>& dof_handler_in,
                      FE_Q<dim>& fe_in,
                      TrilinosWrappers::MPI::Vector& 	locally_relevant_solution_in,
                      MyTensorFunction<dim>& HK_function_in,
                      MyFunction<dim, dim>& porosity_in,
                      ParticleParameters& param_in);

    void trace_particles(std::vector<Streamline<dim>>& streamlines, int iter, std::string prefix);

private:
    MPI_Comm                            mpi_communicator;
    DoFHandler<dim>&                    dof_handler;
    FE_Q<dim>&                          fe;
    TrilinosWrappers::MPI::Vector       locally_relevant_solution;
    MyTensorFunction<dim>               HK_function;
    MyFunction<dim, dim>                porosity;
    ConditionalOStream                  pcout;
    ParticleParameters                  param;

    bool internal_backward_tracking(typename DoFHandler<dim>::active_cell_iterator cell, Streamline<dim>& streamline);
    int compute_point_velocity(Point<dim>& p, Point<dim>& v, typename DoFHandler<dim>::active_cell_iterator cell);
};

template <int dim>
Particle_Tracking<dim>::Particle_Tracking(MPI_Comm& mpi_communicator_in,
                                          DoFHandler<dim>& dof_handler_in,
                                          FE_Q<dim> &fe_in,
                                          TrilinosWrappers::MPI::Vector& 	locally_relevant_solution_in,
                                          MyTensorFunction<dim>& HK_function_in,
                                          MyFunction<dim, dim>& porosity_in,
                                          ParticleParameters& param_in)
    :
    mpi_communicator(mpi_communicator_in),
    dof_handler(dof_handler_in),
    fe(fe_in),
    locally_relevant_solution(locally_relevant_solution_in),
    HK_function(HK_function_in),
    porosity(porosity_in),
    param(param_in),
    pcout(std::cout,(Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
{}

template <int dim>
void Particle_Tracking<dim>::trace_particles(std::vector<Streamline<dim>>& streamlines, int iter, std::string prefix){
    unsigned int my_rank = Utilities::MPI::this_mpi_process(mpi_communicator);
    unsigned int n_proc = Utilities::MPI::n_mpi_processes(mpi_communicator);

    //This is the name file where all particle trajectories are written
    const std::string log_file_name = (prefix + "_" +
                                           Utilities::int_to_string(iter, 4) +
                                           "_particles_"	+
                                           Utilities::int_to_string(my_rank, 4) +
                                           ".traj");
    //This is the name file where we print info of particles that terminate abnornamly
    const std::string err_file_name = (prefix + "_" +
                                       Utilities::int_to_string(iter, 4) +
                                       "_particle_errors_"	+
                                       Utilities::int_to_string(my_rank, 4) +
                                       ".traj");
    std::ofstream log_file;
    std::ofstream err_file;
    log_file.open(log_file_name.c_str());
    err_file.open(err_file_name.c_str());
    std::vector<Streamline<dim>> new_particles;

    int trace_iter = 0;
    int cnt_stuck_particles = 0;
    while (true){
        new_particles.clear();

        // make a Point Set for faster query of particles
        std::vector<ine_Key> prtclsxy;
        for (unsigned int i = 0; i < streamlines.size(); ++i){
            if (dim == 2){
                prtclsxy.push_back(ine_Key(ine_Point3(streamlines[i].P[0][0],
                                                      streamlines[i].P[0][1],
                                                      0), i) );
            }
            else if (dim == 3){
                prtclsxy.push_back(ine_Key(ine_Point3(streamlines[i].P[0][0],
                                                      streamlines[i].P[0][1],
                                                      streamlines[i].P[0][2]), i) );
            }
        }
        Range_tree_3_type ParticlesXY(prtclsxy.begin(), prtclsxy.end());
        int cnt_ptr = 0;int cnt_cells = 0;
        typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();
        for (; cell!=endc; ++cell){
            if (cell->is_locally_owned()){

                std::vector<int> particle_id_in_cell;
                // find the lower and upper points of the cell
                Point<dim>ll;
                Point<dim>uu;
                for (unsigned int ii = 0; ii < dim; ++ii){
                    ll[ii] = 100000000;
                    uu[ii] = -100000000;
                }

                for (unsigned int ii = 0; ii < GeometryInfo<dim>::vertices_per_cell; ++ii){
                    for (unsigned int jj = 0; jj < dim; ++jj){
                        if (ll[jj] > cell->vertex(ii)[jj])
                            ll[jj] = cell->vertex(ii)[jj]-10;
                        if (uu[jj] < cell->vertex(ii)[jj])
                            uu[jj] = cell->vertex(ii)[jj]+10;
                    }
                }

                // Find which particles are inside this Cell bounding box that defined previously
                bool are_particles = any_point_inside(ParticlesXY, ll, uu, particle_id_in_cell);
                if (!are_particles)
                    continue;

                // loop through each point found in the cell box
                for (unsigned int jj = 0; jj < particle_id_in_cell.size(); ++jj){
                    int iprt = particle_id_in_cell[jj];
                    bool is_particle_inside = cell->point_inside(streamlines[iprt].P[0]);
                    if (is_particle_inside){
                        int outcome = internal_backward_tracking(cell, streamlines[iprt]);
                    }


                }

            }
        }
    }
}

template <int dim>
bool Particle_Tracking<dim>::internal_backward_tracking(typename DoFHandler<dim>::active_cell_iterator cell, Streamline<dim>& streamline){
    // ++++++++++ CONVERT THIS TO ENUMERATION+++++++++++
    int reason_to_exit= -99;
    int cnt_iter = 0;
    while(cnt_iter < param.streaml_iter){
        if (cnt_iter == 0){ // If this is the starting point of the streamline we need to compute the velocity
            Point<dim> v;
            reason_to_exit = compute_point_velocity(streamline.P[streamline.size()-1], v, cell);

        }
    }
}

template <int dim>
int Particle_Tracking<dim>::compute_point_velocity(Point<dim>& p, Point<dim>& v, typename DoFHandler<dim>::active_cell_iterator cell){
    int outcome = 0;
    Point<dim> p_unit;
    const MappingQ1<dim> mapping;
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    bool cell_found = false;

    bool success = try_mapping(p, p_unit, cell, mapping);
    if (success){
        cell_found = true;
        for (unsigned int i = 0; i < dim; ++i){
            if (p_unit[i] < 0 || p_unit[i] > 1)
                cell_found = false;
        }
    }

    //First we have to make sure that the point is in the cell
    if (!cell_found){
        if (!cell->point_inside(p)){
            // if the point is outside of the cell then search all neighbors until the point is found
            // or until we have search enough neighbors to make sure that the point is actually outside of the domain

            std::vector<typename DoFHandler<dim>::active_cell_iterator> tested_cells;
            std::vector<typename DoFHandler<dim>::active_cell_iterator> adjacent_cells;
            typename DoFHandler<dim>::active_cell_iterator neighbor_child;
            tested_cells.push_back(cell);
            std::vector<Point<dim>> cells_checked;
            cells_checked.push_back(cell->center());
            int nSearch;
            while (nSearch < param.search_iter){
                for (unsigned int i = 0; i < tested_cells.size(); ++i){
                    // for each face of the tested cell check its neighbors
                    for (unsigned int j = 0; j < GeometryInfo<dim>::faces_per_cell; ++j){
                        if (tested_cells[i]->at_boundary(j))
                            continue;
                        if(!tested_cells[i]->neighbor(j)->active()){
                            // if the neighbor cell is not active then it has children
                            // Here we check those children that touch the face of the tested cell
                            for (unsigned int ichild = 0; ichild < tested_cells[i]->neighbor(j)->n_childern(); ++ ichild){
                                // if the child has children, this child doesnt not touch the test cell and we will examine it later
                                if(!tested_cells[i]->neighbor(j)->child(ichild)->active())
                                    continue;
                                neighbor_child = tested_cells[i]->neighbor(j)->child(ichild);
                                // Check if this cell has been tested
                                double dst = 0.0;
                                for (unsigned int ii = 0; ii < cells_checked.size(); ++ii){
                                    dst = cells_checked[ii].distance(neighbor_child->center());
                                    if (dst < 0.001)
                                        break;
                                }
                                if (dst > 0.0001){// none of the cell centers of the checked cells is almost identical
                                                  // to this neighbor so we will add them to the list of cells for check
                                    adjacent_cells.push_back(neighbor_child);
                                }
                            }
                        }
                        else{
                            // if the neighbor is active then we check first if we have already checked it
                            double dst = 0.0;
                            for (unsigned int ii = 0; ii < cells_checked.size(); ++ii){
                                dst = cells_checked[ii].distance(tested_cells[i]->neighbor(j)->center());
                                if (dst < 0.001)
                                    break;
                            }
                            if (dst < 0.001){// none  of the cell centers of the checked cells is almost identical
                                             // to this neighbor so we will add it to the list of the cells for check
                                adjacent_cells.push_back(tested_cells[i]->neighbor(j));
                            }
                        }
                    }
                }

                // The adjacent_cells is a list of cells that are likely to contain the point
                tested_cells.clear();
                for (unsigned int i = 0; i < adjacent_cells.size(); ++i){
                    bool is_in_cell = adjacent_cells[i]->point_inside(p);
                    if (is_in_cell){
                        cell_found = true;
                        cell = adjacent_cells[i];
                        break;
                    }
                    else{
                        tested_cells.push_back(adjacent_cells[i]);
                        cells_checked.push_back(adjacent_cells[i]->center());
                    }
                }

                nSearch++;
                adjacent_cells.clear();
                if (cell_found)
                    break;
            }
        }
        else{
            cell_found = true;
        }

    }


}




#endif // PARTICLE_TRACKING_H
