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
#include "mpi_help.h"


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
    ~Particle_Tracking();

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

    bool                                bprint_DBG;
    std::ofstream                       dbg_file;
    int                                 dbg_i_strm;
    int                                 dbg_i_step;


    int internal_backward_tracking(typename DoFHandler<dim>::active_cell_iterator cell, Streamline<dim> &streamline);
    int compute_point_velocity(Point<dim>& p, Point<dim>& v, typename DoFHandler<dim>::active_cell_iterator &cell);
    int find_next_point(Streamline<dim> &streamline, typename DoFHandler<dim>::active_cell_iterator &cell);
    void Send_receive_particles(std::vector<Streamline<dim>>    new_particles,
                                std::vector<Streamline<dim>>	&streamlines);

    /**
     * @brief check_cell_point tests the spatial relationship between a given cell and a given point.
     *
     * First checks if the point is inside the cell by calling the dealii method point_inside.
     *
     * If the point is not inside the input cell, then we loop through the cell faces to check if any neighbor cells contain the point.
     *
     * This is executed using the following logic:
     *
     * First we create a list of tested cells and a list of adjacent cells. Initially the adjacent list is empty and the tested cells contains only the initial cell.
     * For each cell in the tested cell list we loop through the cells that touch each of the face of the tested cell and if those cells have not visited already
     * we add them to the adjacent cell list.
     * Next we check the cells int the list of adjacent cells if they contain the point. In addition each of the adjacent cells become tested cell for the next iteration.
     * The number of times we do that is choosen by the user. Typicaly 3 is more than enough.
     * @param cell This is the initial cell. In the case that the return value is 0  or 1 the cell is the same as the input.
     * If the return value is 2 or 3 then the cell reference changes to point the new cell.
     * However if the return value is negative then the cell still points to the original cell.
     * @param p
     * @return Returns
     *      - 0 if the point has not been found inside any cell. This means that the point is possibly outside the domain
     *      - 1 if the point is inside the cell.
     *      - 2 if the point is found inside an adjacent cell that is locally owned
     *      - 3 if the point is found inside an adjacent cell that is ghost
     *      - -3 if the point is found inside an adjacent cell that is artificial.
     * The last case indicates that something wrong happens. For example the step maybe too large so it skips the ghost cell.
     * That's why when this returns negative values we tell the algorithm to take one very small euler step instead of a larger RK4 step
     */
    int check_cell_point(typename DoFHandler<dim>::active_cell_iterator &cell, Point<dim>& p);

    /**
     * @brief compute_point_velocity Computes the velocity of the point p that is found within the cell. This version takes into account the
     * outcome of the check_cell_point method as the last input.
     * @param p
     * @param v
     * @param cell
     * @param check_point_status
     * @return it returns the following values
     *      - -99 if the cell is artificial.
     *      - -88 if the mapping transformation has failed to compute the unit coordinates of the p point.
     *      - 1 if the particle has exited the domain from the top face of the cell. This is the most common case
     *      - -9 if the particle has exited the domain from the bottom face of the cell. if the bottom was supposed to be impermeable boundary that's very wrong
     *      - 2 if the point has left the cell from the side. This is normal if the face is lateral flow boundary
     *      - 0 if the computation of velocity was successful
     *      - -101 If something else obviously wrong has happened
     *
     */
    int compute_point_velocity(Point<dim>& p, Point<dim>& v, typename DoFHandler<dim>::active_cell_iterator &cell, int check_point_status);

    double calculate_step(typename DoFHandler<dim>::active_cell_iterator cell, Point<dim> Vel);

    /**
     * @brief add_streamline_point Adds the point and velocity to the streamline. First checks if the cell is locally owned.
     * If yes adds the point and the velocity. If not adds only the point. This particle position will transfer to other processors
     * and the one that onws the cell that this particle is will the compute the velocity.
     * @param cell
     * @param streamline
     * @param p
     * @param vel
     * @return
     *      - 0  if the cell is locally owned
     *      - 55 if the cell is artificial or ghost
     */
    int add_streamline_point(typename DoFHandler<dim>::active_cell_iterator &cell,
                             Streamline<dim> &streamline,
                             Point<dim> p, Point<dim> vel, int return_value);

    int take_euler_step(typename DoFHandler<dim>::active_cell_iterator &cell,
                        double step_weight, double step_length,
                        Point<dim> P_prev, Point<dim> V_prev,
                        Point<dim>& P_next, Point<dim>& V_next, int& count_nest);

    /**
     * @brief time_step_multiplier calculates a step multiplier for the imput step.
     * In general the step is defined by the user by setting  the #ParticleParameters::step_size
     * variable via the input file "g Step size". This method further reduces the this time step
     * if certain criteria are met. For example if any face of this cell is at boundary then the
     * time step is reduces by 90%. If any neighbor cells is ghost or artificial the time step reduces
     * by 75%. Last if any neighbor cell touch the boundary the the time step is reduced by 50%.
     * If more than two criteria are met then the time multiplier returns the smallest value.
     *
     * Note that this is called once at the begin of the #find_next_point method.
     *
     * @param cell
     * @return returns the time step multiplier.
     */
    double time_step_multiplier(typename DoFHandler<dim>::active_cell_iterator cell);

    void plot_cell(typename DoFHandler<dim>::active_cell_iterator cell);
    void plot_point(Point<dim> p);
    void plot_segment(Point<dim> A, Point<dim> B);
    void print_Cell_var(typename DoFHandler<dim>::active_cell_iterator cell, int cell_type);
    void print_point_var(Point<dim> p, int r);
    void print_strm_exit_info(int r);
    int cell_type(typename DoFHandler<dim>::active_cell_iterator cell);

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
{

    bprint_DBG = true;
    if (bprint_DBG){
        unsigned int my_rank = Utilities::MPI::this_mpi_process(mpi_communicator);
        const std::string dbg_file_name = ("particles_dbg" +
                                           Utilities::int_to_string(my_rank, 4) +
                                           ".m");
        dbg_file.open(dbg_file_name.c_str());
        dbg_i_step = 1;
        dbg_i_strm = 1;
    }
}

template <int dim>
Particle_Tracking<dim>::~Particle_Tracking(){
    dbg_file.close();
}

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
                            ll[jj] = cell->vertex(ii)[jj]-1;
                        if (uu[jj] < cell->vertex(ii)[jj])
                            uu[jj] = cell->vertex(ii)[jj]+1;
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
                        //std::cout << iprt << " : " << streamlines[iprt].E_id << " : " << streamlines[iprt].S_id << std::endl;
                        int outcome = internal_backward_tracking(cell, streamlines[iprt]);
                        if (outcome == -88){// the transformation of the point has failed
                            err_file << "transformation failed" << ",  \t"
                                     << streamlines[iprt].E_id << ",  \t"
                                     << streamlines[iprt].S_id << std::endl;
                            continue;
                        }
                        if (outcome == -66){ // The particle has stuck
                            err_file << "Particle stuck" << ",  \t"
                                     << streamlines[iprt].E_id << ",  \t"
                                     << streamlines[iprt].S_id << std::endl;
                        }
                        // Print the particle positions in the file
                        for (unsigned int i = 0; i < streamlines[iprt].V.size(); ++i){
                            log_file << streamlines[iprt].E_id << "  \t"
                                     << streamlines[iprt].S_id << "  \t"
                                     << outcome << "  \t"
                                     << streamlines[iprt].p_id[i] << "  \t"
                                     << std::setprecision(15);
                            for (unsigned int idim = 0; idim < dim; ++idim)
                                log_file << streamlines[iprt].P[i][idim] << "  \t";
                            for (unsigned int idim = 0; idim < dim; ++idim)
                                log_file << streamlines[iprt].V[i][idim] << "  \t";
                            log_file << std::endl;
                        }

                        if (outcome == 55){
                            // this particle will continue to another processor
                            int n = streamlines[iprt].P.size()-1;
                            Streamline<dim> temp_strm(streamlines[iprt].E_id,
                                                      streamlines[iprt].S_id,
                                                      streamlines[iprt].P[n]);
                            temp_strm.p_id[0] = streamlines[iprt].p_id[n];
                            temp_strm.proc_id = streamlines[iprt].proc_id;
                            temp_strm.BBl = streamlines[iprt].BBl;
                            temp_strm.BBu = streamlines[iprt].BBu;
                            new_particles.push_back(temp_strm);
                        }
                    }
                }
            }
        }
        MPI_Barrier(mpi_communicator);
        std::cout<< "I'm proc" << my_rank << " and have " << new_particles.size() << " particles to send" << std::endl << std::flush;
        MPI_Barrier(mpi_communicator);

        std::vector<int> new_part_per_proc(n_proc);
        Send_receive_size(new_particles.size(), n_proc, new_part_per_proc, mpi_communicator);

        int max_N_part = 0;
        for (unsigned int i = 0; i < n_proc; ++i)
            max_N_part += new_part_per_proc[i];
        MPI_Barrier(mpi_communicator);
        pcout << "------ Number of active particles: " << max_N_part << " --------" << std::endl << std::flush;
        //std::cout << my_rank << " : " << max_N_part << std::endl;

        if (trace_iter>3)
            return;

        if (max_N_part == 0)
            break;


        Send_receive_particles(new_particles, streamlines);

        if (++trace_iter > param.Outmost_iter)
            break;
    }
    log_file.close();
    err_file.close();
}

template <int dim>
int Particle_Tracking<dim>::internal_backward_tracking(typename DoFHandler<dim>::active_cell_iterator cell, Streamline<dim>& streamline){
    // ++++++++++ CONVERT THIS TO ENUMERATION+++++++++++
    int reason_to_exit= -99;
    int cnt_iter = 0;
    if (bprint_DBG){
        print_Cell_var(cell, cell_type(cell));
        print_point_var(streamline.P[streamline.P.size()-1], 1000);
    }
    while(cnt_iter < param.streaml_iter){
        if (cnt_iter == 0){ // If this is the starting point of the streamline we need to compute the velocity
            Point<dim> v;
            reason_to_exit = compute_point_velocity(streamline.P[streamline.P.size()-1], v, cell);
            if (reason_to_exit != 0)
                return  reason_to_exit;
            else{
                streamline.V.push_back(v);
                //plot_cell(cell);
            }
        }
        // if this is not the starting point we already know the velocity at the current point
        // The following function returns both the position with velocity
        reason_to_exit = find_next_point(streamline, cell);
        if (streamline.times_not_expanded > param.Stuck_iter){
            reason_to_exit = -66;
            return reason_to_exit;
        }

        if ( reason_to_exit != 0 )
            break;
        cnt_iter++;
    }
    print_strm_exit_info(reason_to_exit);
    return  reason_to_exit;
}

template <int dim>
int Particle_Tracking<dim>::check_cell_point(typename DoFHandler<dim>::active_cell_iterator& cell, Point<dim>& p){
    int outcome = 0;
    if (cell->point_inside(p)){
        outcome = 1;
    }
    else{
        bool cell_found = false;
        // if the point is outside of the cell then search all neighbors until the point is found
        // or until we have search enough neighbors to make sure that the point is actually outside of the domain
        std::vector<typename DoFHandler<dim>::active_cell_iterator> tested_cells;
        std::vector<typename DoFHandler<dim>::active_cell_iterator> adjacent_cells;
        std::vector<Point<dim>> cells_checked;
        typename DoFHandler<dim>::active_cell_iterator neighbor_child;
        tested_cells.push_back(cell);
        cells_checked.push_back(cell->center());
        int nSearch = 0;
        while (nSearch < param.search_iter){
            for (unsigned int i = 0; i < tested_cells.size(); ++i){
                // for each face of the tested cell check its neighbors
                for (unsigned int j = 0; j < GeometryInfo<dim>::faces_per_cell; ++j){
                    if (tested_cells[i]->at_boundary(j))
                        continue;
                    if(tested_cells[i]->neighbor(j)->active()){
                        // if the neighbor is active then we check first if we have already checked it
                        bool cell_checked = false;
                        for (unsigned int ii = 0; ii < cells_checked.size(); ++ii){
                            double dst = cells_checked[ii].distance(tested_cells[i]->neighbor(j)->center());
                            if (dst < 0.001){
                                cell_checked = true;
                                break;
                            }
                        }
                        if (!cell_checked){
                            // none of the cell centers of the checked cells is almost identical
                            // to this neighbor so we will add it to the list of the cells to be check
                            adjacent_cells.push_back(tested_cells[i]->neighbor(j));
                        }

                    }
                    else{
                        // if the neighbor cell is not active then it has children
                        // Here we check those children that touch the face of the tested cell
                        for (unsigned int ichild = 0; ichild < tested_cells[i]->neighbor(j)->n_children(); ++ ichild){
                            // if the child has children, this child doesnt not touch the test cell and we will examine it later
                            if(!tested_cells[i]->neighbor(j)->child(ichild)->active())
                                continue;
                            neighbor_child = tested_cells[i]->neighbor(j)->child(ichild);
                            // Check if this cell has been tested
                            bool cell_checked = false;
                            for (unsigned int ii = 0; ii < cells_checked.size(); ++ii){
                                double dst = cells_checked[ii].distance(neighbor_child->center());
                                if (dst < 0.001){
                                    cell_checked = true;
                                    break;
                                }
                            }
                            if (!cell_checked){
                                adjacent_cells.push_back(neighbor_child);
                            }
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
                    print_Cell_var(adjacent_cells[i],cell_type(adjacent_cells[i]));
                    if (cell->is_locally_owned()){
                        cell = adjacent_cells[i];
                        //plot_cell(cell);
                        outcome = 2;
                    }
                    else if(cell->is_ghost()){
                        cell = adjacent_cells[i];
                        //plot_cell(cell);
                        outcome = 3;
                    }
                    else if(cell->is_artificial()){
                        outcome = -3;
                    }
                    else
                        std::cerr << "That cant be right. The cell must be either local, ghost or artificial" << std::endl;
                    break;
                }
                else{
                    tested_cells.push_back(adjacent_cells[i]);
                    cells_checked.push_back(adjacent_cells[i]->center());
                }
            }
            if (cell_found){
                break;
            }
            else{
                nSearch++;
                adjacent_cells.clear();
            }
        }
    }
    return outcome;
}

template <int dim>
int Particle_Tracking<dim>::compute_point_velocity(Point<dim>& p, Point<dim>& v, typename DoFHandler<dim>::active_cell_iterator &cell, int check_point_status){
    int outcome = -101;
    if (check_point_status < 0 || cell->is_artificial()){
        std::cerr << "You should not call compute_point_velocity if the check_point_status is negative which means the cell is artificial" << std::endl;
        outcome = -99;
        return outcome;
    }

    Point<dim> p_unit;
    const MappingQ1<dim> mapping;
    bool success = try_mapping(p, p_unit, cell, mapping);
    if (!success){
        std::cerr << "P fail:" << p << std::endl;
        outcome = -88;
        return outcome;
    }


    if (check_point_status == 0){
        // no new cell has been found the particle possible has left the domain for ever
        if (p_unit[dim-1] > 1){ // the particle exits from the top face which is what we want
            outcome = 1;
            return outcome;
        }
        else if (p_unit[dim-1] < 0){ // the particle exits from the bottom face (BAD BAD BAD!!!)
            outcome = -9;
            return outcome;
        }
        else if(p_unit[0] < 0 || p_unit[0] > 1){ // the particle exits from either side in x direction (not ideal but its ok)
            outcome = 2;
            return outcome;
        }
        else if (dim == 3){
            if (p_unit[1] < 0 || p_unit[1] > 1){ // same as above
               outcome = 2;
               return outcome;
            }
        }

        // sometimes it may be possible that the check_cell_point fails to identify that the point lies in the cell therefore we allow for
        // one last chance to change the point status
        if (dim == 2){
            if (p_unit[0]>=0 && p_unit[0] <=1 &&
                p_unit[1]>=0 && p_unit[1] <=1){
                check_point_status = 1;
            }
        }
        else if (dim == 3){
            if (p_unit[0]>=0 && p_unit[0] <=1 &&
                p_unit[1]>=0 && p_unit[1] <=1 &&
                    p_unit[2]>=0 && p_unit[2] <=1){
                check_point_status = 1;
            }
        }
    }

    if (check_point_status > 0){
        //The velocity is equal vx = - Kx*dHx/n
        // dHi is computed as dN1i*H1i + dN2i*H2i + ... + dNni*Hni, where n is dofs_per_cell and i=[x y z]
        // For the current cell we will extract the hydraulic head solution
        const unsigned int dofs_per_cell = fe.dofs_per_cell;
        QTrapez<dim> trapez_formula;
        FEValues<dim> fe_values_trapez(fe, trapez_formula, update_values);
        fe_values_trapez.reinit(cell);
        std::vector<double> current_cell_head(dofs_per_cell);
        fe_values_trapez.get_function_values(locally_relevant_solution, current_cell_head);

        // To compute the head derivatives at the current particle position
        // we set a quadrature rule at the current point
        Quadrature<dim> temp_quadrature(p_unit);
        FEValues<dim> fe_values_temp(fe, temp_quadrature, update_gradients);
        fe_values_temp.reinit(cell);

        // dHead is the head gradient and is initialized to zero
        Tensor<1,dim> dHead;
        for (int i = 0; i < dim; ++i){
            dHead[i] = 0;
        }

        for (unsigned int i = 0; i < dofs_per_cell; ++i){
            Tensor<1,dim> dN = fe_values_temp.shape_grad(i,0);
            for (int i_dim = 0; i_dim < dim; ++i_dim){
                dHead[i_dim] = dHead[i_dim] + dN[i_dim]*current_cell_head[i];
            }
        }

        // divide dHead with the porosity
        double por = porosity.value(p);
        for (int i_dim = 0; i_dim < dim; ++i_dim)
            dHead[i_dim] = dHead[i_dim]/por;
        Tensor<1,dim> temp_v = HK_function.value(p)*dHead;
        for (int i_dim = 0; i_dim < dim; ++i_dim)
            v[i_dim] = temp_v[i_dim];
        return 0;
    }
    return outcome;
}

template <int dim>
int Particle_Tracking<dim>::compute_point_velocity(Point<dim>& p, Point<dim>& v, typename DoFHandler<dim>::active_cell_iterator& cell){
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
            int nSearch = 0;
            while (nSearch < param.search_iter){
                for (unsigned int i = 0; i < tested_cells.size(); ++i){
                    // for each face of the tested cell check its neighbors
                    for (unsigned int j = 0; j < GeometryInfo<dim>::faces_per_cell; ++j){
                        if (tested_cells[i]->at_boundary(j))
                            continue;
                        if(!tested_cells[i]->neighbor(j)->active()){
                            // if the neighbor cell is not active then it has children
                            // Here we check those children that touch the face of the tested cell
                            for (unsigned int ichild = 0; ichild < tested_cells[i]->neighbor(j)->n_children(); ++ ichild){
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
                                if (dst > 0.001){// none of the cell centers of the checked cells is almost identical
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
                            if (dst > 0.001){// none  of the cell centers of the checked cells is almost identical
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

    // We compute the unit coordinates using the new cell. This call is unessecary if the previous try mapping was success
    // and the point was found within this cell
    success = try_mapping(p, p_unit, cell, mapping);

    if(!success){
        std::cerr << "P fail:" << p << std::endl;
        outcome = -88;
        return outcome;
    }

    if (!cell_found){
        if (p_unit[dim-1] > 1) // the particle exits from the top face which is what we want
            outcome = 1;
        else if (p_unit[dim-1] < 0) // the particle exits from the bottom face (BAD BAD BAD!!!)
            outcome = -9;
        else if(p_unit[0] < 0 || p_unit[0] > 1) // the particle exits from either side in x direction (not ideal but its ok)
            outcome = 2;
        else if (dim == 3){
            if (p_unit[1] < 0 || p_unit[1] > 1) // same as above
               outcome = 2;
        }

        if (dim == 2){
            if (p_unit[0]>=0 && p_unit[0] <=1 &&
                p_unit[1]>=0 && p_unit[1] <=1){
                cell_found = true;
            }
        }
        else if (dim == 3){
            if (p_unit[0]>=0 && p_unit[0] <=1 &&
                p_unit[1]>=0 && p_unit[1] <=1 &&
                    p_unit[2]>=0 && p_unit[2] <=1){
                cell_found = true;
            }
        }
    }

    if (cell_found){
        if (cell->is_ghost() || cell->is_artificial()){
            outcome = 55;
        }
        else{
            //The velocity is equal vx = - Kx*dHx/n
            // dHi is computed as dN1i*H1i + dN2i*H2i + ... + dNni*Hni, where n is dofs_per_cell and i=[x y z]
            // For the current cell we will extract the hydraulic head solution
            QTrapez<dim> trapez_formula;
            FEValues<dim> fe_values_trapez(fe, trapez_formula, update_values);
            fe_values_trapez.reinit(cell);
            std::vector<double> current_cell_head(dofs_per_cell);
            fe_values_trapez.get_function_values(locally_relevant_solution, current_cell_head);

            // To compute the head derivatives at the current particle position
            // we set a quadrature rule at the current point
            Quadrature<dim> temp_quadrature(p_unit);
            FEValues<dim> fe_values_temp(fe, temp_quadrature, update_gradients);
            fe_values_temp.reinit(cell);

            // dHead is the head gradient and is initialized to zero
            Tensor<1,dim> dHead;
            for (int i = 0; i < dim; ++i){
                dHead[i] = 0;
            }

            for (unsigned int i = 0; i < dofs_per_cell; ++i){
                Tensor<1,dim> dN = fe_values_temp.shape_grad(i,0);
                for (int i_dim = 0; i_dim < dim; ++i_dim){
                    dHead[i_dim] = dHead[i_dim] + dN[i_dim]*current_cell_head[i];
                }
            }

            // divide dHead with the porosity
            double por = porosity.value(p);
            for (int i_dim = 0; i_dim < dim; ++i_dim)
                dHead[i_dim] = dHead[i_dim]/por;
            Tensor<1,dim> temp_v = HK_function.value(p)*dHead;
            for (int i_dim = 0; i_dim < dim; ++i_dim)
                v[i_dim] = temp_v[i_dim];
        }
    }
    return outcome;
}


template <int dim>
int Particle_Tracking<dim>::find_next_point(Streamline<dim> &streamline, typename DoFHandler<dim>::active_cell_iterator &cell){
    int outcome = -9999;
    int last = streamline.P.size()-1; // this is the index of the last point in the streamline
    double step_lenght = time_step_multiplier(cell) *(cell->minimum_vertex_distance()/param.step_size);
    double step_time;
    Point<dim> next_point;
    Point<dim> temp_velocity;
    typename DoFHandler<dim>::active_cell_iterator init_cell = cell;

    if (param.method == 1){
        // Euler method is the simplest one. The next point is computed as function of the previous
        // point only.
        step_time = step_lenght/streamline.V[last].norm();
        for (int i = 0; i < dim; ++i)
            next_point[i] = streamline.P[last][i] + streamline.V[last][i]*step_time;
        // before we consider the new point as part of the streamline we have to make sure that it is
        // located in the domain. To do so we will compute the velocity of the new point.
        outcome = compute_point_velocity(next_point, temp_velocity, cell);
    }
    else if (param.method == 2){
        // In second order Runge Kutta we compute two points and average
        // their velocities to obtain the final position
        Point<dim> temp_point;
        // First take a step based on the velocity of the previous point
        step_time = step_lenght/streamline.V[last].norm();
        for (int i = 0; i < dim; ++i)
            temp_point[i] = streamline.P[last][i] + streamline.V[last][i]*step_time;
        // compute the velocity of this point
        outcome = compute_point_velocity(temp_point, temp_velocity, cell);
        if (outcome == 0){
            // the final point will be computed as the average velocities
            Point<dim> av_vel;
            for (int i = 0; i < dim; ++i){
                av_vel[i] = 0.5*streamline.V[last][i] + 0.5*temp_velocity[i];
            }
            step_time = step_lenght/av_vel.norm();
            for (int i = 0; i < dim; ++i){
                next_point[i] = streamline.P[last][i] + av_vel[i]*step_time;
            }
            // Compute the velocity of the final point
            outcome = compute_point_velocity(next_point, temp_velocity, cell);
        }
    }
    else if (param.method == 3){
        // Fourth order Runge Kutta computes the next point by averaging 4 sampling points
        // The weights of RK4 are
        std::vector<double> RK_weights(4,1);
        RK_weights[1] = 2; RK_weights[2] = 2;
        std::vector<Point<dim>> RK_steps;

        // The first step uses the velocity of the previous step
        RK_steps.push_back(streamline.V[last]);
        Point<dim> temp_point;
        int count_nest = 0;
        // First we compute a point by taking half step using the initial velocity
        int out = take_euler_step(cell, 0.5, step_lenght, streamline.P[last], RK_steps[0], temp_point, temp_velocity, count_nest);
        if (out != 0){
            return add_streamline_point(cell, streamline, temp_point, temp_velocity, out);
        }
        else{
            RK_steps.push_back(temp_velocity);
             //using the velocity of the mid point take another half step from the initial point
            out = take_euler_step(cell, 0.5, step_lenght, streamline.P[last], RK_steps[1], temp_point, temp_velocity, count_nest);
            if (out !=0 ){
                return add_streamline_point(cell, streamline, temp_point, temp_velocity, out);
            }
            else{
                RK_steps.push_back(temp_velocity);
                // using the velocity of the second point take a full step
                out = take_euler_step(cell, 1.0, step_lenght, streamline.P[last], RK_steps[2], temp_point, temp_velocity, count_nest);
                if (out !=0){
                    return add_streamline_point(cell, streamline, temp_point, temp_velocity, out);
                }
                else{
                    // Finally we average the velocities and take a full step
                    Point<dim> av_vel;
                    for (unsigned int i = 0; i < RK_steps.size(); ++i){
                        for (unsigned int j = 0; j < dim; ++j){
                            av_vel[j] += RK_steps[i][j]*(RK_weights[i]/6.0);
                        }
                    }
                    out = take_euler_step(cell, 1.0, step_lenght, streamline.P[last], av_vel, temp_point, temp_velocity, count_nest);
                    return add_streamline_point(cell, streamline, temp_point, temp_velocity, out);
                }
            }
        }
    }
}

template <int dim>
void Particle_Tracking<dim>::Send_receive_particles(std::vector<Streamline<dim>>    new_particles,
                                                    std::vector<Streamline<dim>>	&streamlines){
    unsigned int my_rank = Utilities::MPI::this_mpi_process(mpi_communicator);
    unsigned int n_proc = Utilities::MPI::n_mpi_processes(mpi_communicator);

    streamlines.clear();
    std::vector<std::vector<double> > px(n_proc);
    std::vector<std::vector<double> > py(n_proc);
    std::vector<std::vector<double> > pz(n_proc);
    std::vector<std::vector<int> > E_id(n_proc);
    std::vector<std::vector<int> > S_id(n_proc);
    std::vector<std::vector<int> > proc_id(n_proc);
    std::vector<std::vector<int> > p_id(n_proc);
    std::vector<std::vector<double> > BBlx(n_proc);
    std::vector<std::vector<double> > BBly(n_proc);
    std::vector<std::vector<double> > BBlz(n_proc);
    std::vector<std::vector<double> > BBux(n_proc);
    std::vector<std::vector<double> > BBuy(n_proc);
    std::vector<std::vector<double> > BBuz(n_proc);

    // copy the data
    for (unsigned int i = 0; i < new_particles.size(); ++i){
        px[my_rank].push_back(new_particles[i].P[0][0]);
        py[my_rank].push_back(new_particles[i].P[0][1]);
        if (dim == 3)
            pz[my_rank].push_back(new_particles[i].P[0][2]);
        E_id[my_rank].push_back(new_particles[i].E_id);
        S_id[my_rank].push_back(new_particles[i].S_id);
        proc_id[my_rank].push_back(new_particles[i].proc_id);
        p_id[my_rank].push_back(new_particles[i].p_id[0]);
        BBlx[my_rank].push_back(new_particles[i].BBl[0]);
        BBly[my_rank].push_back(new_particles[i].BBl[1]);
        if (dim == 3)
            BBlz[my_rank].push_back(new_particles[i].BBl[2]);
        BBux[my_rank].push_back(new_particles[i].BBu[0]);
        BBuy[my_rank].push_back(new_particles[i].BBu[1]);
        if (dim == 3)
            BBuz[my_rank].push_back(new_particles[i].BBu[2]);
    }
    MPI_Barrier(mpi_communicator);
    // Send everything to every processor

    std::vector<int> data_per_proc;
    Send_receive_size(px[my_rank].size(), n_proc, data_per_proc, mpi_communicator);
    Sent_receive_data<double>(px, data_per_proc, my_rank, mpi_communicator, MPI_DOUBLE);
    Sent_receive_data<double>(py, data_per_proc, my_rank, mpi_communicator, MPI_DOUBLE);
    if (dim == 3)
        Sent_receive_data<double>(pz, data_per_proc, my_rank, mpi_communicator, MPI_DOUBLE);
    Sent_receive_data<int>(E_id, data_per_proc, my_rank, mpi_communicator, MPI_INT);
    Sent_receive_data<int>(S_id, data_per_proc, my_rank, mpi_communicator, MPI_INT);
    Sent_receive_data<int>(proc_id, data_per_proc, my_rank, mpi_communicator, MPI_INT);
    Sent_receive_data<int>(p_id, data_per_proc, my_rank, mpi_communicator, MPI_INT);
    Sent_receive_data<double>(BBlx, data_per_proc, my_rank, mpi_communicator, MPI_DOUBLE);
    Sent_receive_data<double>(BBly, data_per_proc, my_rank, mpi_communicator, MPI_DOUBLE);
    if (dim == 3)
        Sent_receive_data<double>(BBlz, data_per_proc, my_rank, mpi_communicator, MPI_DOUBLE);
    Sent_receive_data<double>(BBux, data_per_proc, my_rank, mpi_communicator, MPI_DOUBLE);
    Sent_receive_data<double>(BBuy, data_per_proc, my_rank, mpi_communicator, MPI_DOUBLE);
    if (dim == 3)
        Sent_receive_data<double>(BBuz, data_per_proc, my_rank, mpi_communicator, MPI_DOUBLE);

    // now we loop through the data and each processor will pick the ones that are found on its cells
    for (unsigned int i = 0; i < n_proc; ++i){
        if (i == my_rank)
            continue;
        for (unsigned int j = 0; j < px[i].size(); ++j){
            if (proc_id[i][j] == static_cast<int>(my_rank)){
                Point<dim> p;
                p[0] = px[i][j];
                p[1] = py[i][j];
                if (dim == 3)
                    p[2] = pz[i][j];

                Streamline<dim> temp(E_id[i][j], S_id[i][j], p);
                p[0] = BBlx[i][j];
                p[1] = BBly[i][j];
                if (dim == 3)
                    p[2] = BBlz[i][j];
                temp.BBl = p;
                p[0] = BBux[i][j];
                p[1] = BBuy[i][j];
                if (dim == 3)
                    p[2] = BBuz[i][j];
                temp.BBu = p;
                temp.p_id[0] = p_id[i][j];
                streamlines.push_back(temp);
            }
        }
    }
    MPI_Barrier(mpi_communicator);
}

template <int dim>
double Particle_Tracking<dim>::calculate_step(typename DoFHandler<dim>::active_cell_iterator cell, Point<dim> Vel){
    double xmin, ymin, zmin;
    xmin = 10000000;
    ymin = 10000000;
    zmin = 10000000;
    if (dim == 2){
        // x direction
        double dst = cell->vertex(0).distance(cell->vertex(1));
        if (dst < xmin) xmin = dst;
        dst = cell->vertex(2).distance(cell->vertex(3));
        if (dst < xmin) xmin = dst;

        dst = cell->vertex(0).distance(cell->vertex(2));
        if (dst < zmin) zmin = dst;
        dst = cell->vertex(1).distance(cell->vertex(3));
        if (dst < zmin) zmin = dst;
        double Vn = Vel.norm_sqr();
        Vel[0] = Vel[0]/Vn;
        Vel[1] = Vel[1]/Vn;
        return xmin*Vel[0] + zmin*Vel[1];

    }

}

template <int dim>
int Particle_Tracking<dim>::add_streamline_point(typename DoFHandler<dim>::active_cell_iterator &cell,
                                                 Streamline<dim> &streamline,
                                                 Point<dim> p, Point<dim> vel,
                                                 int return_value){

    if (bprint_DBG){
        print_point_var(p,return_value);
    }

    if (return_value == 0 || return_value == -1){ // Either the computation has been normal or with reduced step
        if (cell->is_ghost() || cell->is_artificial()){
            streamline.add_point(p, cell->subdomain_id());
            return 55;
        }
        else if (cell->is_locally_owned()){
            //plot_segment(streamline.P[streamline.P.size()-1], p);
            streamline.add_point_vel(p, vel, cell->subdomain_id());
            return 0;
        }
    }
    else{
        return return_value;
    }
}

template <int dim>
int Particle_Tracking<dim>::take_euler_step(typename DoFHandler<dim>::active_cell_iterator &cell,
                                            double step_weight, double step_length,
                                            Point<dim> P_prev, Point<dim> V_prev,
                                            Point<dim>& P_next, Point<dim>& V_next, int& count_nest){
    double step_time = step_weight * step_length / V_prev.norm();
    int outcome;

    for (int i = 0; i < dim; ++i){
        P_next[i] = P_prev[i] + V_prev[i] * step_time;
    }
    //plot_point(P_next);

    int check_pnt = check_cell_point(cell, P_next);
    if (check_pnt < 0){
        step_length = step_length/5.0;
        count_nest++;
        outcome = take_euler_step(cell, step_weight, step_length, P_prev, V_prev, P_next, V_next, count_nest);
    }
    else{
        outcome = compute_point_velocity(P_next, V_next, cell, check_pnt);
    }

    if (count_nest == 0)
        return outcome;
    else{
        if (outcome == 0)
            return -1;
        else
            return outcome;

    }
}

template <int dim>
void Particle_Tracking<dim>::plot_cell(typename DoFHandler<dim>::active_cell_iterator cell){
    std::vector<Point<dim>> verts;
    for (unsigned int vertex_no = 0; vertex_no < GeometryInfo<dim>::vertices_per_cell; ++vertex_no){
        verts.push_back(cell->vertex(vertex_no));
    }
    if (dim == 3){
        std::cout << "plot3([" << verts[0][0] << " "
                               << verts[1][0] << " "
                               << verts[3][0] << " "
                               << verts[2][0] << " "
                               << verts[0][0] << "],["
                               << verts[0][1] << " "
                               << verts[1][1] << " "
                               << verts[3][1] << " "
                               << verts[2][1] << " "
                               << verts[0][1] << "],["
                               << verts[0][2] << " "
                               << verts[1][2] << " "
                               << verts[3][2] << " "
                               << verts[2][2] << " "
                               << verts[0][2] << "])" << std::endl;

        std::cout << "plot3([" << verts[4][0] << " "
                               << verts[5][0] << " "
                               << verts[7][0] << " "
                               << verts[6][0] << " "
                               << verts[4][0] << "],["
                               << verts[4][1] << " "
                               << verts[5][1] << " "
                               << verts[7][1] << " "
                               << verts[6][1] << " "
                               << verts[4][1] << "],["
                               << verts[4][2] << " "
                               << verts[5][2] << " "
                               << verts[7][2] << " "
                               << verts[6][2] << " "
                               << verts[4][2] << "])" << std::endl;

        std::cout << "plot3([" << verts[0][0] << " "
                               << verts[4][0] << "],["
                               << verts[0][1] << " "
                               << verts[4][1] << "],["
                               << verts[0][2] << " "
                               << verts[4][2] << "])" << std::endl;

        std::cout << "plot3([" << verts[1][0] << " "
                               << verts[5][0] << "],["
                               << verts[1][1] << " "
                               << verts[5][1] << "],["
                               << verts[1][2] << " "
                               << verts[5][2] << "])" << std::endl;

        std::cout << "plot3([" << verts[3][0] << " "
                               << verts[7][0] << "],["
                               << verts[3][1] << " "
                               << verts[7][1] << "],["
                               << verts[3][2] << " "
                               << verts[7][2] << "])" << std::endl;

        std::cout << "plot3([" << verts[2][0] << " "
                               << verts[6][0] << "],["
                               << verts[2][1] << " "
                               << verts[6][1] << "],["
                               << verts[2][2] << " "
                               << verts[6][2] << "])" << std::endl;

    }
    else if (dim == 2){
        std::cout << "plot3([" << verts[0][0] << " "
                               << verts[1][0] << " "
                               << verts[3][0] << " "
                               << verts[2][0] << " "
                               << verts[0][0] << "],["
                               << verts[0][1] << " "
                               << verts[1][1] << " "
                               << verts[3][1] << " "
                               << verts[2][1] << " "
                               << verts[0][1] << "])" << std::endl;
    }
}

template <int dim>
void Particle_Tracking<dim>::plot_point(Point<dim> p){
    if (dim == 3)
        std::cout << "plot3(" << p[0] << ", " << p[1] << ", " << p[2] << ", 'x')" << std::endl;
    else if (dim == 2)
        std::cout << "plot(" << p[0] << ", " << p[1] << ", x')" << std::endl;
}

template <int dim>
void Particle_Tracking<dim>::plot_segment(Point<dim> A, Point<dim> B){
    if (dim == 3){
        std::cout << "plot3([" << A[0] << " "
                               << B[0] << "],["
                               << A[1] << " "
                               << B[1] << "],["
                               << A[2] << " "
                               << B[2] << "], 'o-r')" << std::endl;

    }
    else if (dim == 2){
        std::cout << "plot([" << A[0] << " "
                              << B[0] << "],["
                              << A[1] << " "
                              << B[1] << "], 'o-r')" << std::endl;

    }
}

template <int dim>
void Particle_Tracking<dim>::print_Cell_var(typename DoFHandler<dim>::active_cell_iterator cell, int cell_type){
    std::vector<Point<dim>> verts;
    for (unsigned int vertex_no = 0; vertex_no < GeometryInfo<dim>::vertices_per_cell; ++vertex_no){
        verts.push_back(cell->vertex(vertex_no));
    }

    if (dim == 3){
        dbg_file << "S(" << dbg_i_strm << ",1).Cells(" << dbg_i_step << ",1).Nodes = ["
                 << verts[0][0] << " " << verts[0][1] << " " << verts[0][2] << ";"
                 << verts[1][0] << " " << verts[1][1] << " " << verts[1][2] << ";"
                 << verts[3][0] << " " << verts[3][1] << " " << verts[3][2] << ";"
                 << verts[2][0] << " " << verts[2][1] << " " << verts[2][2] << ";"
                 << verts[4][0] << " " << verts[4][1] << " " << verts[4][2] << ";"
                 << verts[5][0] << " " << verts[5][1] << " " << verts[5][2] << ";"
                 << verts[7][0] << " " << verts[7][1] << " " << verts[7][2] << ";"
                 << verts[6][0] << " " << verts[6][1] << " " << verts[6][2] << "];" << std::endl;
    }
    else if (dim == 2) {
        dbg_file << "S(" << dbg_i_strm << ",1).Cells(" << dbg_i_step << ",1).nodes = ["
                 << verts[0][0] << " " << verts[0][1] << " " << verts[0][2] << ";"
                 << verts[1][0] << " " << verts[1][1] << " " << verts[1][2] << ";"
                 << verts[3][0] << " " << verts[3][1] << " " << verts[3][2] << ";"
                 << verts[2][0] << " " << verts[2][1] << " " << verts[2][2] << "];" << std::endl;
    }

    dbg_file << "S(" << dbg_i_strm << ",1).Cells(" << dbg_i_step << ",1).Type = " << cell_type << ";" << std::endl;

}

template <int dim>
void Particle_Tracking<dim>::print_point_var(Point<dim> p, int r){
    if (dim == 3)
        dbg_file << "S(" << dbg_i_strm << ",1).P(" << dbg_i_step << ",1).XYZ = ["
                 << p[0] << " " << p[1] << " " << p[2] <<"];" << std::endl;
    else if (dim == 2)
        dbg_file << "S(" << dbg_i_strm << ",1).P(" << dbg_i_step << ",:) = ["
                 << p[0] << " " << p[1] << " " << p[2] <<"];" << std::endl;
    dbg_file << "S(" << dbg_i_strm << ",1).P(" << dbg_i_step << ",1).type = " << r << ";" << std::endl;
    dbg_i_step++;
}

template <int dim>
void Particle_Tracking<dim>::print_strm_exit_info(int r){
    dbg_file << "S(" << dbg_i_strm << ",1).Exit =" << r << ";" << std::endl;
    dbg_i_strm++;
    dbg_i_step = 1;
    dbg_file << "%-----------------------------------------------------------------------" << std::endl;
}

template <int dim>
int Particle_Tracking<dim>::cell_type(typename DoFHandler<dim>::active_cell_iterator cell){
    if (cell->is_locally_owned())
        return 0;
    else if (cell->is_ghost())
        return 1;
    else if (cell->is_artificial())
        return 2;
}

template <int dim>
double Particle_Tracking<dim>::time_step_multiplier(typename DoFHandler<dim>::active_cell_iterator cell){
    double time_step = 1.0;
    if (!cell->is_locally_owned())
        time_step = 0.5;

    for (unsigned int j = 0; j < GeometryInfo<dim>::faces_per_cell; ++j){
        if (cell->at_boundary(j)){
            if (time_step > 0.1)
                time_step = 0.1;
        }
        else{
            if (cell->neighbor(j)->active()){
                if (!cell->neighbor(j)->is_locally_owned()){
                    if (time_step > 0.25)
                        time_step = 0.25;
                }
                for (unsigned int k = 0; k < GeometryInfo<dim>::faces_per_cell; ++k){
                    if(cell->neighbor(j)->at_boundary(k))
                        if (time_step > 0.5)
                            time_step = 0.5;
                }
            }
            else{
                for (unsigned int ichild = 0; ichild < cell->neighbor(j)->n_children(); ++ ichild){
                    if(cell->neighbor(j)->child(ichild)->active()){
                        if (!cell->neighbor(j)->child(ichild)->is_locally_owned())
                            if (time_step > 0.25)
                                time_step = 0.25;

                        for (unsigned int k = 0; k < GeometryInfo<dim>::faces_per_cell; ++k){
                            if(cell->neighbor(j)->child(ichild)->at_boundary(k))
                                if (time_step > 0.5)
                                    time_step = 0.5;
                        }
                    }
                }
            }
        }
    }
    return time_step;
}

#endif // PARTICLE_TRACKING_H
