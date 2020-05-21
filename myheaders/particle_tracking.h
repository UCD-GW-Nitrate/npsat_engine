#ifndef PARTICLE_TRACKING_H
#define PARTICLE_TRACKING_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/bounding_box.h>

#include "my_functions.h"
#include "dsimstructs.h"
#include "streamlines.h"
#include "cgal_functions.h"
#include "mpi_help.h"
//#include "velocity_rt0.h"


using namespace dealii;

enum class ParticleExit{
    NO_EXIT, /**< THe particle should not exit */
    TOP, /**< The particle exits from the top */
    BOT, /**< The particle exits from the bottom */
    SIDEX, /**< The particle exits from the side */
    SIDEY, /**< The particle exits from the side */
    CHANGE_PROC, /**< The particle should continue to another processor */
    MAX_STEPS, /**< Reach the maximum number of steps */
    STUCK, /**< The particle stuck for a given number of iterations */
    FAIL, /**< Transformation failed */
    GHOST, /**< Attempt to compute velocity on artificial cell */
    INIT_OUT, /**< The first point is outside the cell */
    DOF_NOT_FOUND, /**< A dof could not be found in the velocity map */
    UNKNOWN /**< The particle exits from the top */
};

/*!
 * \brief The AverageVel class is a container to hold the averaged values of a vector type (Velocity head gradient)
 */
template <int dim>
class AverageVel{
public:
    /*!
     * \brief AverageVel One if the constructors of this class
     * \param local If the dof is local this should be true otherwise false
     * \param dof_in THe degree of freedom of this velocity
     * \param v The vector values associated with the input dof
     * \param c A list of dofs that this dof is constraint to. If the list is empty or containts only the #dof_in then this dof is not constraint.
     */
    AverageVel(bool local, int dof_in, Point<dim> v, std::vector<types::global_dof_index> c);

    /*!
     * \brief AverageVel constructor is used when we already know the average vector values for this dof.
     * This is possible when the dof is not local and comes averaged from another processor
     * \param dof_in
     * \param av_vel_in
     */
    AverageVel(int dof_in, Point<dim> av_vel_in);

    /*!
     * \brief Addvelocity adds a vector value to the dof.
     * \param v the vector value which can be velocity or head gradient
     */
    void Addvelocity(Point<dim> v);

    //! A vector with the dofs that the #dof is constraint to
    std::vector<types::global_dof_index> cnstr;

    //! A vector with the velocities to be averaged. The size of this vector is equal to the cells that have this dof as vertex
    std::vector<Point<dim>> V;

    //! A flag indicating whether this dof is local to the current processor
    bool is_local;

    //! The dof
    int dof;

    //! The avereged velocity
    Point<dim> av_vel;

    //! A flag indicating whether this dof has been averaged or not. #av_vel should be used only if
    //! #is_averaged is true.
    bool is_averaged;
};

template <int dim>
AverageVel<dim>::AverageVel(bool local, int dof_in, Point<dim> v, std::vector<types::global_dof_index> c){
    is_local = local;
    dof = dof_in;
    V.push_back(v);
    for (unsigned int ii = 0; ii < c.size(); ++ii){
        if (static_cast<int>(c[ii]) != dof)
            cnstr.push_back(c[ii]);
    }
    is_averaged = false;
    for (unsigned int idim = 0; idim < dim; ++idim)
        av_vel[idim] = -9999.9;
}

template <int dim>
AverageVel<dim>::AverageVel(int dof_in, Point<dim> av_vel_in){
    dof = dof_in;
    av_vel = av_vel_in;
    is_averaged = true;
    is_local = false;
    cnstr.clear();
    V.clear();
}

template <int dim>
void AverageVel<dim>::Addvelocity(Point<dim> v){
    V.push_back(v);
}


template <int dim>
class Particle_Tracking{
public:
    Particle_Tracking(MPI_Comm& mpi_communicator_in,
                      DoFHandler<dim>& dof_handler_in,
                      FE_Q<dim>& fe_in,
                      ConstraintMatrix& constraints_in,
                      TrilinosWrappers::MPI::Vector& 	locally_relevant_solution_in,
                      TrilinosWrappers::MPI::Vector& 	system_rhs_in,
                      MyTensorFunction<dim>& HK_function_in,
                      MyFunction<dim, dim>& porosity_in,
                      ParticleParameters& param_in);


    void trace_particles(std::vector<Streamline<dim>>& streamlines, int iter);
    void print_all_cell_velocity();

    bool average_velocity_field();
    bool average_velocity_field1();

private:
    MPI_Comm                            mpi_communicator;
    DoFHandler<dim>&                    dof_handler;
    FE_Q<dim>&                          fe;
    ConstraintMatrix&                   Headconstraints;
    TrilinosWrappers::MPI::Vector       locally_relevant_solution;
    TrilinosWrappers::MPI::Vector       system_rhs;
    MyTensorFunction<dim>               HK_function;
    MyFunction<dim, dim>                porosity;
    ParticleParameters                  param;
    ConditionalOStream                  pcout;

    std::map<unsigned int, AverageVel<dim>> VelocityMap;

    bool                                bprint_DBG;
    std::ofstream                       dbg_file;
    std::ofstream                       dbg_cell_file;
    int                                 dbg_i_strm;
    int                                 dbg_i_step;
    int                                 dbg_curr_Eid;
    int                                 dbg_curr_Sid;
    int                                 dbg_my_rank;

    /**
     * @brief internal_backward_tracking
     * @param cell
     * @param streamline
     * @return
     * - -99 Exit because the number of steps have exceeded the maximum allowable number defined by the user
     * - -88 For some reason the starting point if the streamline was not found inside the cell.
     * - -66 The particle has stuck in the flow field. After a certain amount of steps the bounding box
     * of the streamline has not been expanded
     */
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
    int check_cell_point(typename DoFHandler<dim>::active_cell_iterator &cell, Point<dim> p);

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
    int compute_point_velocity(Point<dim>& p, Point<dim>& v, typename DoFHandler<dim>::active_cell_iterator &cell, int check_point_status, int caller);

    double calculate_step(typename DoFHandler<dim>::active_cell_iterator& cell, Point<dim>& Vel);

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

    /**
     * @brief take_euler_step Computes the position and the velocity if possible of the next point
     * @param cell is the Cell that the particle is currently moving. However if the particle has to move to
     * an adjacent cell then this method returns the new cell.
     * @param step_weight This is the weight for the step. In RK 4 for example the first trial point has weight 0.5 to take half step.
     * @param step_length The step length.
     * @param P_prev The coordinates of the current point
     * @param V_prev The Velocity of the current point. This have been calculated from the previous iteration
     * @param P_next The coordinates of the points after taking the step
     * @param V_next The velocity of the new point if its possible to calculate. Whether the velocity calculation was possible
     * is dictated by the return statement
     * @param count_nest This method can call it self if the step was too big so that it skipped two cells and found an artificial.
     * It calles it self by reducing the step to half.
     * @return This function returns an integer which can be any of the following
     *  - -1 The method has called it self one or more times. This is very unsusuall by the way.
     *  - The same list as the return codes of the #compute_point_velocity returns
     */
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

    bool cell_exists(std::vector<Point<dim>> PointList, Point<dim> test_point);

    void plot_cell(typename DoFHandler<dim>::active_cell_iterator cell);
    void plot_point(Point<dim> p);
    void plot_segment(Point<dim> A, Point<dim> B);
    void print_Cell_var(typename DoFHandler<dim>::active_cell_iterator cell, int cell_type);
    void print_point_var(Point<dim> p, int r);
    void print_strm_exit_info(int r, int Eid, int Sid);
    int cell_type(typename DoFHandler<dim>::active_cell_iterator cell);
    void print_cell_velocity(std::vector<Point<dim>> p);
    void calculate_cell_velocity(typename DoFHandler<dim>::active_cell_iterator& cell,
                                 std::vector<Point<dim>>& p,
                                 std::vector<Point<dim>>& vel);
    /**
     * @brief calc_vel_on_point Calculates the velocity on a point that is inside the cell.
     *
     * Note that this method assumes that the point exists inside the cell and doesnt do any check to
     * test if the point is inside.
     * @param cell
     * @param p
     * @param vel This is the returned velocity
     * @return Returns true if the velocity was successfull and false if not. This can be false if the
     *  mapping from real to unit point has failed
     */
    bool calc_vel_on_point(typename DoFHandler<dim>::active_cell_iterator& cell,
                           Point<dim> p,
                           Point<dim>& vel);

    bool calc_vel_on_point_n_info(typename DoFHandler<dim>::active_cell_iterator& cell,
                                  Point<dim> p,
                                  Point<dim>& vel,
                                  Point<dim>& dH,
                                  double& H);

    void printAveragedVelocityField(std::string filename);



};

template <int dim>
Particle_Tracking<dim>::Particle_Tracking(MPI_Comm& mpi_communicator_in,
                                          DoFHandler<dim>& dof_handler_in,
                                          FE_Q<dim> &fe_in,
                                          ConstraintMatrix& constraints_in,
                                          TrilinosWrappers::MPI::Vector& 	locally_relevant_solution_in,
                                          TrilinosWrappers::MPI::Vector& 	system_rhs_in,
                                          MyTensorFunction<dim>& HK_function_in,
                                          MyFunction<dim, dim>& porosity_in,
                                          ParticleParameters& param_in)
    :
    mpi_communicator(mpi_communicator_in),
    dof_handler(dof_handler_in),
    fe(fe_in),
    Headconstraints(constraints_in),
    locally_relevant_solution(locally_relevant_solution_in),
    system_rhs(system_rhs_in),
    HK_function(HK_function_in),
    porosity(porosity_in),
    param(param_in),
    pcout(std::cout,(Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
{
    bprint_DBG = false;
    if (bprint_DBG){
        dbg_i_step = 1;
        dbg_i_strm = 1;
    }
}

template <int dim>
void Particle_Tracking<dim>::trace_particles(std::vector<Streamline<dim>>& streamlines, int iter){
    unsigned int my_rank = Utilities::MPI::this_mpi_process(mpi_communicator);
    unsigned int n_proc = Utilities::MPI::n_mpi_processes(mpi_communicator);
    dbg_my_rank = my_rank;

    //This is the name file where all particle trajectories are written
    const std::string log_file_name = (param.particle_prefix +
                                       Utilities::int_to_string(static_cast<unsigned int>(iter), 4) +
                                       "_particles_"	+
                                       Utilities::int_to_string(my_rank, 4) +
                                       ".traj");

    //This is the name file where we print info of particles that terminate abnornamly
    const std::string err_file_name = (param.particle_prefix +
                                       Utilities::int_to_string(static_cast<unsigned int>(iter), 4) +
                                       "_particle_errors_"	+
                                       Utilities::int_to_string(my_rank, 4) +
                                       ".traj");

    if (bprint_DBG){
        // This is the name file where the particles in matlab code format will be saved
        const std::string dbg_file_name = (param.particle_prefix + "_" +
                                           Utilities::int_to_string(static_cast<unsigned int>(iter), 4) +
                                           "_particles_dbg_" +
                                           Utilities::int_to_string(my_rank, 4) +
                                           ".m");
        dbg_file.open(dbg_file_name.c_str());

        /*
        const std::string dbg_cell_file_name = (prefix + "_" +
                                                Utilities::int_to_string(static_cast<unsigned int>(iter), 4) +
                                                "_cellInfo_dbg_" +
                                                Utilities::int_to_string(my_rank, 4) +
                                                ".txt");
        dbg_cell_file.open(dbg_cell_file_name);

        print_all_cell_velocity();
        */

    }



    std::ofstream log_file;
    std::ofstream err_file;
    log_file.open(log_file_name.c_str());
    err_file.open(err_file_name.c_str());
    std::vector<Streamline<dim>> new_particles;

    int trace_iter = 0;
    //int cnt_stuck_particles = 0;
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
        //int cnt_ptr = 0;int cnt_cells = 0;
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
        //std::cout<< "I'm proc" << my_rank << " and have " << new_particles.size() << " particles to send" << std::endl << std::flush;
        MPI_Barrier(mpi_communicator);

        std::vector<int> new_part_per_proc(n_proc);
        Send_receive_size(new_particles.size(), n_proc, new_part_per_proc, mpi_communicator);

        int max_N_part = 0;
        for (unsigned int i = 0; i < n_proc; ++i)
            max_N_part += new_part_per_proc[i];
        MPI_Barrier(mpi_communicator);
        pcout << "          Number of active particles: " << max_N_part << " --------" << std::endl << std::flush;
        //std::cout << my_rank << " : " << max_N_part << std::endl;

        //if (trace_iter>3)
        //    return;

        if (max_N_part == 0)
            break;


        Send_receive_particles(new_particles, streamlines);

        if (++trace_iter > param.Outmost_iter)
            break;
    }

    log_file.close();
    err_file.close();

    if (bprint_DBG){
        dbg_file.close();
        dbg_cell_file.close();
    }
}

template <int dim>
int Particle_Tracking<dim>::internal_backward_tracking(typename DoFHandler<dim>::active_cell_iterator cell, Streamline<dim>& streamline){
    dbg_curr_Eid = streamline.E_id;
    dbg_curr_Sid = streamline.S_id;
    //std::cout << "Eid: " << streamline.E_id << ", Sid: " << streamline.S_id << std::endl;

    // ++++++++++ CONVERT THIS TO ENUMERATION+++++++++++
    int reason_to_exit= -99; //UNKNOWN
    int cnt_iter = 0;
    if (bprint_DBG){
        print_Cell_var(cell, cell_type(cell));
        print_point_var(streamline.P[streamline.P.size()-1], 1000);
    }
    while(true){
        if (cnt_iter == 0){ // If this is the starting point of the streamline we need to compute the velocity
            int check_id = check_cell_point(cell, streamline.P[streamline.P.size()-1]);
            Point<dim> v;
            if (check_id == 1){
                reason_to_exit = compute_point_velocity(streamline.P[streamline.P.size()-1], v, cell, check_id, 0);
            }
            else{
                reason_to_exit = -88; //INIT_OUT
            }
            if (reason_to_exit != 0){
                if (bprint_DBG)
                    print_strm_exit_info(reason_to_exit, streamline.E_id, streamline.S_id);
                return  reason_to_exit;
            }
            else{
                streamline.V.push_back(v);
                //plot_cell(cell);
            }
        }
        // if this is not the starting point we already know the velocity at the current point
        // The following function returns both the position with velocity
        reason_to_exit = find_next_point(streamline, cell);
        if (streamline.times_not_expanded > param.Stuck_iter){
            reason_to_exit = -66; //STUCK
            if (bprint_DBG)
                print_strm_exit_info(reason_to_exit, streamline.E_id, streamline.S_id);
            return reason_to_exit;
        }

        if ( reason_to_exit != 0 )
            break;
        cnt_iter++;
        if (cnt_iter > param.streaml_iter){
            reason_to_exit = 777; //MAX_STEPS
        }
    }
    print_strm_exit_info(reason_to_exit, streamline.E_id, streamline.S_id);
    return  reason_to_exit;
}

template <int dim>
int Particle_Tracking<dim>::check_cell_point(typename DoFHandler<dim>::active_cell_iterator& cell, Point<dim> p){
    //Point<dim> testp(327.79903158960485,
    //                 859.7581620110702,
    //                 10.91061720969245);
    //bool dbg_pnt = false;
    //if (testp.distance(p) < 0.1){
    //    std::cout << "Debug this point" << std::endl;
    //    dbg_pnt = true;
    //}
    int outcome = 0;
    if (cell->point_inside(p)){
        outcome = 1;
        //if (dbg_pnt){
        //    std::cout << "Point found inside cell" << std::endl;
        //}
    }
    else{
        //if (dbg_pnt){
        //    std::cout << "Point is NOT found inside cell" << std::endl;
        //}
        bool cell_found = false;
        // if the point is outside of the cell then search all neighbors until the point is found
        // or until we have search enough neighbors to make sure that the point is actually outside of the domain
        std::vector<typename DoFHandler<dim>::active_cell_iterator> tested_cells;
        std::vector<typename DoFHandler<dim>::active_cell_iterator> adjacent_cells;
        std::vector<Point<dim>> cells_checked;
        std::vector<Point<dim>> cells2check;
        typename DoFHandler<dim>::active_cell_iterator neighbor_child;
        tested_cells.push_back(cell);
        cells_checked.push_back(cell->center());
        int nSearch = 0;
        while (nSearch < param.search_iter){
            for (unsigned int i = 0; i < tested_cells.size(); ++i){
                // for each face of the tested cell check its neighbors
                for (unsigned int j = 0; j < GeometryInfo<dim>::faces_per_cell; ++j){
                    if (!tested_cells[i]->at_boundary(j)){
                        if(tested_cells[i]->neighbor(j)->active()){
                            // if the neighbor is active then we check first if we have already checked it
                            // or have already marked it for check and then we add it accordingly
                            if (!cell_exists(cells_checked, tested_cells[i]->neighbor(j)->center()) &
                                    !cell_exists(cells2check, tested_cells[i]->neighbor(j)->center())){
                                if (!tested_cells[i]->neighbor(j)->is_artificial()){
                                    adjacent_cells.push_back(tested_cells[i]->neighbor(j));
                                    cells2check.push_back(tested_cells[i]->neighbor(j)->center());
                                }
                            }
                        }
                        else{
                            // if the neighbor cell is not active then it has children
                            // Here we check those children that touch the face of the tested cell
                            for (unsigned int ichild = 0; ichild < tested_cells[i]->neighbor(j)->n_children(); ++ichild){
                                // if the child has children, this child doesnt not touch the test cell and we will examine it later
                                // However if the child is active we will add it this iteration even if it doesnt touch the cell in question
                                if(tested_cells[i]->neighbor(j)->child(ichild)->active()){
                                    neighbor_child = tested_cells[i]->neighbor(j)->child(ichild);
                                    if (!cell_exists(cells_checked, neighbor_child->center()) &
                                            !cell_exists(cells2check, neighbor_child->center())){
                                        if (!neighbor_child->is_artificial()){
                                            adjacent_cells.push_back(neighbor_child);
                                            cells2check.push_back(neighbor_child->center());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // The adjacent_cells is a list of cells that are likely to contain the point
            tested_cells.clear();
            //if (dbg_pnt)
            //    std::cout << "^^^^^^^^^^^^ [" << nSearch << "] ^^^^^^^^^^^^^^" << std::endl;
            for (unsigned int i = 0; i < adjacent_cells.size(); ++i){
                //if (dbg_pnt){
                    //Point<dim> minp(10000, 10000, 10000);
                    //Point<dim> maxp(-10000, -10000, -10000);
                    //Point <dim> ttt;
                    //for (unsigned int vertex_no = 0; vertex_no < GeometryInfo<dim>::vertices_per_cell; ++vertex_no){
                    //    ttt = adjacent_cells[i]->vertex(vertex_no);
                    //    for (unsigned int iii = 0; iii < dim; ++iii){
                    //        if (ttt[iii] < minp[iii])
                    //            minp[iii] = ttt[iii];
                    //        if (ttt[iii] > maxp[iii])
                    //            maxp[iii] = ttt[iii];
                    //    }
                    //}
                    //std::cout << "cell: [" << minp[0] << " " << minp[1] << " " << minp[2] << "], ["
                    //          << maxp[0] << " " << maxp[1] << " " << maxp[2] << "]" << std::endl;

                //}
                bool is_in_cell = adjacent_cells[i]->point_inside(p);
                if (is_in_cell){
                    cell_found = true;
                    if (bprint_DBG)
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
                cells2check.clear();
            }
        }
    }
    return outcome;
}

template <int dim>
int Particle_Tracking<dim>::compute_point_velocity(Point<dim>& p, Point<dim>& v, typename DoFHandler<dim>::active_cell_iterator &cell, int check_point_status, int caller){
    int outcome = -101;
    if (check_point_status < 0 || cell->is_artificial()){
        std::cerr << "Proc " << dbg_my_rank << " attempts compute_point_velocity for point ("
                  << p[0] << "," << p[1] << "," << p[2]
                  << "), for Eid: " << dbg_curr_Eid << " and Sid: "
                  << dbg_curr_Sid
                  << " however the check_point_status is negative" << std::endl;
        outcome = -99; //GHOST
        return outcome;
    }

    Point<dim> p_unit;
    const MappingQ1<dim> mapping;
    bool success = try_mapping(p, p_unit, cell, mapping);
    if (!success){
        std::cerr << "P fail v1:" << p << " caller: " << caller << std::endl;
        outcome = -88; //FAIL
        return outcome;
    }


    if (check_point_status == 0){
        // no new cell has been found the particle possible has left the domain for ever
        if (p_unit[dim-1] > 1){ // the particle exits from the top face which is what we want
            outcome = 1; //TOP
            return outcome;
        }
        else if (p_unit[dim-1] < 0){ // the particle exits from the bottom face (BAD BAD BAD!!!)
            outcome = -9; //BOT
            return outcome;
        }
        else if(p_unit[0] < 0 || p_unit[0] > 1){ // the particle exits from either side in x direction (not ideal but its ok)
            outcome = 2; //SIDEX
            return outcome;
        }
        else if (dim == 3){
            if (p_unit[1] < 0 || p_unit[1] > 1){ // same as above
               outcome = 2;
               return outcome; //SIDEY
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
        bool new_way = true;
        if (new_way){
            double por = porosity.value(p);
            Tensor<2,dim> HK = HK_function.value(p);

            for (unsigned int idim = 0; idim < dim; ++idim)
                v[idim] = 0;
            typename std::map<unsigned int, AverageVel<dim>>::iterator vel_it;
            const unsigned int dofs_per_cell = fe.dofs_per_cell;
            std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
            cell->get_dof_indices (local_dof_indices);

            Quadrature<dim> temp_quadrature(p_unit);
            FEValues<dim> fe_values_temp(fe, temp_quadrature, update_values);
            fe_values_temp.reinit(cell);
            for (unsigned int i = 0; i < dofs_per_cell; ++i){
                double N = fe_values_temp.shape_value(i,0);
                vel_it = VelocityMap.find(local_dof_indices[i]);
                if (vel_it != VelocityMap.end()){
                    for (unsigned int idim = 0; idim < dim; ++idim)
                        v[idim] += N * vel_it->second.av_vel[idim];
                }
                else {
                    return -98;//DOF_NOT_FOUND
                }
            }
            for (unsigned int idim = 0; idim < dim; ++idim){
                v[idim] = (HK[idim][idim]*v[idim])/por;
            }
            return 0;//NO_EXIT
        }
        else {
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
            return 0;//NO_EXIT
        }
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
    else{
        std::cerr << "P fail v2:" << p << std::endl;
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
                            for (unsigned int ichild = 0; ichild < tested_cells[i]->neighbor(j)->n_children(); ++ichild){
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
        std::cerr << "P fail v3:" << p << std::endl;
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
    int outcome = -9999;//UNKNOWN
    int last = streamline.P.size()-1; // this is the index of the last point in the streamline
    double step_lenght = time_step_multiplier(cell) *(cell->minimum_vertex_distance()/param.step_size);
    double step_time;
    Point<dim> next_point;
    Point<dim> temp_velocity;
    //typename DoFHandler<dim>::active_cell_iterator init_cell = cell;

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
        if (out != 0 ){
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
    return outcome;
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
double Particle_Tracking<dim>::calculate_step(typename DoFHandler<dim>::active_cell_iterator& cell, Point<dim>& Vel){
    BoundingBox<dim> BB = cell->bounding_box();
    std::pair<Point<dim>, Point<dim>> bb_points = BB.get_boundary_points();

    return 0.0;




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
        double Vn = Vel.norm_square();
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
            return 55; //CHANGE_PROC
        }
        else if (cell->is_locally_owned()){
            //plot_segment(streamline.P[streamline.P.size()-1], p);
            streamline.add_point_vel(p, vel, cell->subdomain_id());
            return 0; //NO_EXIT
        }
    }
    else{
        return return_value;
    }
    // The code should never reach this point but I added return statement to supress warnings
    std::cerr << "Particle_Tracking<dim>::add_streamline_point should never reach this point" << std::endl;
    return return_value;
}

template <int dim>
int Particle_Tracking<dim>::take_euler_step(typename DoFHandler<dim>::active_cell_iterator &cell,
                                            double step_weight, double step_length,
                                            Point<dim> P_prev, Point<dim> V_prev,
                                            Point<dim>& P_next, Point<dim>& V_next, int& count_nest){
    if (std::abs(V_prev.norm()) < 0.00000001)
        std::cout << "velocity is almost zero" << std::endl;
    double step_time = step_weight * step_length / V_prev.norm();
    int outcome;

    for (int i = 0; i < dim; ++i){
        P_next[i] = P_prev[i] + V_prev[i] * step_time;
        if (std::isnan(P_next[i])){
            std::cout << "Next point is nan = " << P_prev[i] << " + " << V_prev[i] << "*" << step_time << std::endl;
        }
    }
    //plot_point(P_next);

    int check_pnt = check_cell_point(cell, P_next);

    if (check_pnt < 0){
        step_length = step_length/5.0;
        count_nest++;
        outcome = take_euler_step(cell, step_weight, step_length, P_prev, V_prev, P_next, V_next, count_nest);
    }
    else{
        outcome = compute_point_velocity(P_next, V_next, cell, check_pnt, 1);
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
        dbg_file << "S(" << dbg_i_strm << ",1).Cells(" << dbg_i_step << ",1).Nodes = ["
                 << verts[0][0] << " " << verts[0][1] << ";"
                 << verts[1][0] << " " << verts[1][1] << ";"
                 << verts[3][0] << " " << verts[3][1] << ";"
                 << verts[2][0] << " " << verts[2][1] << "];" << std::endl;
    }

    dbg_file << "S(" << dbg_i_strm << ",1).Cells(" << dbg_i_step << ",1).Type = " << cell_type << ";" << std::endl;

}

template <int dim>
void Particle_Tracking<dim>::print_point_var(Point<dim> p, int r){
    if (dim == 3)
        dbg_file << "S(" << dbg_i_strm << ",1).P(" << dbg_i_step << ",1).XYZ = ["
                 << p[0] << " " << p[1] << " " << p[2] << "];" << std::endl;
    else if (dim == 2)
        dbg_file << "S(" << dbg_i_strm << ",1).P(" << dbg_i_step << ",1).XYZ = ["
                 << p[0] << " " << p[1] << "];" << std::endl;
    dbg_file << "S(" << dbg_i_strm << ",1).P(" << dbg_i_step << ",1).type = " << r << ";" << std::endl;
    dbg_i_step++;
}

template <int dim>
void Particle_Tracking<dim>::print_strm_exit_info(int r, int Eid, int Sid){
    dbg_file << "S(" << dbg_i_strm << ",1).Exit =" << r << ";" << std::endl;
    dbg_file << "S(" << dbg_i_strm << ",1).Eid =" << Eid << ";" << std::endl;
    dbg_file << "S(" << dbg_i_strm << ",1).Sid =" << Sid << ";" << std::endl;
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

    return -9;
}

template <int dim>
double Particle_Tracking<dim>::time_step_multiplier(typename DoFHandler<dim>::active_cell_iterator cell){

    double time_step = 1.0;
    return time_step;
    if (!cell->is_locally_owned())
        time_step = 0.5;

    for (unsigned int j = 0; j < GeometryInfo<dim>::faces_per_cell; ++j){
        if (cell->at_boundary(j)){
            if (time_step > 0.3)
                time_step = 0.3;
        }
        else{
            if (cell->neighbor(j)->active()){
                if (!cell->neighbor(j)->is_locally_owned()){
                    if (time_step > 0.5)
                        time_step = 0.5;
                }
                for (unsigned int k = 0; k < GeometryInfo<dim>::faces_per_cell; ++k){
                    if(cell->neighbor(j)->at_boundary(k))
                        if (time_step > 0.75)
                            time_step = 0.75;
                }
            }
            else{
                for (unsigned int ichild = 0; ichild < cell->neighbor(j)->n_children(); ++ ichild){
                    if(cell->neighbor(j)->child(ichild)->active()){
                        if (!cell->neighbor(j)->child(ichild)->is_locally_owned())
                            if (time_step > 0.5)
                                time_step = 0.5;

                        for (unsigned int k = 0; k < GeometryInfo<dim>::faces_per_cell; ++k){
                            if(cell->neighbor(j)->child(ichild)->at_boundary(k))
                                if (time_step > 0.75)
                                    time_step = 0.75;
                        }
                    }
                }
            }
        }
    }
    return time_step;
}

template <int dim>
bool Particle_Tracking<dim>::cell_exists(std::vector<Point<dim>> PointList, Point<dim> test_point){
    for (unsigned int i = 0; i < PointList.size(); ++i){
        if (test_point.distance(PointList[i]) < 0.001)
            return true;
    }

    return false;
}

template <int dim>
void Particle_Tracking<dim>::print_cell_velocity(std::vector<Point<dim>> p){



    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell){
        if (cell->is_locally_owned()){
            for (unsigned int ii = 0; ii < p.size(); ++ii){
                if (p[ii].distance(cell->center()) < 0.01){
                    std::vector<Point<dim>> verts;
                    std::vector<Point<dim>> vel;
                    calculate_cell_velocity(cell, verts, vel);
                    for (unsigned int jj = 0; jj < verts.size(); ++jj){
                        if (dim == 3)
                            dbg_cell_file << verts[0] << ", " << verts[1] << ", " << verts[2] << ", " << vel[0] << ", " << vel[1] << ", " << vel[2] << std::endl;
                        else if (dim == 2)
                            dbg_cell_file << verts[0] << ", " << verts[1] << ", " << vel[0] << ", " << vel[1] << std::endl;
                    }
                    break;
                }
            }
        }
    }
}

template <int dim>
void Particle_Tracking<dim>::calculate_cell_velocity(typename DoFHandler<dim>::active_cell_iterator& cell,
                                                     std::vector<Point<dim>>& p,
                                                     std::vector<Point<dim>>& vel){

    p.clear();
    vel.clear();
    for (unsigned int vertex_no = 0; vertex_no < GeometryInfo<dim>::vertices_per_cell; ++vertex_no){
        Point<dim> pp = cell->vertex(vertex_no);
        Point<dim> vv;
        if (calc_vel_on_point(cell, pp, vv)){
            p.push_back(pp);
            vel.push_back(vv);
        }
    }
}

template<int dim>
bool Particle_Tracking<dim>::calc_vel_on_point(typename DoFHandler<dim>::active_cell_iterator& cell,
                                               Point<dim> p,
                                               Point<dim>& vel){

    Point<dim> p_unit;
    const MappingQ1<dim> mapping;
    bool success = try_mapping(p, p_unit, cell, mapping);
    if (!success){
        std::cerr << "P fail v4:" << p << std::endl;
        return false;
    }

    // Replace the manual gradient calculation with a more general simpler "Deal.ii" way:

    // To compute the head derivatives at the current unit position
    // we set a quadrature rule at the current point
    Quadrature<dim> point_quadrature(p_unit);
    FEValues<dim> fe_values_point(fe, point_quadrature, update_gradients);
    fe_values_point.reinit(cell);
    const unsigned int n_q_points = point_quadrature.size();

    std::vector<Tensor<1, dim> > hgrad(n_q_points);
    fe_values_point.get_function_gradients(locally_relevant_solution, hgrad);
    Tensor<1,dim> dHead;
    for (int i = 0; i < dim; ++i){
        dHead[i] = hgrad[0][i];
    }
    //if (p[0] < 0.1){
    //    std::cout << p[0] << ", " << p[1] << "," << p[2] << "," << dHead[0] << "," << dHead[1] << "," << dHead[2] << std::endl;
    //}
    /*
    //The velocity is equal vx = - Kx*dHx/n
    // dHi is computed as dN1i*H1i + dN2i*H2i + ... + dNni*Hni, where n is dofs_per_cell and i=[x y z]
    // For the current cell we will extract the hydraulic head solution


    const unsigned int dofs_per_cell = fe.dofs_per_cell;

    // The trapezoid formula evaluate the solution at the corner of the cell
    QTrapez<dim> trapez_formula;
    FEValues<dim> fe_values_trapez(fe, trapez_formula, update_values);
    fe_values_trapez.reinit(cell);
    std::vector<double> current_cell_head(dofs_per_cell);
    fe_values_trapez.get_function_values(locally_relevant_solution, current_cell_head);


    // To compute the head derivatives at the current unit position
    // we set a quadrature rule at the current point
    Quadrature<dim> temp_quadrature(p_unit);
    FEValues<dim> fe_values_temp(fe, temp_quadrature, update_gradients);
    fe_values_temp.reinit(cell);
    const unsigned int n_temp_q_points = temp_quadrature.size();

    // dHead is the head gradient and is initialized to zero
    Tensor<1,dim> dHead;
    //for (int i = 0; i < dim; ++i){
    //    dHead[i] = 0;
    //}
    for (unsigned int i = 0; i < dofs_per_cell; ++i){
        Tensor<1,dim> dN = fe_values_temp.shape_grad(i,0);
        for (int i_dim = 0; i_dim < dim; ++i_dim){
            dHead[i_dim] = dHead[i_dim] + dN[i_dim]*current_cell_head[i];
        }
    }
    */


    //Tensor<2,dim> KK = HK_function.value(p);

    //Vector<double> local_dof_values(cell->get_fe().dofs_per_cell);
    //cell->get_dof_values(locally_relevant_solution, local_dof_values);
    //std::vector< typename OutputType< typename InputVector::value_type >::gradient_type> head_grad;
    //FEValuesViews::Scalar<dim>::get_function_gradients_from_local_dof_values(  local_dof_values, head_grad);

    bool returnHeadGrad = true;
    if (!returnHeadGrad){
        // divide dHead with the porosity
        double por = porosity.value(p);
        Tensor<1,dim> KdH = HK_function.value(p)*dHead;
        for (int i_dim = 0; i_dim < dim; ++i_dim){
            vel[i_dim] = KdH[i_dim] / por;
        }
    }
    else{
        for (int i_dim = 0; i_dim < dim; ++i_dim){
            vel[i_dim] = dHead[i_dim];
        }
    }

    //------------------------------
    return true;
}

template <int dim>
bool Particle_Tracking<dim>::calc_vel_on_point_n_info(typename DoFHandler<dim>::active_cell_iterator& cell,
                                                      Point<dim> p,
                                                      Point<dim>& vel,
                                                      Point<dim> &dH,
                                                      double& H){

    Point<dim> p_unit;
    const MappingQ1<dim> mapping;
    bool success = try_mapping(p, p_unit, cell, mapping);
    if (!success){
        std::cerr << "P fail v4:" << p << std::endl;
        return false;
    }

    //The velocity is equal vx = - Kx*dHx/n
    // dHi is computed as dN1i*H1i + dN2i*H2i + ... + dNni*Hni, where n is dofs_per_cell and i=[x y z]
    // For the current cell we will extract the hydraulic head solution

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    // The trapezoid formula evaluate the solution at the corner of the cell
    QTrapez<dim> trapez_formula;
    FEValues<dim> fe_values_trapez(fe, trapez_formula, update_values);
    fe_values_trapez.reinit(cell);
    std::vector<double> current_cell_head(dofs_per_cell);
    fe_values_trapez.get_function_values(locally_relevant_solution, current_cell_head);

    // To compute the head derivatives at the current unit position
    // we set a quadrature rule at the current point
    Quadrature<dim> temp_quadrature(p_unit);
    FEValues<dim> fe_values_temp(fe, temp_quadrature, update_values | update_gradients | update_jacobians |
                                                       update_quadrature_points | update_inverse_jacobians | update_JxW_values);
    fe_values_temp.reinit(cell);

    // dHead is the head gradient and is initialized to zero
    Tensor<1,dim> dHead, dHead1;

    std::cout << fe_values_temp.JxW(0) << std::endl;
    H = 0;
    for (unsigned int i = 0; i < dofs_per_cell; ++i){
        Tensor<1,dim> dN = fe_values_temp.shape_grad(i,0);
        std::cout << dN[0] << ", " << dN[1] << " | " << fe_values_temp.shape_value(i,0) <<  std::endl;
        H += fe_values_temp.shape_value(i,0)*current_cell_head[i];
    }


    for (unsigned int irow = 0; irow < dim; ++irow){
        for (unsigned int i = 0; i < dofs_per_cell; ++i){
            Tensor<1,dim> dN = fe_values_temp.shape_grad(i,0);
            double JN = 0;
            for (unsigned int icol = 0; icol < dim; ++icol){
                double inv = fe_values_temp.inverse_jacobian(0)[icol][irow];
                double Jac = fe_values_temp.jacobian(0)[icol][irow];
                std::cout << Jac << " | " << inv << std::endl;
                //double dn = dN[icol];
                JN += fe_values_temp.inverse_jacobian(0)[icol][irow]*dN[icol];
            }
            dHead1[irow] += JN*current_cell_head[i];
        }
    }

    for (unsigned int i = 0; i < dofs_per_cell; ++i){
        Tensor<1,dim> dN = fe_values_temp.shape_grad(i,0);
        for (int i_dim = 0; i_dim < dim; ++i_dim){
            dHead[i_dim] = dHead[i_dim] + dN[i_dim]*current_cell_head[i];
        }
    }




    // divide dHead with the porosity
    double por = porosity.value(p);
    Tensor<1,dim> KdH = HK_function.value(p)*dHead;
    for (int i_dim = 0; i_dim < dim; ++i_dim){
        vel[i_dim] = KdH[i_dim] / por;
    }

    for (unsigned int idim = 0; idim < dim; ++idim)
        dH[idim] = dHead[idim];

    return true;
}

template <int dim>
void Particle_Tracking<dim>::print_all_cell_velocity(){
    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell){
        if (cell->is_locally_owned()){
            std::vector<Point<dim>> verts;
            std::vector<Point<dim>> vel;
            calculate_cell_velocity(cell, verts, vel);
            for (unsigned int jj = 0; jj < verts.size(); ++jj){
                dbg_cell_file << verts[jj][0] << ", " << verts[jj][1] << ", " << verts[jj][2] << ", " << vel[jj][0] << ", " << vel[jj][1] << ", " << vel[jj][2] << std::endl;
            }
        }
    }
}

/*
template <int dim>
void Particle_Tracking<dim>::average_velocity_field(DoFHandler<dim>& velocity_dof_handler,
                                                    FESystem<dim>& velocity_fe){
    MPI_Barrier(mpi_communicator);
    const MappingQ1<dim> mapping;

    pcout << "Average Velocity field" << std::endl << std::flush;
    velocity_dof_handler.distribute_dofs(velocity_fe);

    IndexSet    velocity_locally_owned;
    IndexSet    velocity_locally_relevant;

    velocity_locally_owned = velocity_dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(velocity_dof_handler, velocity_locally_relevant);

    TrilinosWrappers::MPI::Vector               velocity_vector;
    TrilinosWrappers::MPI::Vector               distributed_velocity_vector;
    velocity_vector.reinit(velocity_locally_owned, velocity_locally_relevant, mpi_communicator);
    distributed_velocity_vector.reinit(velocity_locally_owned, velocity_locally_relevant, mpi_communicator);

    const std::vector<Point<dim> > velocity_support_points
                                  = velocity_fe.base_element(0).get_unit_support_points();

    FEValues<dim> fe_velocities (mapping,
                                 velocity_fe,
                                 velocity_support_points,
                                 update_quadrature_points);

    ConstraintMatrix    velocity_constraints;
    velocity_constraints.clear();
    velocity_constraints.reinit(velocity_locally_relevant);
    DoFTools::make_hanging_node_constraints(velocity_dof_handler, velocity_constraints);
    velocity_constraints.close();

    //std::map<unsigned int, AverageVel> VelocityMap;
    std::map<unsigned int, AverageVel>::iterator vel_it, vel_it2;

    MPI_Barrier(mpi_communicator);
    std::vector<unsigned int> cell_dof_indices (velocity_fe.dofs_per_cell);
    typename DoFHandler<dim>::active_cell_iterator
    cell = velocity_dof_handler.begin_active(),
    endc = velocity_dof_handler.end();

    typename DoFHandler<dim>::active_cell_iterator cell_sol = dof_handler.begin_active();


    for (; cell != endc; ++cell){
        if (cell->is_locally_owned() || cell->is_ghost()){
            fe_velocities.reinit(cell);
            cell->get_dof_indices (cell_dof_indices);

            for (unsigned int idof = 0; idof < velocity_fe.base_element(0).dofs_per_cell; ++idof){
                Point <dim> current_node;
                std::vector<unsigned int> current_dofs(dim);
                for (unsigned int dir = 0; dir < dim; ++dir){
                    unsigned int support_point_index = velocity_fe.component_to_system_index(dir, idof );
                    current_dofs[dir] = cell_dof_indices[support_point_index];
                    current_node[dir] = fe_velocities.quadrature_point(idof)[dir];
                }
                Point<dim> vel;
                calc_vel_on_point(cell_sol, current_node, vel);
                for (unsigned int dir = 0; dir < dim; ++dir){
                    vel_it = VelocityMap.find(current_dofs[dir]);
                    if (vel_it != VelocityMap.end()){
                        vel_it->second.Addvelocity(vel[dir]);
                    }
                    else{
                        std::vector<types::global_dof_index> temp_cnstr;
                        temp_cnstr.push_back(current_dofs[dir]);
                        velocity_constraints.resolve_indices(temp_cnstr);
                        VelocityMap.insert(std::pair<unsigned int, AverageVel>(current_dofs[dir],
                                                                               AverageVel(distributed_velocity_vector.in_local_range(current_dofs[dir]),
                                                                                          current_dofs[dir],
                                                                                          vel[dir],
                                                                                          temp_cnstr)));
                    }
                }
            }
        }
        ++cell_sol;
    }

    // Average Velocities
    while (true){
        int count_non_average = 0;
        vel_it = VelocityMap.begin();
        for (; vel_it != VelocityMap.end(); ++vel_it){
            if (vel_it->second.is_local && !vel_it->second.is_averaged){
                if (vel_it->second.cnstr.size() == 0){
                    // if its not constraint
                    double sum = 0;
                    for (unsigned int ii = 0; ii < vel_it->second.V.size(); ++ii){
                        sum += vel_it->second.V[ii];
                    }
                    vel_it->second.av_vel = sum / vel_it->second.V.size();
                    vel_it->second.is_averaged = true;
                }
                else{
                    double sum = 0;
                    bool can_average = true;
                    for (unsigned int ii = 0; ii < vel_it->second.cnstr.size(); ++ii){
                        vel_it2 = VelocityMap.find(vel_it->second.cnstr[ii]);
                        if (vel_it2 != VelocityMap.end()){
                            if (vel_it2->second.is_averaged){
                                sum += vel_it2->second.av_vel;
                            }
                            else{
                                can_average = false;
                                break;
                            }
                        }

                    }
                    if (can_average){
                        vel_it->second.av_vel = sum/vel_it->second.cnstr.size();
                        vel_it->second.is_averaged = true;
                    }
                    else{
                        count_non_average++;
                    }
                }
            }
        }
        std::cout << "velocities to average: " << count_non_average << std::endl;
        if (count_non_average == 0)
            break;

    }

    // just a check
    vel_it = VelocityMap.begin();
    int count = 0;
    for (; vel_it != VelocityMap.end(); ++vel_it){
        if (!vel_it->second.is_averaged)
            count++;
    }
    std::cout << "Not averaged " << count << std::endl;
}
*/

template <int dim>
bool Particle_Tracking<dim>::average_velocity_field(){
    unsigned int my_rank = Utilities::MPI::this_mpi_process(mpi_communicator);
    unsigned int n_proc = Utilities::MPI::n_mpi_processes(mpi_communicator);

    //const std::string vField_file_name = ("VelField_"	+
    //                                   Utilities::int_to_string(my_rank, 4) +
    //                                   ".dbg");
    //std::ofstream vField_file;
    //vField_file.open(vField_file_name);

    MPI_Barrier(mpi_communicator);
    pcout << "Calculating Velocities..." << std::endl << std::flush;

    typename std::map<unsigned int, AverageVel<dim>>::iterator vel_it, vel_it2;
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    // Calculate velocities

    IndexSet locally_owned_indices = locally_relevant_solution.locally_owned_elements();
    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell != endc; ++cell){
        if (cell->is_locally_owned() || cell->is_ghost()){
            //fe.reinit(cell);
            cell->get_dof_indices (local_dof_indices);
            for (unsigned int ii = 0; ii < local_dof_indices.size(); ++ii){
                Point<dim> vel;
                bool tf = calc_vel_on_point(cell, cell->vertex(ii), vel);
                //vField_file << cell->vertex(ii)[0] << " " << cell->vertex(ii)[1] << " " << cell->vertex(ii)[2] << " " << static_cast<int>(cell->is_locally_owned()) << std::endl;
                if (tf){
                    vel_it = VelocityMap.find(local_dof_indices[ii]);
                    if (vel_it != VelocityMap.end()){
                        vel_it->second.Addvelocity(vel);
                    }
                    else{
                        std::vector<types::global_dof_index> temp_cnstr;
                        temp_cnstr.push_back(local_dof_indices[ii]);
                        Headconstraints.resolve_indices(temp_cnstr);

                        VelocityMap.insert(std::pair<unsigned int, AverageVel<dim>>(local_dof_indices[ii],
                                           AverageVel<dim>(locally_owned_indices.is_element(local_dof_indices[ii]),
                                                           local_dof_indices[ii], vel, temp_cnstr)));
                    }
                }
            }
        }
    }

    // DEBUG--------------------
    //vel_it = VelocityMap.begin();
    //for (; vel_it != VelocityMap.end(); ++vel_it){
    //    vField_file << vel_it->second.dof << "\t"
    //                << vel_it->second.is_local << "\t"
    //                << vel_it->second.V.size() << "\t"
    //                << vel_it->second.cnstr.size() << "\t";
    //    if (vel_it->second.cnstr.size() > 0){
    //        vField_file << "| ";
    //        for (unsigned int ii = 0; ii < vel_it->second.cnstr.size(); ++ii){
    //            vField_file << vel_it->second.cnstr[ii] << " | ";
    //        }
    //    }
    //    vField_file << std::endl;
    //}
    //vField_file.close();


    // DEBUG--------------------


    // Average Velocities
    pcout << "Averaging Velocities..." << std::endl << std::flush;
    int count_iter = 0;
    std::vector<std::vector<int>> not_known_id(n_proc);
    std::vector<std::vector<int>> known_ids(n_proc);
    std::vector<std::vector<double>> velX(n_proc);
    std::vector<std::vector<double>> velY(n_proc);
    std::vector<std::vector<double>> velZ(n_proc);
    std::map<int, Point<dim>> received_vel;
    typename std::map<int, Point<dim>>::iterator rec_it;
    while (true){
        not_known_id[my_rank].clear();
        known_ids[my_rank].clear();
        velX[my_rank].clear();
        velY[my_rank].clear();
        velZ[my_rank].clear();
        int count_non_average = 0;

        vel_it = VelocityMap.begin();
        for (; vel_it != VelocityMap.end(); ++vel_it){
            if (vel_it->second.is_local){
                if (!vel_it->second.is_averaged){
                    if (vel_it->second.cnstr.size() == 0){ // if its not constraint
                        Point<dim> sum;
                        for (unsigned int ii = 0; ii < vel_it->second.V.size(); ++ii){
                            for (unsigned int idim = 0; idim < dim; ++idim)
                                sum[idim] += vel_it->second.V[ii][idim];
                        }
                        for (unsigned int idim = 0; idim < dim; ++idim)
                            vel_it->second.av_vel[idim] = sum[idim]/vel_it->second.V.size();
                        vel_it->second.is_averaged = true;
                    }
                    else{
                        Point<dim> sum;
                        bool can_average = true;
                        for (unsigned int ii = 0; ii < vel_it->second.cnstr.size(); ++ii){
                            vel_it2 = VelocityMap.find(vel_it->second.cnstr[ii]);
                            if (vel_it2 != VelocityMap.end()){
                                if (vel_it2->second.is_averaged){
                                    for (unsigned int idim = 0; idim < dim; ++idim)
                                        sum[idim] += vel_it2->second.av_vel[idim];
                                }
                                else{
                                    can_average = false;
                                    break;
                                }
                            }
                            else{
                                rec_it = received_vel.find(vel_it->second.cnstr[ii]);
                                if (rec_it != received_vel.end()){
                                    for (unsigned int idim = 0; idim < dim; ++idim)
                                        sum[idim] += rec_it->second[idim];
                                }
                                else{
                                    not_known_id[my_rank].push_back(vel_it->second.cnstr[ii]);
                                    can_average = false;
                                    break;
                                }
                            }
                        }
                        if (can_average){
                            for (unsigned int idim = 0; idim < dim; ++idim)
                                vel_it->second.av_vel[idim] = sum[idim] / vel_it->second.cnstr.size();
                            vel_it->second.is_averaged = true;
                        }
                        else{
                            count_non_average++;
                        }
                    }
                }
            }
            else {
                // if the velocity its not local then we should get it from the processor that it has as local
                // check first if we have asked for it in a previous iteration
                rec_it = received_vel.find(vel_it->second.dof);
                if (rec_it != received_vel.end()){
                    for (unsigned int idim = 0; idim < dim; ++idim)
                        vel_it->second.av_vel[idim] = rec_it->second[idim];
                    vel_it->second.is_averaged = true;
                }
                else{
                    not_known_id[my_rank].push_back(vel_it->second.dof);
                    count_non_average++;
                }
            }
        }

        //std::cout << "I'm Rank " << my_rank << " has " << count_non_average << " unkwown velocities" << std::endl;

        // Check if all points have been set
        sum_scalar<int>(count_non_average, n_proc, mpi_communicator, MPI_INT);
        pcout << "There are: " << count_non_average << " point velocities to be averaged" << std::endl;
        if (count_non_average == 0)
            break;



        std::vector<int> sent_size;
        Send_receive_size(static_cast<unsigned int>(not_known_id[my_rank].size()), n_proc, sent_size, mpi_communicator);
        Sent_receive_data<int>(not_known_id, sent_size, my_rank, mpi_communicator, MPI_INT);



        for (unsigned int i_proc = 0; i_proc < n_proc; ++i_proc){
            if (my_rank == i_proc)
                continue;
            for (unsigned int ii = 0; ii < not_known_id[i_proc].size(); ++ii){
                vel_it = VelocityMap.find(not_known_id[i_proc][ii]);
                if (vel_it != VelocityMap.end()){
                    if (vel_it->second.is_local && vel_it->second.is_averaged){
                        known_ids[my_rank].push_back(vel_it->second.dof);
                        velX[my_rank].push_back(vel_it->second.av_vel[0]);
                        velY[my_rank].push_back(vel_it->second.av_vel[1]);
                        if (dim == 3)
                            velZ[my_rank].push_back(vel_it->second.av_vel[2]);
                    }
                }
            }
        }

        //std::cout << "I'm Rank " << my_rank << " and I have " << known_ids[my_rank].size() << " to send" << std::endl;
        Send_receive_size(static_cast<unsigned int>(known_ids[my_rank].size()), n_proc, sent_size, mpi_communicator);
        Sent_receive_data<int>(known_ids, sent_size, my_rank, mpi_communicator, MPI_INT);
        Sent_receive_data<double>(velX, sent_size, my_rank, mpi_communicator, MPI_DOUBLE);
        Sent_receive_data<double>(velY, sent_size, my_rank, mpi_communicator, MPI_DOUBLE);
        if (dim == 3)
            Sent_receive_data<double>(velZ, sent_size, my_rank, mpi_communicator, MPI_DOUBLE);

        // loop through the replies and get th ones you need
        // but first put them in a map for faster querying

        for (unsigned int i_proc = 0; i_proc < n_proc; ++ i_proc){
            if (my_rank == i_proc)
                continue;
            for (unsigned int ii = 0; ii < known_ids[i_proc].size(); ++ii){
                Point<dim> tempV;
                tempV[0] = velX[i_proc][ii];
                tempV[1] = velY[i_proc][ii];
                if (dim == 3)
                    tempV[2] = velZ[i_proc][ii];
                received_vel.insert(std::pair<int,Point<dim>>(known_ids[i_proc][ii], tempV));
            }
        }


        count_iter++;
        if (count_iter >= 20){
            std::cerr << " Not all point velocities could be averaged after 20 iterations" << std::endl;
            return false;
        }
    }
    return true;
}

template <int dim>
bool Particle_Tracking<dim>::average_velocity_field1(){
    unsigned int my_rank = Utilities::MPI::this_mpi_process(mpi_communicator);
    unsigned int n_proc = Utilities::MPI::n_mpi_processes(mpi_communicator);

    MPI_Barrier(mpi_communicator);
    pcout << "Calculating Head gradients..." << std::endl << std::flush;

    typename std::map<unsigned int, AverageVel<dim>>::iterator vel_it, vel_it2;
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    std::map<int,int> dof_on_ghost;
    std::map<int,int> dof_on_shared;
    std::map<int,int>::iterator itint;

    //double premax_v = -99999999999;
    //double premin_v =  99999999999;

    // Printing to file information. This should be removed -----------
    //unsigned int cell_id = 0;
    //const std::string info_vel_name = (param.particle_prefix + "vel_field.dat");
    //std::ofstream info_vel_file;
    //info_vel_file.open(info_vel_name.c_str());
    //-------------------------------------------------------------------

    // Logic of the following loop
    // We loop through the localy owned cells of the triangulation
    // Then loop through the faces of each cell and check if any active neighbors are ghost
    // if Yes then make a list (ids_that_touch_ghost_cells) of those dofs that are shared with the ghost cell.
    // Next loop through the dofs of the cell and calculate the velocity for each dof.

    IndexSet locally_owned_indices = locally_relevant_solution.locally_owned_elements();
    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell != endc; ++cell){
        if (cell->is_locally_owned()){
            cell->get_dof_indices (local_dof_indices);
            std::map <int, int> ids_that_touch_ghost_cells;
            // find if there are any ghost cells that are touching this one
            for (unsigned int iface = 0; iface < GeometryInfo<dim>::faces_per_cell; ++iface){
                if (!cell->at_boundary(iface)){ // if the iface is not a boundary
                    if (cell->neighbor(iface)->active()){
                        // if the neighbor is active check if its ghost
                        if (cell->neighbor(iface)->is_ghost()){
                            for (unsigned int ivert = 0; ivert < GeometryInfo<dim>::vertices_per_face; ++ivert){
                                unsigned int id = GeometryInfo<dim>::face_to_cell_vertices(iface,ivert);
                                ids_that_touch_ghost_cells.insert(std::pair<int,int>(id, id));
                            }

                        }
                    }
                    else{
                        // if the neighbor is not active check if any of the children are ghost
                        for (unsigned int ichild = 0; ichild < cell->neighbor(iface)->n_children(); ++ichild){
                            // If the child of a neighbor has children then its divided once more and therefore does not touch the current cell
                            if (cell->neighbor(iface)->child(ichild)->active()){
                                if (cell->neighbor(iface)->child(ichild)->is_ghost()){
                                    for (unsigned int ivert = 0; ivert < GeometryInfo<dim>::vertices_per_face; ++ivert){
                                        unsigned int id = GeometryInfo<dim>::face_to_cell_vertices(iface,ivert);
                                        ids_that_touch_ghost_cells.insert(std::pair<int,int>(id, id));
                                    }
                                    break;
                                }
                            }
                        }
                    }
                }
            }


            for (unsigned int ii = 0; ii < local_dof_indices.size(); ++ii){
                Point<dim> vel;
                bool tf = calc_vel_on_point(cell, cell->vertex(ii), vel);
                //for (unsigned int idim = 0; idim < dim; ++idim){
                //    if (vel[idim] > premax_v)
                //        premax_v = vel[idim];
                //    if (vel[idim] < premin_v)
                //        premin_v = vel[idim];
                //}

                // Printing to file information. This should be removed -----------
                /*Point<dim> vtmp, dhtmp;
                double Htmp;
                bool tf1 = calc_vel_on_point_n_info(cell, cell->vertex(ii), vtmp, dhtmp, Htmp);
                if (tf1){
                    info_vel_file << std::setprecision(8)
                                  << std::fixed
                                  << std::setw(5) << cell_id << ", "
                                  << std::setw(15) << cell->vertex(ii)[0] << ", "
                                  << std::setw(15) << cell->vertex(ii)[1] << ", "
                                  << std::setw(15) << 0 << ", "
                                  << std::setw(15) << vtmp[0] << ", "
                                  << std::setw(15) << vtmp[1] << ", "
                                  << std::setw(15) << dhtmp[0] << ", "
                                  << std::setw(15) << dhtmp[1] << ", "
                                  << std::setw(15) << Htmp
                                  << std::endl;
                }
                else{
                    info_vel_file << std::setprecision(3)
                                  << std::fixed
                                  << std::setw(5) << cell_id << ", "
                                  << std::setw(15) << cell->vertex(ii)[0] << ", "
                                  << std::setw(15) << cell->vertex(ii)[1] << ", "
                                  << std::setw(15) << -9999 << ", "
                                  << std::setw(15) << -9999 << ", "
                                  << std::setw(15) << -9999 << ", "
                                  << std::setw(15) << -9999
                                  << std::endl;
                }*/
                //-------------------------------------------------------------------

                if (tf){
                    vel_it = VelocityMap.find(local_dof_indices[ii]);
                    if (vel_it != VelocityMap.end()){
                        vel_it->second.Addvelocity(vel);
                    }
                    else{
                        std::vector<types::global_dof_index> temp_cnstr;
                        temp_cnstr.push_back(local_dof_indices[ii]);
                        Headconstraints.resolve_indices(temp_cnstr);
                        VelocityMap.insert(std::pair<unsigned int, AverageVel<dim>>(local_dof_indices[ii],
                                           AverageVel<dim>(locally_owned_indices.is_element(local_dof_indices[ii]),
                                                           local_dof_indices[ii], vel, temp_cnstr)));
                    }
                    itint = ids_that_touch_ghost_cells.find(ii);
                    if (itint != ids_that_touch_ghost_cells.end()){
                        dof_on_shared.insert(std::pair<int,int>(local_dof_indices[ii],local_dof_indices[ii]));
                    }

                }
                else
                    std::cerr << "There was a point that failed to calculate velocity even in local cell" << std::endl;
            }
        }
        else if (cell->is_ghost()){
            cell->get_dof_indices (local_dof_indices);
            for (unsigned int ii = 0; ii < local_dof_indices.size(); ++ii)
                dof_on_ghost.insert(std::pair<int,int>(local_dof_indices[ii],local_dof_indices[ii]));
        }
        //cell_id++;
    }
    //info_vel_file.close();
    //std::cout << my_rank << " Pre Max vel: " << premax_v << ", Pre Min vel: " << premin_v << std::endl;

    //premax_v = -99999999999;
    //premin_v =  99999999999;

    // copy the dofs on ghost from map to vector
    std::vector<std::vector<int>> dof_on_shared_vec(n_proc);
    std::vector<std::vector<int>> nvel(n_proc);
    std::vector<std::vector<double>> velshared(n_proc);
    for (itint = dof_on_shared.begin(); itint !=dof_on_shared.end(); ++itint){
        vel_it = VelocityMap.find(itint->first);
        if (vel_it != VelocityMap.end()){
            dof_on_shared_vec[my_rank].push_back(itint->first);
            nvel[my_rank].push_back(vel_it->second.V.size());
            for (unsigned int ii = 0; ii < vel_it->second.V.size(); ++ii)
                for (unsigned int idim = 0; idim < dim; ++idim){
                    velshared[my_rank].push_back(vel_it->second.V[ii][idim]);
                    //if (vel_it->second.V[ii][idim] > premax_v)
                    //    premax_v = vel_it->second.V[ii][idim];
                    //if (vel_it->second.V[ii][idim] < premin_v)
                    //    premin_v = vel_it->second.V[ii][idim];
                }
        }
    }

    //std::cout << my_rank << " Pre Max vel: " << premax_v << ", Pre Min vel: " << premin_v << std::endl;

    // in the velocity map each processor has put the velocity contributions from its locally own elements.
    // Before averaging we have to transfer contributions from the other processors local elements for the vertices that touch
    // ghost elements. These dofs have been stored to dof_on_shared map
    // Each processor will notify which dofs in locally own cells have ghost neighbors

    std::vector<int> sent_dof_on_share_size, sent_vel_on_share_size;
    Send_receive_size(static_cast<unsigned int>(dof_on_shared_vec[my_rank].size()), n_proc, sent_dof_on_share_size, mpi_communicator);
    Sent_receive_data<int>(dof_on_shared_vec, sent_dof_on_share_size, my_rank, mpi_communicator, MPI_INT);
    Sent_receive_data<int>(nvel, sent_dof_on_share_size, my_rank, mpi_communicator, MPI_INT);
    Send_receive_size(static_cast<unsigned int>(velshared[my_rank].size()), n_proc, sent_vel_on_share_size, mpi_communicator);
    Sent_receive_data<double>(velshared, sent_vel_on_share_size, my_rank, mpi_communicator, MPI_DOUBLE);

    //premax_v = -99999999999;
    //premin_v =  99999999999;

    // each processor loops through the sent data that touch ghost cells and picks the ones that has in its map and coming form the other processors
    for (unsigned int i_proc = 0; i_proc < n_proc; ++i_proc){
        if (my_rank == i_proc)
            continue;
        int vel_cnt = 0;
        for (unsigned int ii = 0; ii < dof_on_shared_vec[i_proc].size(); ++ii){
            vel_it = VelocityMap.find(dof_on_shared_vec[i_proc][ii]);
            int nv = nvel[i_proc][ii];
            if (vel_it != VelocityMap.end()){
                // if I have a dof in my velocity map that another processor has sent its local information then I have to add this in my map
                for (int jj = 0; jj < nv; ++jj){
                    Point<dim> vv;
                    for (unsigned int idim = 0; idim < dim; ++idim){
                        vv[idim] = velshared[i_proc][vel_cnt];
                        vel_cnt++;

                        //if (vv[idim] > premax_v)
                        //    premax_v = vv[idim];
                        //if (vv[idim] < premin_v)
                        //    premin_v = vv[idim];
                    }
                    vel_it->second.Addvelocity(vv);
                }
            }
            else{
                vel_cnt = vel_cnt + nv*dim;
            }
        }
    }

    //std::cout << my_rank << " Pre Max vel: " << premax_v << ", Pre Min vel: " << premin_v << std::endl;

    // Now all processors should have all the values they need to average their velocities
    int count_av_vel_iter = 0;
    std::vector<std::vector<int>> not_known_id(n_proc);
    std::vector<std::vector<int>> known_ids(n_proc);
    std::vector<std::vector<double>> velX(n_proc);
    std::vector<std::vector<double>> velY(n_proc);
    std::vector<std::vector<double>> velZ(n_proc);

    std::map<int, Point<dim>> received_vel;

    double min_vel = 99999999999;
    double max_vel = -99999999999;
    typename std::map<int, Point<dim>>::iterator rec_it;
    while (true){
        MPI_Barrier(mpi_communicator);
        for (unsigned int iproc = 0; iproc < n_proc; ++iproc){
            not_known_id[iproc].clear();
            known_ids[iproc].clear();
            velX[iproc].clear();
            velY[iproc].clear();
            velZ[iproc].clear();
        }
        int count_non_average = 0;
        vel_it = VelocityMap.begin();
        for (; vel_it != VelocityMap.end(); ++vel_it){
            if (!vel_it->second.is_averaged){
                if (vel_it->second.is_local){
                    if (vel_it->second.cnstr.size() == 0){
                        Point<dim> sum;
                        for (unsigned int ii = 0; ii < vel_it->second.V.size(); ++ii){
                            for (unsigned int idim = 0; idim < dim; ++idim)
                                sum[idim] += vel_it->second.V[ii][idim];
                        }
                        for (unsigned int idim = 0; idim < dim; ++idim){
                            vel_it->second.av_vel[idim] = sum[idim]/vel_it->second.V.size();
                            if (vel_it->second.av_vel[idim] > max_vel)
                                max_vel = vel_it->second.av_vel[idim];
                            if (vel_it->second.av_vel[idim] < min_vel)
                                min_vel = vel_it->second.av_vel[idim];
                        }
                        vel_it->second.is_averaged = true;
                    }
                    else{
                        Point<dim> sum;
                        bool can_average = true;
                        for (unsigned int ii = 0; ii < vel_it->second.cnstr.size(); ++ii){
                            vel_it2 = VelocityMap.find(vel_it->second.cnstr[ii]);
                            if (vel_it2 != VelocityMap.end()){
                                if (vel_it2->second.is_averaged){
                                    for (unsigned int idim = 0; idim < dim; ++idim)
                                        sum[idim] += vel_it2->second.av_vel[idim];
                                }
                                else{
                                    can_average = false;
                                    break;
                                }
                            }
                            else{
                                rec_it = received_vel.find(vel_it->second.cnstr[ii]);
                                if (rec_it != received_vel.end()){
                                    for (unsigned int idim = 0; idim < dim; ++idim)
                                        sum[idim] += rec_it->second[idim];
                                }
                                else{
                                    not_known_id[my_rank].push_back(vel_it->second.cnstr[ii]);
                                    can_average = false;
                                    break;
                                }
                            }
                        }
                        if (can_average){
                            for (unsigned int idim = 0; idim < dim; ++idim){
                                vel_it->second.av_vel[idim] = sum[idim] / vel_it->second.cnstr.size();
                                if (vel_it->second.av_vel[idim] > max_vel)
                                    max_vel = vel_it->second.av_vel[idim];
                                if (vel_it->second.av_vel[idim] < min_vel)
                                    min_vel = vel_it->second.av_vel[idim];
                            }

                            vel_it->second.is_averaged = true;
                        }
                        else{
                            count_non_average++;
                        }
                    }

                }
                else{
                    // if the velocity is not local but the velocity Map containts the dof then most likely this will be needed during
                    // particle tracking.
                    // check if we have asked for this in a previous iteration
                    rec_it = received_vel.find(vel_it->second.dof);
                    if (rec_it != received_vel.end()){
                        // the received velocities are always averaged, otherwise the owner processor wont sent them
                        for (unsigned int idim = 0; idim < dim; ++idim)
                            vel_it->second.av_vel[idim] = rec_it->second[idim];
                        vel_it->second.is_averaged = true;
                    }
                    else{
                        not_known_id[my_rank].push_back(vel_it->second.dof);
                        count_non_average++;
                    }
                }
            }
        }

        // Check if all points have been set
        sum_scalar<int>(count_non_average, n_proc, mpi_communicator, MPI_INT);
        pcout << "There are: " << count_non_average << " point velocities to be averaged" << std::endl;
        if (count_non_average == 0)
            break;

        std::vector<int> sent_size;
        Send_receive_size(static_cast<unsigned int>(not_known_id[my_rank].size()), n_proc, sent_size, mpi_communicator);
        Sent_receive_data<int>(not_known_id, sent_size, my_rank, mpi_communicator, MPI_INT);

        for (unsigned int i_proc = 0; i_proc < n_proc; ++i_proc){
            if (my_rank == i_proc)
                continue;
            for (unsigned int ii = 0; ii < not_known_id[i_proc].size(); ++ii){
                vel_it = VelocityMap.find(not_known_id[i_proc][ii]);
                if (vel_it != VelocityMap.end()){
                    if (vel_it->second.is_local && vel_it->second.is_averaged){
                        known_ids[my_rank].push_back(vel_it->second.dof);
                        velX[my_rank].push_back(vel_it->second.av_vel[0]);
                        velY[my_rank].push_back(vel_it->second.av_vel[1]);
                        if (dim == 3)
                            velZ[my_rank].push_back(vel_it->second.av_vel[2]);
                    }
                }
            }
        }

        Send_receive_size(static_cast<unsigned int>(known_ids[my_rank].size()), n_proc, sent_size, mpi_communicator);
        Sent_receive_data<int>(known_ids, sent_size, my_rank, mpi_communicator, MPI_INT);
        Sent_receive_data<double>(velX, sent_size, my_rank, mpi_communicator, MPI_DOUBLE);
        Sent_receive_data<double>(velY, sent_size, my_rank, mpi_communicator, MPI_DOUBLE);
        if (dim == 3)
            Sent_receive_data<double>(velZ, sent_size, my_rank, mpi_communicator, MPI_DOUBLE);

        // loop through the replies and get the ones you need
        // but first put them in a map for faster querying
        for (unsigned int i_proc = 0; i_proc < n_proc; ++ i_proc){
            if (my_rank == i_proc)
                continue;
            for (unsigned int ii = 0; ii < known_ids[i_proc].size(); ++ii){
                Point<dim> tempV;
                tempV[0] = velX[i_proc][ii];
                tempV[1] = velY[i_proc][ii];
                if (dim == 3)
                    tempV[2] = velZ[i_proc][ii];
                received_vel.insert(std::pair<int,Point<dim>>(known_ids[i_proc][ii], tempV));
            }
        }


        count_av_vel_iter++;
        if (count_av_vel_iter > 30){
            pcout << "The velocities have not fully averaged" << std::endl;
            return false;
        }
    }

    //now loop through the dofs that exist in ghost cells
    for (unsigned int iproc = 0; iproc < n_proc; ++iproc){
        not_known_id[iproc].clear();
        known_ids[iproc].clear();
        velX[iproc].clear();
        velY[iproc].clear();
        velZ[iproc].clear();
    }

    for (itint = dof_on_ghost.begin(); itint != dof_on_ghost.end(); ++itint){
        vel_it = VelocityMap.find(itint->first);
        if (vel_it == VelocityMap.end()){
            // this dof doesnt even exist on the velocity map therefore we have to get it from another processor
            not_known_id[my_rank].push_back(itint->first);
        }
    }
    std::vector<int> sent_size;
    Send_receive_size(static_cast<unsigned int>(not_known_id[my_rank].size()), n_proc, sent_size, mpi_communicator);
    Sent_receive_data<int>(not_known_id, sent_size, my_rank, mpi_communicator, MPI_INT);

    for (unsigned int i_proc = 0; i_proc < n_proc; ++i_proc){
        if (my_rank == i_proc)
            continue;
        for (unsigned int ii = 0; ii < not_known_id[i_proc].size(); ++ii){
            vel_it = VelocityMap.find(not_known_id[i_proc][ii]);
            if (vel_it != VelocityMap.end()){
                if (vel_it->second.is_local && vel_it->second.is_averaged){
                    known_ids[my_rank].push_back(vel_it->second.dof);
                    velX[my_rank].push_back(vel_it->second.av_vel[0]);
                    velY[my_rank].push_back(vel_it->second.av_vel[1]);
                    if (dim == 3)
                        velZ[my_rank].push_back(vel_it->second.av_vel[2]);
                }
            }
        }
    }
    Send_receive_size(static_cast<unsigned int>(known_ids[my_rank].size()), n_proc, sent_size, mpi_communicator);
    Sent_receive_data<int>(known_ids, sent_size, my_rank, mpi_communicator, MPI_INT);
    Sent_receive_data<double>(velX, sent_size, my_rank, mpi_communicator, MPI_DOUBLE);
    Sent_receive_data<double>(velY, sent_size, my_rank, mpi_communicator, MPI_DOUBLE);
    if (dim == 3)
        Sent_receive_data<double>(velZ, sent_size, my_rank, mpi_communicator, MPI_DOUBLE);

    for (unsigned int i_proc = 0; i_proc < n_proc; ++ i_proc){
        if (my_rank == i_proc)
            continue;
        for (unsigned int ii = 0; ii < known_ids[i_proc].size(); ++ii){
            itint = dof_on_ghost.find(known_ids[i_proc][ii]);
            vel_it = VelocityMap.find(known_ids[i_proc][ii]);
            if (itint != dof_on_ghost.end() && vel_it == VelocityMap.end()){
                Point<dim> tempV;
                tempV[0] = velX[i_proc][ii];
                tempV[1] = velY[i_proc][ii];
                if (dim == 3)
                    tempV[2] = velZ[i_proc][ii];
                VelocityMap.insert(std::pair<unsigned int, AverageVel<dim>>(itint->first, AverageVel<dim>(itint->first, tempV)));
            }
        }
    }
    std::cout << "Max vel: " << max_vel << ", Min vel: " << min_vel << std::endl;
    printAveragedVelocityField(param.particle_prefix + "AvVelField" + Utilities::int_to_string(my_rank, 4) + ".vel");

    return true;
}

template <int dim>
void Particle_Tracking<dim>::printAveragedVelocityField(std::string filename){
    std::cout << "Printing Velocity field" << std::endl;
    std::ofstream vel_file;
    vel_file.open(filename.c_str());
    typename std::map<unsigned int, AverageVel<dim>>::iterator it;
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    double vel_mult = 1000.0;

    std::vector<bool> printed(dof_handler.n_dofs(), false);

    QTrapez<dim> trapez_formula;
    std::vector<double> cell_head(dofs_per_cell);
    FEValues<dim> fe_values_trapez(fe, trapez_formula, update_values);

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell != endc; ++cell){
        if (cell->is_locally_owned()){



            fe_values_trapez.reinit(cell);
            fe_values_trapez.get_function_values(locally_relevant_solution, cell_head);

            cell->get_dof_indices (local_dof_indices);
            for (unsigned int ii = 0; ii < local_dof_indices.size(); ++ii){
                if (!printed[local_dof_indices[ii]]){
                    Point<dim> p = cell->vertex(ii);
                    it = VelocityMap.find(local_dof_indices[ii]);
                    if (it != VelocityMap.end()){

                        double temp = calculate_step(cell, it->second.av_vel);

                        vel_file << local_dof_indices[ii] << " " << std::fixed << std::setprecision(2)
                                 << p[0] << " "
                                 << p[1] << " ";
                        if (dim == 3){
                            vel_file << p[2] << " ";
                        }
                        vel_file << std::fixed << std::setprecision(5) << it->second.av_vel[0]*vel_mult << " "
                                 << it->second.av_vel[1]*vel_mult << " ";
                        if (dim == 3){
                            vel_file << it->second.av_vel[2]*vel_mult << " ";
                        }
                        vel_file << std::fixed << std::setprecision(4) << cell_head[ii] << std::endl;
                        printed[local_dof_indices[ii]] = true;
                    }
                    else{
                        std::cout << "That can't be happening" << std::endl;
                    }
                }
            }
        }
    }
    vel_file.close();
}

#endif // PARTICLE_TRACKING_H
