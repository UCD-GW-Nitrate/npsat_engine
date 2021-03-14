#ifndef WELLS_H
#define WELLS_H

#include <deal.II/base/point.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/fe/fe_q.h>

#include "helper_functions.h"
//#include "cgal_functions.h"
#include "nanoflann_structures.h"
#include "my_functions.h"
#include "mpi_help.h"
#include "streamlines.h"


using namespace dealii;

/*!
* \brief The Well class contains data structures to hold information for one well.
*/
template <int dim>
class Well{
public:
    Well();
    ~Well(){
        Q_cell.clear();
        L_cell.clear();
        well_cells.clear();
        K_cell.clear();
        owned.clear();
        mid_point.clear();
    };

    //! This is the top point of the well screen.
    Point<dim> top;

    //! This is the bottom point of the well screen
    Point <dim> bottom;

    //! The pumping rate of the well
    double Qtot;

    //! The well id
    int well_id;

    //! A vector that stores the cell elements that the screen length intersects
    std::vector<typename DoFHandler<dim>::active_cell_iterator > well_cells;

    //! A vector that stores the well rates that correspond to each cell
    std::vector<double > Q_cell;

    //! A vector that stores the length of the screen that is contained within each cell
    std::vector<double > L_cell;

    //! A vector that stores the hydraulic conductivity of each cell
    std::vector<double>  K_cell;

    //! A vector that is set to true if the cell is locally owned by the current processor
    std::vector<bool> 	 owned;

    //! A vector that stores the middle point of the screen that is contained within each cell
    std::vector<Point<dim> > mid_point;

    /*!
     * \brief distribute_particles distributes particles around the well screen
     * \param particles is the output vector of Point<dim> with the location particles
     * \param Nppl is the number of particles per layers
     * \param Nlay is the number of layers
     * \param radius is the distance to place the particles around the well
     */
    void distribute_particles(std::vector<Point<dim> >& particles,
                              int Nppl, int Nlay, double radius,
                              wellParticleDistributionType partDirtibType);

};

template <int dim>
Well<dim>::Well(){}

template <int dim>
void Well<dim>::distribute_particles(std::vector<Point<dim> >& particles,
                                     int Nppl, int Nlay, double radius,
                                     wellParticleDistributionType partDirtibType){
    if (dim == 2){
        std::vector<double> zval = linspace(bottom[dim-1], top[dim-1],Nlay);
        std::vector<double> r;
        r.push_back(radius);
        r.push_back(-radius);
        for(int i = 0; i < Nlay; ++i){
            for(unsigned int j = 0; j < r.size(); ++j){
                Point<dim>temp;
                temp[0] = top[0] + r[j];
                temp[1] = zval[i];
                particles.push_back(temp);
            }
        }

    }
    else if (dim == 3){

        switch (partDirtibType) {
        case wellParticleDistributionType::LAYERED:
        {
            std::vector<double> zval = linspace(bottom[dim-1], top[dim-1],Nlay);
            double rads = (2.0*numbers::PI)/Nppl;
            std::vector<double> rads1 = linspace(0, 2.0*numbers::PI,Nlay);
            std::vector<std::vector<double> > radpos;
            for (unsigned int i = 0; i < static_cast<unsigned int>(Nlay); ++i){
                radpos.push_back(linspace(0 + rads/2.0 + rads1[i],
                                 2.0*numbers::PI - rads/2.0 + rads1[i], Nppl));
            }

            for(int i = 0; i < Nlay; ++i){
                for( int j = 0; j < Nppl; ++j){
                    Point<dim>temp;
                    temp[0] = cos(radpos[i][j])*radius + top[0];
                    temp[1] = sin(radpos[i][j])*radius + top[1];
                    temp[2] = zval[i];
                    particles.push_back(temp);
                }
            }
        }
            break;
        case wellParticleDistributionType::SPIRAL:
        {
            double maxt = 2.0 * numbers::PI * static_cast<double>(Nlay);
            double dt = 1 / (static_cast<double>(Nppl*Nlay) - 1);
            double t;
            double r = radius;
            double h = top[dim-1] - bottom[dim-1];
            for (double i = 0; i <= 1+dt/2.0; i = i + dt){
                Point<dim>temp;
                t = i * maxt;
                temp[0] = top[0] + r * cos(t);
                temp[1] = top[1] + r * sin(t);
                temp[2] = bottom[dim-1] + h * i;
                particles.push_back(temp);
            }
        }
            break;
        case wellParticleDistributionType::LAYROT:
        {
            std::vector<double> zval = linspace(bottom[dim-1], top[dim-1],Nlay);
            double t_lay = 0.0; // parametric offset within the layer
            double t_rot = 0.0; //parametric offset between layers
            double dt_lay = 2.0 * numbers::PI / static_cast<double>(Nppl);
            double dt_rot = 2.0 * numbers::PI / static_cast<double>(Nlay);
            for (int i = 0; i < Nlay; ++i){
                t_lay = 0.0;
                for (int j = 0; j < Nppl; ++j){
                    Point<dim>temp;
                    double t = t_lay + t_rot;
                    temp[0] = top[0] + radius * cos(t);
                    temp[1] = top[1] + radius * sin(t);
                    temp[2] = zval[i];
                    particles.push_back(temp);
                    t_lay += dt_lay;
                }
                t_rot += dt_rot;
            }
        }
            break;
        default:
            distribute_particles(particles, Nppl, Nlay, radius, wellParticleDistributionType::SPIRAL);

        }

    }
}

template <int dim>
class Well_Set{
public:

    //! The constructor initialize some helper data structures
    Well_Set();

    //! A vector of SourceSinks::#Well wells which containt the well info
    //std::vector<Well<dim>> wells;
    std::map<int, Well<dim> > wells;

    //! The total number of wells
    int Nwells;

    //! This is a helper triangulation which consist of one element and gets initialized by the contructor
    //! It is use to provide some needed functionality at dim-1
    //Triangulation<dim-1> tria;

    //! This is a structure that contains the XY locations of the wells into a structure provided by CGAL.
    //! It is used for fast searching
    // PointSet2 WellsXY;

    //! This is a structure that is used by CGAL during fast searching and it is used to obtain the id of the included wells
    // std::vector< std::pair<ine_Point2,int> > wellxy;

    PointIdCloud WellsAsCloud;
    std::shared_ptr<pointid_kd_tree> wellIndex;

    //! Given a 3D cell sets up the 2D Well_Set::#tria cell.
    void setup_cell(typename DoFHandler<dim>::active_cell_iterator Cell3D, Triangulation<dim-1> &tria);

    //! Given a 3D cell sets up the 2D WELLS::#tria cell.
    void setup_cell(typename parallel::distributed::Triangulation<dim>::active_cell_iterator Cell3D, Triangulation<dim-1> &tria);

    /*!
     * \brief read_wells Reads the wells and initializes the CGAL structures WELLS#WellsXY and WELLS#wellxy.
     *
     * The format of the well input file is the following:
     *
     * WELLS#Nwells the number of wells
     *
     * Repeat the followin line WELLS#Nwells times
     *
     * X Y Ztop Zbot Q
     *
     * where:
     *
     * -X and Y are the x and y coordinates of the well. In 2D just set y = 0
     *
     * -Ztop and Zbot are the elevations of the top and bottom of the screen
     *
     * -Q is the pumping(Negative) or recharging(positive) well rate
     *
     * \param base_filename is the name of the well input file
     * \return true if there are no errors during reading
     */
    bool read_wells(std::string base_filename);

    /*!
    * \brief add_contributions add the contributions from the wells to the right hand size vector
    * \param system_rhs this is the right hand size vector
    * \param dof_handler
    * \param fe
    * \param constraints
    * \param hydraulic_conductivity
    * \param mpi_communicator
    */
    void add_contributions(TrilinosWrappers::MPI::Vector& system_rhs,
                            const DoFHandler<dim>& dof_handler,
                            const FE_Q<dim>& fe,
                            const AffineConstraints<double>& constraints,
                            const MyTensorFunction<dim>& hydraulic_conductivity,
                            MPI_Comm mpi_communicator);

    /*!
     * \brief flag_cells_for_refinement flags for refinement the elements that are intersected by a well
     * \param triangulation
     */
    void flag_cells_for_refinement(parallel::distributed::Triangulation<dim>& 	triangulation);

    /*!
     * \brief distribute_particles Distributes the particles around all wells of the domain
     *
     * In practice this function loops through the wells and calls the Well#distribute_particles method.
     * \param Streamlines Is the output vector of streamlines with the initial particle positions.
     * \param Nppl
     * \param Nlay
     * \param radius
     *
     * \sa Well#distribute_particles method for explanation of the inputs
     */
    void distribute_particles(std::vector<Streamline<dim>>& Streamlines,
                              int Nppl, int Nlay, double radius,
                              wellParticleDistributionType partDirtibType);


    //! Prints the well info. It is used for debuging.
    void print_wells();

    //! well multiplier
    double well_multiplier = 1.0;

    bool search4NearbyWells(double x, double y, double searchRadius,
                            std::vector<int>& ids);
};

template <int dim>
Well_Set<dim>::Well_Set(){
    Nwells = 0;
}


template <int dim>
void Well_Set<dim>::setup_cell(typename DoFHandler<dim>::active_cell_iterator Cell3D, Triangulation<dim-1>& tria){
    typename Triangulation<dim-1>::active_cell_iterator
    cell = tria.begin_active();
    for (unsigned int i = 0; i < GeometryInfo<dim-1>::vertices_per_cell; ++i){
        Point<dim-1> &v = cell->vertex(i);
        if (dim == 2){
            v[0] = Cell3D->face(2)->vertex(i)[0];
        }
        else if (dim == 3){
            v[0] = Cell3D->face(4)->vertex(i)[0];
            v[1] = Cell3D->face(4)->vertex(i)[1];
        }

    }

}

template <int dim>
void Well_Set<dim>::setup_cell(typename parallel::distributed::Triangulation<dim>::active_cell_iterator Cell3D, Triangulation<dim-1>& tria){
    typename Triangulation<dim-1>::active_cell_iterator
    cell = tria.begin_active();
    for (unsigned int i=0; i < GeometryInfo<dim-1>::vertices_per_cell; ++i){
        Point<dim-1> &v = cell->vertex(i);
        if (dim == 2){
            v[0] = Cell3D->face(2)->vertex(i)[0];
        }
        else if (dim == 3){
            v[0] = Cell3D->face(4)->vertex(i)[0];
            v[1] = Cell3D->face(4)->vertex(i)[1];
        }
    }
}

template <int dim>
bool Well_Set<dim>::read_wells(std::string base_filename)
{
    std::ifstream  datafile(base_filename.c_str());
    if (!datafile.good()){
        std::cout << "Can't open the file" << base_filename << std::endl;
        return false;
    }
    else{
        char buffer[512];
        double Xcoord, Ycoord, top, bot, Q;
        datafile.getline(buffer,512);
        std::istringstream inp1(buffer);
        inp1 >> Nwells;
        //wells.resize(Nwells);
        for (int i = 0; i < Nwells; i++){
            PointId wp;
            datafile.getline(buffer,512);
            std::istringstream inp(buffer);
            inp >> Xcoord;
            wp.x = Xcoord;
            wp.id = i;
            if (dim == 2){
                Ycoord = 0;
                wp.y = 0;
            }
            else if (dim == 3){
                inp >> Ycoord;
                wp.y = Ycoord;
            }
            WellsAsCloud.pts.push_back(wp);

            inp >> top;
            inp >> bot;
            inp >> Q;

            if (top - bot <= 0){
                std::cerr << "Well " << i << " has " << top-bot << " screen length" << std::endl;
            }

            Point<dim> p_top;
            Point<dim> p_bot;
            if (dim == 2){
                p_top[0] = Xcoord; p_top[1] = top;
                p_bot[0] = Xcoord; p_bot[1] = bot;
            }
            else if (dim == 3){
                p_top[0] = Xcoord; p_top[1] = Ycoord; p_top[2] = top;
                p_bot[0] = Xcoord; p_bot[1] = Ycoord; p_bot[2] = bot;
            }
            Well<dim> w;
            w.top = p_top;
            w.bottom = p_bot;
            w.Qtot = Q;
            w.well_id = i;
            wells.insert(std::pair<int, Well<dim>>(i,w));
            //wells[i].top = p_top;
            //wells[i].bottom = p_bot;
            //wells[i].Qtot = Q;
            //wells[i].well_id = i;
        }
        wellIndex = std::shared_ptr<pointid_kd_tree>(new pointid_kd_tree(
                2, WellsAsCloud,
                nanoflann::KDTreeSingleIndexAdaptorParams(10)));
        wellIndex->buildIndex();
        return true;
    }
}

template <int dim>
void Well_Set<dim>::flag_cells_for_refinement(parallel::distributed::Triangulation<dim>& triangulation){
    Triangulation<dim-1> tria;
    initTria<dim>(tria);

    const MappingQ1<dim-1> mapping2D;
    Point<dim-1> well_point_2d;

    typename std::map<int, Well<dim>>::iterator itw;
    typename parallel::distributed::Triangulation<dim>::active_cell_iterator
    cell = triangulation.begin_active(),
    endc = triangulation.end();
    for (; cell!=endc; ++cell){
        if (cell->is_locally_owned()){
            std::vector<int> well_id_in_cell;
            std::vector<double> xp; std::vector<double> yp;
            if (dim == 2){
                xp.push_back(cell->face(2)->vertex(0)[0]);yp.push_back(0);
                xp.push_back(cell->face(2)->vertex(1)[0]);yp.push_back(0);
            }
            else if (dim == 3){
                xp.push_back(cell->face(4)->vertex(0)[0]); yp.push_back(cell->face(4)->vertex(0)[1]);
                xp.push_back(cell->face(4)->vertex(1)[0]); yp.push_back(cell->face(4)->vertex(1)[1]);
                xp.push_back(cell->face(4)->vertex(3)[0]); yp.push_back(cell->face(4)->vertex(3)[1]);
                xp.push_back(cell->face(4)->vertex(2)[0]); yp.push_back(cell->face(4)->vertex(2)[1]);
            }
            //bool are_wells = get_point_ids_in_set(WellsXY, xp, yp, well_id_in_cell);
            Point<dim> bc = cell->barycenter();
            bool are_wells = search4NearbyWells(bc[0], bc[1], cell->diameter(), well_id_in_cell);

            if (!are_wells)
                continue;

            setup_cell(cell,tria);
            typename Triangulation<dim-1>::active_cell_iterator cell2D = tria.begin_active();


            for (unsigned int iw = 0; iw < well_id_in_cell.size(); ++iw){
                itw = wells.find(well_id_in_cell[iw]);
                //int i = well_id_in_cell[iw];
                well_point_2d[0] = itw->second.top[0];
                if (dim == 3)
                    well_point_2d[1] = itw->second.top[1];

                Point<dim-1> p_unit2D;
                bool mapping_done = try_mapping<dim-1>(well_point_2d, p_unit2D, cell2D, mapping2D);
                if (!mapping_done)
                    continue;

                Point<dim> p_unit_top, p_unit_bot;
                for (unsigned int ii = 0; ii < dim-1; ++ii){
                    p_unit_top[ii] = p_unit2D[ii];
                    p_unit_bot[ii] = p_unit2D[ii];
                }
                p_unit_top[dim-1] = 1;
                p_unit_bot[dim-1] = 0;
                double z_top = 0;
                double z_bot = 0;

                for (unsigned int j = 0; j < GeometryInfo<dim>::vertices_per_cell ; j++){
                    z_top = z_top +  GeometryInfo<dim>::d_linear_shape_function(p_unit_top, j)*cell->vertex(j)[dim-1];
                    z_bot = z_bot +  GeometryInfo<dim>::d_linear_shape_function(p_unit_bot, j)*cell->vertex(j)[dim-1];
                }

                double well_TPF = itw->second.top[dim-1];
                double well_BPF = itw->second.bottom[dim-1];
                bool cell_flaged = false;
                if  (well_BPF < z_bot && well_TPF > z_top){
                    cell->set_refine_flag ();
                    cell_flaged = true;
                }
                if  (well_BPF > z_bot && well_TPF < z_top){
                    cell->set_refine_flag ();
                    cell_flaged = true;
                }
                if (well_BPF < z_bot && well_TPF < z_top && well_TPF > z_bot){
                    cell->set_refine_flag ();
                    cell_flaged = true;
                }
                if (well_BPF > z_bot && well_BPF < z_top && well_TPF > z_top){
                    cell->set_refine_flag ();
                    cell_flaged = true;
                }
                if (cell_flaged)
                    break;
            }
        }
    }
}

template <int dim>
void Well_Set<dim>::add_contributions(TrilinosWrappers::MPI::Vector& system_rhs,
                                      const DoFHandler<dim>& dof_handler,
                                      const FE_Q<dim>& fe,
                                      const AffineConstraints<double>& constraints,
                                      const MyTensorFunction<dim>& hydraulic_conductivity,
                                      MPI_Comm mpi_communicator){

    if (Nwells == 0)
        return;

    int my_rank = Utilities::MPI::this_mpi_process(mpi_communicator);
    int n_proc = Utilities::MPI::n_mpi_processes(mpi_communicator);
    Triangulation<dim-1> tria;
    initTria<dim>(tria);
    const MappingQ1<dim-1> mapping2D;
    const MappingQ1<dim> mapping;
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    Vector<double> cell_rhs_wells (dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    std::vector<std::vector<int> > 		well_id(n_proc);
    std::vector<std::vector<double> > 	cell_length(n_proc);
    std::vector<std::vector<double> > 	cell_cond(n_proc);

    double Qwell_total = 0;

    Point<dim-1> well_point_2d;
    typename std::map<int, Well<dim>>::iterator itw;
    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell){
        if (cell->is_locally_owned()){
            std::vector<int> well_id_in_cell;
            std::vector<double> xp; std::vector<double> yp;
            //double xm,ym;
            if (dim == 2){
                xp.push_back(cell->face(2)->vertex(0)[0]);yp.push_back(0);
                xp.push_back(cell->face(2)->vertex(1)[0]);yp.push_back(0);
            }
            else if (dim == 3){
                xp.push_back(cell->face(4)->vertex(0)[0]); yp.push_back(cell->face(4)->vertex(0)[1]);
                xp.push_back(cell->face(4)->vertex(1)[0]); yp.push_back(cell->face(4)->vertex(1)[1]);
                xp.push_back(cell->face(4)->vertex(3)[0]); yp.push_back(cell->face(4)->vertex(3)[1]);
                xp.push_back(cell->face(4)->vertex(2)[0]); yp.push_back(cell->face(4)->vertex(2)[1]);
                //xm = 0;ym = 0;
                //for (unsigned int kk = 0; kk < xp.size(); kk++){
                //    xm += xp[kk];
                //    ym += yp[kk];
                //}
                //xm = xm/static_cast<double>(xp.size());
                //ym = ym/static_cast<double>(xp.size());
            }
            //std::cout << xm << " " << ym << std::endl;
            //double dst = std::sqrt(std::pow(xm - 33474.74070, 2.0) + std::pow(ym - 1027.05735, 2.0));
            //std::cout << dst << std::endl;
            //if (dst < 1.0){
            //    std::cout << "You should really check this cell" << std::endl;
            //    int a = 0;
            //    dummy_function(true,a);
            //}
            Point<dim> bc = cell->barycenter();
            if (dim == 3)
                bc[2] = 0;
            else if (dim == 2)
                bc[1] = 0;

            bool are_wells = search4NearbyWells(bc[0], bc[1], cell->diameter(), well_id_in_cell);

            //bool are_wells = get_point_ids_in_set(WellsXY, xp, yp, well_id_in_cell);
            if (!are_wells)
                continue;

            setup_cell(cell,tria);
            typename Triangulation<dim-1>::active_cell_iterator cell2D = tria.begin_active();

            for (unsigned int iw = 0; iw < well_id_in_cell.size(); iw++){
                itw = wells.find(well_id_in_cell[iw]);
                //int i = well_id_in_cell[iw];
                /*
                bool dgb_print = false;
                if (i == 714){
                    std::cout << "Debug this well" << std::endl;
                    int cccnt=1;
                    int ff = 0;
                    cccnt++;
                    ff = cccnt;
                    well_point_2d[0] = wells[i].top[0];
                    dgb_print = true;
                }
                */

                well_point_2d[0] = itw->second.top[0];
                if (dim == 3)
                    well_point_2d[1] = itw->second.top[1];

                bool is_in_cell = cell2D->point_inside(well_point_2d);

                if (is_in_cell == true){
                    //std::cout << "Well id " << itw->first << std::endl;
                    // If the point is inside the cell find its unit coordinates
                    // and the top and bottom of the cell at the well point
                    Point<dim-1> p_unit2D;
                    bool mapping_done = try_mapping<dim-1>(well_point_2d, p_unit2D, cell2D, mapping2D);
                    if (!mapping_done)
                        continue;

                    Point<dim> p_unit_top, p_unit_bot;
                    for (unsigned int ii = 0; ii < dim-1; ++ii){
                        p_unit_top[ii] = p_unit2D[ii];
                        p_unit_bot[ii] = p_unit2D[ii];
                    }
                    p_unit_top[dim-1] = 1;
                    p_unit_bot[dim-1] = 0;
                    double z_top = 0;
                    double z_bot = 0;
                    for (unsigned int j = 0; j < dofs_per_cell; j++){
                        z_top = z_top +  GeometryInfo<dim>::d_linear_shape_function(p_unit_top, j)*cell->vertex(j)[dim-1];
                        z_bot = z_bot +  GeometryInfo<dim>::d_linear_shape_function(p_unit_bot, j)*cell->vertex(j)[dim-1];
                    }
                    double well_TPF = itw->second.top[dim-1];
                    double well_BPF = itw->second.bottom[dim-1];
                    bool add_this_cell = false;
                    double segment_length;
                    Point<dim> p_mid;
                    for (unsigned int ii = 0; ii < dim-1; ++ii)
                        p_mid(ii) = well_point_2d[ii];
                    double K;
                    // case 1
                    if  (well_BPF < z_bot && well_TPF > z_top){
                        // the well screen fully penetrates this cell.
                        p_mid(dim-1) = (z_top - z_bot)/2.0 + z_bot;
                        Tensor<2,dim> K_tensor = hydraulic_conductivity.value(p_mid);
                        K = K_tensor[0][0];
                        segment_length = z_top - z_bot;
                        add_this_cell = true;
                    }
                    //case 2
                    if  (well_BPF > z_bot && well_TPF < z_top){
                        // the well screen is all within this cell.
                        p_mid(dim-1) = (well_TPF - well_BPF)/2.0 + well_BPF;
                        Tensor<2,dim> K_tensor = hydraulic_conductivity.value(p_mid);
                        K = K_tensor[0][0];
                        segment_length = well_TPF - well_BPF;
                        add_this_cell = true;
                    }
                    //case 3
                    if (well_BPF < z_bot && well_TPF < z_top && well_TPF > z_bot){
                        // the bottom of the screen is below the cell and the top is in the cell
                        p_mid(dim-1) = (well_TPF - z_bot)/2.0 + z_bot;
                        Tensor<2,dim> K_tensor = hydraulic_conductivity.value(p_mid);
                        K = K_tensor[0][0];
                        segment_length = well_TPF - z_bot;
                        add_this_cell = true;
                    }
                    //case 4
                    if (well_BPF > z_bot && well_BPF < z_top && well_TPF > z_top){
                        // the bottom of the screen is in the cell and the top of the screen is above the cell
                        p_mid(dim-1) = (z_top - well_BPF)/2.0 + well_BPF;
                        Tensor<2,dim> K_tensor = hydraulic_conductivity.value(p_mid);
                        K = K_tensor[0][0];
                        segment_length = z_top - well_BPF;
                        add_this_cell = true;
                    }
                    //case 5
                   if (well_BPF > z_top && cell->face(2*dim-1)->at_boundary()){ //2*dim-1 returns the top face
                        // The well is above of the water table. Still we want the first layer to take out water
                        //std::cout << "Zt:" << z_top << ", Zb:" << z_bot << ", Wb:" << well_BPF << std::endl;
                        //std::cout << "The bottome of the well is above the top cell" << std::endl;
                        p_mid(dim-1) = (z_top - z_bot)/2 + z_bot;
                        Tensor<2,dim> K_tensor = hydraulic_conductivity.value(p_mid);
                        K = K_tensor[0][0];
                        segment_length = z_top - z_bot;
                        add_this_cell = true;
                    }
                    if (add_this_cell){
                        //int nn = itw->second.well_cells.size() + 1;
                        //if (dgb_print){
                            //std::cout << itw->second.well_cells.size() << std::endl;
                            //std::cout << cell->center() << std::endl;
                        //}

                        //std::cout << "Well id " << itw->first << std::endl;
                        //wells[i].mid_point.resize(nn);
                        //wells[i].mid_point[nn-1] = p_mid;
                        itw->second.L_cell.push_back(segment_length);
                        itw->second.K_cell.push_back(K);
                        itw->second.owned.push_back(true);
                        itw->second.mid_point.push_back(p_mid);
                        //itw->second.well_cells.resize(itw->second.well_cells.size()+1);
                        //itw->second.well_cells[itw->second.well_cells.size()-1] = cell;
                        itw->second.well_cells.push_back(cell);

                        well_id[my_rank].push_back(itw->first);
                        cell_length[my_rank].push_back(segment_length);
                        cell_cond[my_rank].push_back(K);
                        /*
                        if (dgb_print){
                            std::cout << wells[i].well_cells[wells[i].well_cells.size()-1]->center() << std::endl;
                            std::cout << wells[i].mid_point[wells[i].mid_point.size()-1] << std::endl;;
                        }
                        */
                    }
                }
            }
        }
    }

    // So far we have loop through the cells that are locally owned and each processor
    // has stored into the wells class the info about K and segment length.
    // Next all processors will send everything to all processors so that
    // each processor can weight the pumping independently

    MPI_Barrier(mpi_communicator);
    std::vector<int> N_data;
    // First send the number of cells that each processor has calculated the required data
    Send_receive_size(cell_cond[my_rank].size(), n_proc, N_data, mpi_communicator);
    // Next send the data
    Sent_receive_data<int>(well_id, N_data, my_rank, mpi_communicator, MPI_INT);
    Sent_receive_data<double>(cell_cond, N_data, my_rank, mpi_communicator, MPI_DOUBLE);
    Sent_receive_data<double>(cell_length, N_data, my_rank, mpi_communicator, MPI_DOUBLE);
    MPI_Barrier(mpi_communicator);

    // Each processor will add the data to its wells variable
    // But these data are not owned
    for (int i = 0; i < n_proc; ++i){
        if (i == my_rank)// This has already the data
            continue;

        for (int j = 0; j < N_data[i]; ++j){
            itw = wells.find(well_id[i][j]);
            itw->second.K_cell.push_back(cell_cond[i][j]);
            itw->second.L_cell.push_back(cell_length[i][j]);
            itw->second.owned.push_back(false);
            typename DoFHandler<dim>::active_cell_iterator dummy_cell;
            itw->second.well_cells.push_back(dummy_cell);
            Point<dim> dummy_point;
            itw->second.mid_point.push_back(dummy_point);
        }
    }
    // Now each processor will loop though the wells.
    // if there are well with owned == true we will distribute the pumping rate to all cells
    // even to those that have owned == false but we will assign rates only to the owned == true
    //int Cnt_wellsQ = 0;
    //for (int i = 0; i < Nwells; ++i){
    for (itw = wells.begin(); itw != wells.end(); itw++){
        //double Qwell_temp = 0;
        //std::cout << "well-> i: " << i << " wid:  " << wells[i].well_id << " pnt: " << wells[i].bottom << std::endl;
        bool any_true = false;
        int N_cells = itw->second.owned.size();
        for (int j = 0; j < N_cells; ++j){
            if (itw->second.owned[j] == true){
                any_true = true;
                break;
            }
        }
        if (any_true){
            std::vector<double> wK(N_cells);
            std::vector<double> wL(N_cells);
            std::vector<double> wKL(N_cells);
            double sum_K, sum_L, sum_KL;
            sum_K = 0; sum_L = 0; sum_KL = 0;
            for (int j = 0; j < N_cells; ++j){
                wK[j] = itw->second.K_cell[j];
                wL[j] = itw->second.L_cell[j];
                sum_K += wK[j];
                sum_L += wL[j];
            }
            for (int j = 0; j < N_cells; ++j){
                wKL[j] = (wK[j]/sum_K) * (wL[j]/sum_L);
                sum_KL += wKL[j];
            }
            for (int j = 0; j < N_cells; ++j){
                if (itw->second.owned[j] == true){
                    cell_rhs_wells = 0;
                    // convert the mid point to unit point
                    Point<dim> p_unit_mid_point;
                    bool mapping_done = try_mapping<dim>(itw->second.mid_point[j], p_unit_mid_point, itw->second.well_cells[j],mapping);
                    if (mapping_done){
                        Quadrature<dim> temp_quadrature(p_unit_mid_point);
                        FEValues<dim> fe_values_temp(fe, temp_quadrature, update_values | update_quadrature_points);
                        fe_values_temp.reinit(itw->second.well_cells[j]);
                        for (unsigned int q_point = 0; q_point < temp_quadrature.size(); ++q_point){
                            for (unsigned int ii = 0; ii < dofs_per_cell; ++ii){
                                double Q_temp = (wKL[j]/sum_KL)*itw->second.Qtot*fe_values_temp.shape_value(ii,q_point);
                                //std::cout << i << " : " << Q_temp << std::endl;
                                cell_rhs_wells(ii) += Q_temp * well_multiplier;
                                Qwell_total += Q_temp * well_multiplier;
                                //Qwell_temp += Q_temp;
                            }
                        }
                        //std::cout << "Qtot: " << Qwell_total << ", iter: " << i << std::endl;
                        itw->second.well_cells[j]->get_dof_indices (local_dof_indices);
                        constraints.distribute_local_to_global(cell_rhs_wells,
                                                               local_dof_indices,
                                                               system_rhs);
                    }
                }
            }
        }

        //if (std::abs(Qwell_temp) > 0){
        //    Cnt_wellsQ++;
        //}
        //else{
        //    std::cout << "Well " << wells[i].well_id << " at " << wells[i].top << " has zero Q" << std::endl;
        //}

    }
    //std::cout << "Wells with pumping rates: " << Cnt_wellsQ << std::endl;
    MPI_Barrier(mpi_communicator);
    sum_scalar<double>(Qwell_total,n_proc, mpi_communicator, MPI_DOUBLE);
    if (my_rank == 0)
        std::cout << "\t QWELLS: [" << Qwell_total << "]" << std::endl;
    MPI_Barrier(mpi_communicator);
    //for (int i = 0; i < Nwells; ++i){
    for (itw = wells.begin(); itw != wells.end(); itw++){
        itw->second.Q_cell.clear();
        itw->second.L_cell.clear();
        itw->second.K_cell.clear();
        itw->second.owned.clear();
        itw->second.mid_point.clear();
        itw->second.well_cells.clear();
    }
}


template <int dim>
void Well_Set<dim>::distribute_particles(std::vector<Streamline<dim>> &Streamlines,
                                         int Nppl, int Nlay, double radius,
                                         wellParticleDistributionType partDirtibType){
    typename std::map<int, Well<dim>>::iterator itw;
    for (itw = wells.begin(); itw != wells.end(); itw++){
        std::vector<Point<dim>> particles;
        itw->second.distribute_particles(particles, Nppl, Nlay, radius, partDirtibType);
        for (unsigned int j = 0; j < particles.size(); ++j){
            Streamlines.push_back(Streamline<dim>(itw->first, j, particles[j]));
        }
    }
}

template<int dim>
bool Well_Set<dim>::search4NearbyWells(double x, double y, double searchRadius,
                                       std::vector<int>& ids){
    searchRadius = searchRadius * searchRadius; // Nanoflann requires the square distance
    const double query_pt[2] = {x, y};
    std::vector<std::pair<size_t,double> > ret_matches;
    nanoflann::SearchParams params;
    params.sorted = false;
    const int nMatches = wellIndex->radiusSearch(&query_pt[0], searchRadius, ret_matches, params);
    for (int i = 0; i < nMatches; ++i){
        ids.push_back(ret_matches[i].first);
    }
    if (ids.size() > 0)
        return true;
    else
        return false;
}


#endif // WELLS_H
