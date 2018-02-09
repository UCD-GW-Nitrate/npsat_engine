#ifndef WELLS_H
#define WELLS_H

#include <deal.II/base/point.h>

#include "helper_functions.h"
#include "cgal_functions.h"

namespace SourceSinks {

using namespace dealii;

/*!
* \brief The Well class contains data structures to hold information for one well.
*/
template <int dim>
class Well{
public:
    Well();

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
                              int Nppl, int Nlay, double radius);
};

template <int dim>
Well<dim>::Well(){}

template <int dim>
void Well<dim>::distribute_particles(std::vector<Point<dim> >& particles,
                                     int Nppl, int Nlay, double radius){
    double rads = (2.0*numbers::PI)/Nppl;
    std::vector<double> rads1 = linspace(0, 2.0*numbers::PI,Nlay);
    std::vector<std::vector<double> > radpos;
    for (int i = 0; i < Nlay; ++i){
        radpos.push_back(linspace(0 + rads/2.0 + rads1[i],
                         2.0*numbers::PI - rads/2.0 + rads1[i], Nppl));
    }
    std::vector<double> zval = linspace(bottom[dim-1], top[dim-1],Nlay);
    for(int i = 0; i < Nlay; ++i){
        for( int j = 0; j < Nppl; ++j){
            Point<dim>temp;
            temp[0] = cos(radpos[i][j])*radius + top[0];
            if (dim == 2){
                temp[1] = zval[i];
            }
            else if (dim == 3){
                temp[1] = sin(radpos[i][j])*radius + top[1];
                temp[2] = zval[i];
            }
            particles.push_back(temp);
        }
    }
}

template <int dim>
class Well_Set{
public:

    //! The constructor initialize some helper data structures
    Well_Set();

    //! A vector of SourceSinks::#Well wells which containt the well info
    std::vector<Well> wells;

    //! The total number of wells
    int Nwells;

    //! This is a helper triangulation which consist of one element and gets initialized by the contructor
    //! It is use to provide some needed functionality at dim-1
    Triangulation<dim-1> tria;

    //! This is a structure that contains the XY locations of the wells into a structure provided by CGAL.
    //! It is used for fast searching
    PointSet2 WellsXY;

    //! This is a structure that is used by CGAL during fast searching and it is used to obtain the id of the included wells
    std::vector< std::pair<ine_Point2,unsigned> > wellxy;

    //! Given a 3D cell sets up the 2D Well_Set::#tria cell.
    void setup_cell(typename DoFHandler<dim>::active_cell_iterator Cell3D);

    //! Given a 3D cell sets up the 2D WELLS::#tria cell.
    void setup_cell(typename parallel::distributed::Triangulation<dim>::active_cell_iterator Cell3D);

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
                            const ConstraintMatrix& constraints,
                            const MyTensorFunction<dim>& hydraulic_conductivity,
                            MPI_Comm mpi_communicator);

    /*!
     * \brief flag_cells_for_refinement flags for refinement the elements that are intersected by a well
     * \param triangulation
     */
    void flag_cells_for_refinement(parallel::distributed::Triangulation<dim>& 	triangulation);

//    /*!
//     * \brief distribute_particles Distributes the particles around all wells of the domain
//     *
//     * In practice this function loops through the wells and calls the Well#distribute_particles method.
//     * \param Streamlines Is the output vector of streamlines with the initial particle positions.
//     * \param Nppl
//     * \param Nlay
//     * \param radius
//     *
//     * \sa Well#distribute_particles method for explanation of the inputs
//     */
//    void distribute_particles(std::vector<PART_TRACK::Streamline >& Streamlines,
//                              int Nppl, int Nlay, double radius);


    //! Prints the well info. It is used for debuging.
    void print_wells();
};

template <int dim>
Well_Set<dim>::Well_Set(){
    std::vector< Point<dim-1> > vertices(GeometryInfo<dim-1>::vertices_per_cell);
    std::vector< CellData<dim-1> > cells(1);

    if (dim == 2){
        vertices[0] = Point<dim-1>(0);
        vertices[1] = Point<dim-1>(1);
    }else if (dim == 3){
        vertices[0] = Point<dim-1>(0,0);
        vertices[1] = Point<dim-1>(1,0);
        vertices[2] = Point<dim-1>(0,1);
        vertices[3] = Point<dim-1>(1,1);
    }
    tria.create_triangulation(vertices, cells, SubCellData());
    Nwells = 0;
}

template <int dim>
Well_Set<dim>::setup_cell(typename DoFHandler<dim>::active_cell_iterator Cell3D){
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
Well_Set<dim>::setup_cell(typename parallel::distributed::Triangulation<dim>::active_cell_iterator Cell3D){
    typename Triangulation<dim>::active_cell_iterator
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

}

#endif // WELLS_H
