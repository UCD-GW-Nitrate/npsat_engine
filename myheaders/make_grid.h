#ifndef MAKE_GRID_H
#define MAKE_GRID_H

#include <deal.II/grid/tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include "dsimstructs.h"

namespace AquiferGrid{
    using namespace dealii;

    //===================================================
    // DECLARATIONS
    //===================================================

    /*!
    * \brief The GridGenerator class provides the required functionality to generate a grid based on the
    * user input
    */
    template <int dim>
    class GridGenerator{
    public:
        /*!
        * \brief GridGenerator The constructor simply sets the input
        * \param g class with the data for generating the geometry domain
        */
        GridGenerator(AquiferProperties<dim>& g);

        /*!
        * \brief make_grid This is the main entry of this class.
        * \param triangulation will hold the output mesh
        */
        void make_grid(parallel::distributed::Triangulation<dim>& triangulation);

        //! Print the initial grid to file
        void print_Initial_grid(std::string filename);

    private:
        //! Parameters that control the shape of the domain and all properties
        AquiferProperties<dim> geom_param;

        //! This method is called when the aquifer is a box. Creates a box domain with dimensions
        //! indicated by #AquiferProperties::left_lower_point and #AquiferProperties::Length.
        //! However the #dim-1 dimension is set to 0 with lenght 1. The actual elevation is set
        //! afterwards using the inputs from the #AquiferProperties::top_elevation and
        //! #AquiferProperties::bottom_elevation functions.
        void make_box(parallel::distributed::Triangulation<dim>& triangulation);

        //! Reads the mesh file and creates a 2D triangulation
        bool read_2D_grid(Triangulation<dim>& triangulation);

        //! Converts the triangulation to a parallel version
        void convert_to_parallel(Triangulation<dim>& tria3D, parallel::distributed::Triangulation<dim>& triangulation);

        //! Assigns boundary ids
        void assign_default_boundary_id(parallel::distributed::Triangulation<dim>& triangulation);


    };

    //===================================================
    // IMPLEMENTATION
    //===================================================
    template<int dim>
    GridGenerator<dim>::GridGenerator(AquiferProperties<dim>& g)
        :
        geom_param(g)
    {}

    template <int dim>
    void GridGenerator<dim>::make_box(parallel::distributed::Triangulation<dim>& triangulation){
        Point<dim> left_bottom, right_top;
        std::vector<unsigned int>	n_cells;
        for (unsigned int i = 0; i < dim; ++i){
            left_bottom[i] = geom_param.left_lower_point[i];
            right_top[i] = left_bottom[i] + geom_param.Length[i];
            n_cells.push_back(geom_param.Nxyz[i]);
        }
        if (geom_param.vert_discr.size() > 1)
            n_cells[dim-1] = geom_param.vert_discr.size()-1;

        left_bottom[dim-1] = 0;
        right_top[dim-1] = 1;

        dealii::GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                          n_cells,
                                                          left_bottom,
                                                          right_top,
                                                          true);
    }

    template <int dim>
    void GridGenerator<dim>::make_grid(parallel::distributed::Triangulation<dim>& triangulation){
        if (geom_param.geomtype == "BOX"){
            make_box((triangulation));

        }
        else if (geom_param.geomtype == "FILE"){
            if (dim !=3 ){
                std::cerr << "You cannot use the FILE geometry with problem dimension other that 3D" << std::endl;
            }else{
                Triangulation<dim-1> tria2D;
                Triangulation<dim> tria3D;
            }
        }

        std::ofstream out ("test_tria.vtk");
        GridOut grid_out;
        grid_out.write_ucd(triangulation, out);
    }

}



#endif // MAKE_GRID_H
