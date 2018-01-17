#ifndef DSIMSTRUCTS_H
#define DSIMSTRUCTS_H

#include <iostream>
#include <vector>

#include "interpinterface.h"

/*!
 * \brief A struct to hold the data for the aquifer Properties.
 *
 *
 */
template<int dim>
struct AquiferProperties{
public:
    // Geometry
    //! The geometry type of the aquifer. This can be either a BOX or FILE.
    //! In case of BOX the AquiferProperties#Length, AquiferProperties#NXYZ and AquiferProperties#left_lower_point
    //! have to be set. In case of FILE only the AquiferProperties#input_mesh_file is needed.
    std::string                     geomtype;

    //! The length of the aquifer. This is a vector <x,y,z> with the dimensions of the BOX type aquifer.
    std::vector<double>             Length;

    //! Defines the initial discretization of the aquifer. Note that the z value that corresponds to
    //! the vertical discretization maybe overwritten by the AquiferProperties#vert_discr
    std::vector<int>                Nxyz;

    //! This is the left lower point of the Aquifer
    std::vector<double>             left_lower_point;

    //! This will hold the name of the user defined mesh file
    std::string                     input_mesh_file;

    //! The number of initial refinements before the simulation
    unsigned int                    N_init_refinement;

    //! The number of initial refinements around the well screens
    unsigned int                    N_well_refinement;

    //! The number of initial refinements around the streams
    unsigned int                    N_streams_refinement;

    //! A vector that defines the vertical discretization of the aquifer. The size of the vector defines the number of the layers,
    //! while the distribution is dictated by the values. The values should range between 0 and 1. For example a vector 0 0.2 0.4 0.6 0.8 1
    //! will discretize the aquifer uniformly into 5 layers. If vert_discr is empty the number of layers will be defined by the
    //! AquiferProperties#Nxyz
    std::vector<double>             vert_discr;

    //! This is a 2D interpolation function for the top elevation
    InterpInterface<dim-1>          top_elevation;

    //! This is a 2D interpolation function for the bottom elevation
    InterpInterface<dim-1>          bottom_elevation;

    //! The minimum cell dimension that is expected in the x-y plane
    //! Triangulation points closer than #xy_thres distance will be considered identical
    //! if the triangulation cells have size smaller than this value the the moving mesh routines
    //! will not work properly so choose something reasonable
    double                          xy_thres;

    //! The minimum cell height that the triangulation cells are expected to have
    //! The same principle that when chooseing #xy_thres applies to the #z_thres.
    //! For NPSAT problems where xy dimensions are greater that the z dimensions the #z_thres
    //! should be much smaller as well
    double                          z_thres;

    //! The dbg_scale_x scales the x y coordinates. This is used for debuging only when the mesh data are printed for
    //! visualization with software such as Houdini, as very large coordinates are not handled well.
    double dbg_scale_x;

    //! The dbg_scale_z scales the z coordinates. In typical groundwater aquifers the z dimension is much smaller than the
    //! x-y therefore a different scale factor in z allows to better visualize the domains
    double dbg_scale_z;
};

#endif // DSIMSTRUCTS_H
