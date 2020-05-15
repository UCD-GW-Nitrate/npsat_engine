#ifndef DSIMSTRUCTS_H
#define DSIMSTRUCTS_H

#include <iostream>
#include <vector>

#include "interpinterface.h"
#include "wells.h"
#include "streams.h"
#include "my_functions.h"


struct Directories{
public:
    std::string input;
    std::string output;
};

//! Parameters related to solver
struct SolverParameters{
    //! Maximum number of iterations of the non
    int Maxiter;

    //! Maximum number of refine and solve cycles
    int NonLinearIter;

    //! Solver tolerance
    double solver_tol;

    //! Load a previously computed solution
    int load_solution;

    //! Save the final solution before particle tracking
    int save_solution;

    //! Print detailed output of the ML preconditioner
    int output_details;

    //! This multipliers are used for special cases to make the incoming equal to outcoming flows.
    //! Typically you run the model once and then if there is a discrepancy you can further refine
    //! the volumes to make sure they are equal. (if that's important for the application)
    double rch_multiplier = 1.0;

};

//! RefinementParameters is a struct with parameters that control the mesh refinements
struct RefinementParameters{

    //! Specifies the maximum number of refinements
    int MaxRefinement;

    //!  This is the top fraction of the refinement criterion. (See deal documentation)
    double TopFraction;

    //! This is the bottom fraction of the refinement criterion (See deal documentation)
    double BottomFraction;

    //! Any element with size smaller that MinElementSize will not be further refined
    double MinElementSize;
};

/*!
 * \brief The ParticleParameters struct is a container for the parameters that control the particle tracking algorithm
 */
struct ParticleParameters{
    //! This is a prefix for the output files generated from the particle tracking process.
    std::string particle_prefix;

    //! Entity_freq controls for which wells or streams the particle tracking will be triggered. The particle tracking will
    //! run every Entity_freq wells or streams. Use 1 to trigger particle tracking for all wells
    int Entity_freq;

    //! Streaml_freq controls the number of streamlines per well or stream. For example the particle tracking will be triggered
    //! every Streaml_freq. Use 1 to trigger particle tracking for all streams
    int Streaml_freq;

    //! The program creates a bounding box for each streamline. During particle tracking, the bounding box is expected to expand at every step.
    //! If the bounding box does not expand after Stuck_iter iterations the particle tracking terminates.
    int Stuck_iter;

    //! When the program runs on multiple processors it is necessary to exchange particles between processors.
    //! Outmost_iter specifies the maximum number of exchanges between the processors. This is usefull to avoid situations
    //! where a particle is stuck between two processors for example.
    int Outmost_iter;

    //! streaml_iter is the maximum number of iterations for each particle
    int streaml_iter;

    /*! There are three methods available for the particle tracking
     *   - 1 -> for Euler
     *   - 2 -> for Runge Kutta 2nd order
     *   - 3 -> for Runge Kutta 4nd order
     */
    int method;

    //! The step size is a function of the cell size. step_size actually defines an approximate number of steps to take within each cell.
    //! The algorithm divides the the element size by step_size to calculate the actual step size in length.
    double step_size;

    //! When a particle exits an element a search process is triggerd to find the cell that the particle is moving to.
    //! The algorithm creates first a list of elements that surround the current element and search if any contains the new particle position.
    //! The search_iter controls how many cells will be searched. Note that search_iter is not the number cells to be searched but rather
    //! the number of search layers around the current cell.
    int search_iter;

    //! Typically each stramline is represented by a polyline that may consist of a large number of vertices. To remove unnecessary
    //! vertices the program applies a Ramer–Douglas–Peucker algorithm (https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm)
    //!  where simplify_thres is the threshold of the simplification algorithm.
    double simplify_thres;

    //! We can turn off particle tracking by setting this parameter to 0. Any other integer value will run particle tracking
    int bDoParticleTracking;

    //! We can turn on or off the particle tracking from wells. We miay want to trace particles defined in a file
    int trace_wells;

    //! Number of particles that we execute in parallel
    int Nparallel_particles;

    //! Number of layers per well
    int Wells_N_Layers;

    //! Number of particles per layer for the wells
    int Wells_N_per_layer;

    //! The type of particle distribution
    wellParticleDistributionType PartDistribType;

    //! Well radius. The distance from the well that the particles will be realized
    double radius;

    //! Filename that defines particle locations to be traced. The format of the file is:
    //!
    //! N
    //! ID X Y Z
    //!
    //! Where N is the number of particles and ID X Y Z are repeated N times. ID is an integer identification for each particle
    std::string Particles_in_file;
};


/*!
 * \brief A struct to hold the data for the aquifer Properties.
 *
 *
 */
template<int dim>
class AquiferProperties{
public:
    AquiferProperties();

    //! Geometry
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

    //! The number ot initial refinements around the top
    unsigned int                    N_top_refinements;

    //! A vector that defines the vertical discretization of the aquifer. The size of the vector defines the number of the layers,
    //! while the distribution is dictated by the values. The values should range between 0 and 1. For example a vector 0 0.2 0.4 0.6 0.8 1
    //! will discretize the aquifer uniformly into 5 layers. If vert_discr is empty the number of layers will be defined by the
    //! AquiferProperties#Nxyz
    std::vector<double>             vert_discr;

    //! This is a 2D interpolation function for the top elevation
    InterpInterface<dim>          top_elevation;

    //! This is a 2D interpolation function for the bottom elevation
    InterpInterface<dim>          bottom_elevation;

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

    //! Saves the mesh after the initial refinement or after loading to a vtk file.
    //! The name of the file is #sim_prefix + _init_mesh.vtk
    int                             print_init_mesh;

    //! The dbg_scale_x scales the x y coordinates. This is used for debuging only when the mesh data are printed for
    //! visualization with software such as Houdini, as very large coordinates are not handled well.
    double dbg_scale_x;

    //! The dbg_scale_z scales the z coordinates. In typical groundwater aquifers the z dimension is much smaller than the
    //! x-y therefore a different scale factor in z allows to better visualize the domains
    double dbg_scale_z;

    //! A structure containing the input and output directories
    Directories                         Dirs;

    //! dirichlet_file_names is file name that contains a list of filenames with the dirichlet boundary conditions
    std::string                         dirichlet_file_names;

    //! This is a prefix name that is used as prefix for the various output files
    std::string                         sim_prefix;

    //! This is a sufix that is appended a solution is saved or loaded
    std::string                         solution_suffix;

    //! Print the solution into vtk file
    int                                 print_solution_vtk;

    //! Prints the vertices of the free surface as point cloud.
    //! THis can be read easily in houdini for visual inspections
    int                                 print_point_top_cloud;

    //! Prints the velocity field.
    int                                 print_velocity_cloud;

    //! Print boundary conditions
    int                                 print_bnd_cond;

    //! this is a constant number that the velocity is multiplied before printed.
    //! The purpose of this is to increase the accuracy of the printed velocity without
    //! increasing the number of printed digits. For example if the velocity is
    //! 0.00006134234529 then by multiply it by 1 is going to print 6 digits e.g 0.000061
    //! However if we multiply it by 10000 then in the file will be printed as 0.613423.
    //! Of course it is the users responsibility to convert the velocity back to the actual value
    //! in the application that is going to use the velocity field
    double                              multiplier_velocity_print;

    //! This is a structure that will store the hydraulic conductivity values
    std::vector<InterpInterface<dim> >	HydraulicConductivity;

    MyTensorFunction<dim>*              HK_function;

    //! This boolean variable specifies which functions of the AquiferProperties#HydraulicConductivity are actually used.
    //! For example when the aquifer is isotropic the HKuse[0] should be true while the HKuse[1] and HKuse[2] should be set to false.
    std::vector<bool>                   HKuse;

    //! A 3D interpolation function for the aquifer porosity
    InterpInterface<dim>                Porosity;

    //! This is a 2D interpolation function for the groundwater recharge
    InterpInterface<dim>              GroundwaterRecharge;

    //! This holds the information for the streams
    Streams<dim>                        streams;

    //! This holds the information about the wells in the aquifer
    Well_Set<dim>                       wells;

    //! A boolean value which is set to false if there are no wells
    bool                                have_wells;

    //! A boolean value which is set to false if there are no streams
    bool                                have_streams;

    //! Holds the Refinements parameters
    RefinementParameters                refine_param;

    //! Holds the Particle tracking parameters
    ParticleParameters                  part_param;

    //! Holds the Solver parameters
    SolverParameters                    solver_param;

    //! This is the main parameter file.
    std::string main_param_file;
};

template <int dim>
AquiferProperties<dim>::AquiferProperties(){
    have_wells = false;
    have_streams = false;
}


#endif // DSIMSTRUCTS_H
