#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/conditional_ostream.h>

#include <iostream>
#include <list>

#include "myheaders/my_macros.h"

#include "myheaders/user_input.h"
#include "myheaders/dsimstructs.h"
#include "myheaders/cgal_functions.h"
#include "myheaders/interpinterface.h"
#include "myheaders/npsat.h"
#include "myheaders/gather_data.h"


using namespace dealii;

/*! \mainpage Home
 *
 * \section intro_sec Introduction
 *
 * The Non Point Source Assessment Toolbox (NPSAT) is a modelling framework
 * that can be used to assess and evaluate the dynamic, spatio‐temporally
 * distributed linkages between nonpoint sources above a groundwater basin
 * and groundwater discharges to wells, streams, or other compliance discharge
 * surfaces (CDSs) within a groundwater basin.
 *
 * The NPSAT framework consist of two phases:
 *  - Construction Phase:
 *      The construction phase of the NPSAT involves several steps, which are time
 *      consuming but need to executed only once
 *      -# Simulation of a fully three-dimensional groundwater flow field at a spatial
 *         resolution that can properly capture individual sources (e.g., crop fields,
 *         lagoons, septic leach fields) and the impact to individual CDSs (wells, streams).
 *      -# Streamline-based transport simulation at high-spatial resolution.
 *      -# Computation of Unit Response Functions which essentially link a fraction of the
 *         source area with a fraction of the CDS
 *
 *
 * \section install_sec Installation
 *
 * \subsection step1 Step 1: Opening the box
 *
 * etc...
 */


/*! \page page1 A Tule River example
  \tableofcontents
  Leading text.
  \section sec An example section
  This page contains the subsections \ref subsection1 and \ref subsection2.
  For more info see page \ref page2.
  \subsection subsection1 The first subsection
  Text.
  \subsection subsection2 The second subsection
  More text.
*/
/*! \page page2 Modesto example
  Even more info.
*/

int main (int argc, char **argv){
    deallog.depth_console (1);
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  CL_arguments<_DIM> CLI;
    //std::cout << "Is here?" << std::endl;
    //return 0;
    if (CLI.parse_command_line(argc,argv)){
        bool read_param = CLI.read_param_file();
        if (read_param){
            if (CLI.do_gather){
                if (CLI.AQprop.part_param.particle_prefix.empty()){
                    std::cerr << "You havent specify a particle prefix name to gather data from" << std::endl;
                    return 0;
                }
                Gather_Data::gather_particles<_DIM> G;
                G.gather_streamlines(CLI.AQprop.part_param.particle_prefix, CLI.get_np(), CLI.get_nSc());
                G.print_streamlines4URF(CLI.AQprop.part_param.particle_prefix, CLI.AQprop.part_param);
                G.calculate_age(true, 365);
                G.simplify_XYZ_streamlines(CLI.AQprop.part_param.simplify_thres);
                G.print_vtk(CLI.AQprop.part_param.particle_prefix, CLI.AQprop.part_param);
            }
            else{
                //CLI.Debug_Prop();
                NPSAT<_DIM> npsat(CLI.AQprop);
                if (CLI.AQprop.solver_param.load_solution <=0)
                    npsat.solve_refine();
                if (CLI.AQprop.part_param.bDoParticleTracking > 0)
                    npsat.particle_tracking();
            }
        }
        else
            std::cerr << "Error while reading the parameter file" << std::endl;
    }
	return 0;
}
