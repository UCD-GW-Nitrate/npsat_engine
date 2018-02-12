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

using namespace dealii;


int main (int argc, char **argv){
    deallog.depth_console (1);
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    CL_arguments<_DIM> CLI;
    if (CLI.parse_command_line(argc,argv)){
        bool read_param = CLI.read_param_file();
        if (read_param){
            CLI.Debug_Prop();
            NPSAT<_DIM> npsat(CLI.AQprop);
            npsat.solve_refine();
        }
    }

    //InterpInterface<2> II;
    //II.get_data("10");
    //std::cout << II.interpolate(Point<2>(0,0)) << std::endl;

	return 0;
}
