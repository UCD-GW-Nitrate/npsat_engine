#ifndef USER_INPUT_H
#define USER_INPUT_H

#include <deal.II/base/mpi.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/utilities.h>


#include <iostream>
#include <fstream>
#include <list>

#include "dsimstructs.h"
#include "helper_functions.h"

using namespace dealii;

//===================================================
// DECLARATIONS
//===================================================


/*!
 * \brief The CL_arguments class provides the functionality to read the Command line user inputs
 *
 * Typically the program starts by typing ./npsat3d -p inputfile.
 * This class is responsible to read the user input.
 */
template<int dim>
class CL_arguments{
public:
    //! The is the constructor. Besides basic initialization code sets CL_arguments#dim equal 3
        CL_arguments();

    /*!
    * \brief parse_command_line reads the command line user input
    *
    * Current availale options are
    * - -p <input_file>
    * - -h displays help options
    */
    bool parse_command_line(const int argc, char *const *argv);

    //! Prints a message
    void print_usage_message();

    //! Reads the parameter file.
    bool read_param_file();

    //! Returns the dimension of the problem
    int get_dim();

    //! returns the user input file name
    std::string get_param_file_name();

    //! Returns the number of processor that the NPSAT was solved during construction phase
    int get_np();

    //! Returns the number of chunks that the streamlines have been divided during the contruction phase
    int get_nSc();

    //! A struct that holds all the aquifer properties needed for the simulation
    AquiferProperties<dim>   AQprop;

    void Debug_Prop();

private:
    //! This is the typical MPI communicator
    MPI_Comm                                  mpi_communicator;

    //! This is used to print the output only on one processor
    ConditionalOStream                        pcout;

    // //! Dimension of the problem. Currently only 3D is used
    //int dim;

    //! A variable to store the name of the input file
    std::string param_file;

    //! This is the number of processors that the NPSAT problem was solved.

    /*!
     * \brief nproc_solve is the number of processors that the NPSAT problem was solved.
     *
     * This parameter is used only when we gather the streamlines
     */
    int nproc_solve;

    /*!
     * \brief nStreamlineChunks specifies in how many chunks the streamlines have been divided in particle
     * tracking during the construction phase of the NPSAT
     */
    int nStreamlineChunks;

    //! This is the typical deal parameter handler
    ParameterHandler                    prm;

    //! This method declares the parameters. Any new parameter should be added here.
    void declare_parameters();

    //!This is the directory where all the input files are.
    std::string input_dir;

    //!This is the directory where all the output files will be saved
    std::string output_dir;
};


//===================================================
// IMPLEMENTATION
//===================================================
template<int dim>
CL_arguments<dim>::CL_arguments()
    :
    mpi_communicator (MPI_COMM_WORLD),
    pcout(std::cout,(Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
{
    declare_parameters();
}
template<int dim>
bool CL_arguments<dim>::parse_command_line(const int argc, char *const *argv){
    bool out = false;
    // The number of input arguments has to be at least two since they are pairs
    if (argc < 2){
        print_usage_message();
    }else{
        std::list<std::string> args;
        for (int i = 1; i < argc; ++i)
            args.push_back(argv[i]);

        while (args.size()){
            if (args.front() == "-p"){
                args.pop_front();
                if (args.size() == 0){
                    // produce some error
                    std::cerr << "Error: flag '-p' must be followed by the "
                              << "name of a parameter file."
                              << std::endl;
                }else{
                    param_file = args.front();
                    args.pop_front();
                    out = true;
                }
            }
            else if (args.front() == "-h"){
                args.pop_front();
                print_usage_message();
            }
            else {
                args.pop_front();
            }
        }
    }
    return out;
}

template<int dim>
void CL_arguments<dim>::print_usage_message(){
    static const char *message
        =
          "\n"
          "NPSAT simulation.\n"
          "\n"
          "Usage:\n"
          "    ./npsat2d [-p parameter_file] Simulation input file \n"
          "              [-h] creates a template input file\n"
          "              [-m or -g] Runs the main simulation or gather particles\n"
          "\n"
          "The input file has the following format and allows the following\n"
          "values (you can cut and paste this and use it for your own parameter\n"
          "file):\n"
          "\n";
      pcout << message;

      prm.print_parameters(pcout.get_stream(),ParameterHandler::Text);
}


template<int dim>
void CL_arguments<dim>::declare_parameters(){

    prm.enter_subsection("Workspace directories");
    {
       prm.declare_entry ("Input directory", "",Patterns::Anything(),
                          "The directory with all input data. \n"
                          "All input data must be under the same directory.");

       prm.declare_entry ("Output directory", "",Patterns::Anything(),
                          "The directory where all output data will be saved.");
    }
    prm.leave_subsection();


     prm.enter_subsection("Geometry");
     {
         prm.declare_entry("Geometry Type","BOX", Patterns::Anything(),
                           "Geometry Type. Valid entries are ""BOX"" and ""FILE""");

         prm.declare_entry ("Input mesh file", "",Patterns::Anything(),
                            "If the geometry type is FILE specify the file name\n"
                            "that contains the initial mesh.\n If the type is box this is ignored.");

         prm.declare_entry("XYZ dimensions", "5000,5000,300", Patterns::List(Patterns::Double(1,1000000),3,3,","),
                            "The length of the aquifer along the X Y Z directions.\n"
                            "In case of 2D the 3rd element is ignored but it should be > 0\n."
                            "This is valid for BOX geometry type");

         prm.declare_entry("Left lower point","0,0,0", Patterns::List(Patterns::Double(-1000000,1000000),3,3,","),
                                 "The coordinates of the left lower point of the domain.\n"
                                 "In case of 2D the 3rd element is ignored\n."
                                 "This is valid for BOX geometry type");

         prm.declare_entry ("Top elevation function", "",Patterns::Anything(),
                            "Top elevation function must be either a single value\n"
                            "or the name of a file");

         prm.declare_entry ("Bottom elevation function", "",Patterns::Anything(),
                            "Bottom elevation function must be either a single value\n"
                            "or the name of a file");

         prm.declare_entry("Initial mesh file name","", Patterns::Anything(),
                           "The name of the file to print the initial mesh. \n"
                           "Leave empty if no print is required");

         prm.declare_entry("x-y threshold", "0.1", Patterns::Double(0,10000),
                           "Threshold value along the horizontal plane");

         prm.declare_entry("z threshold", "0.01", Patterns::Double(0,10000),
                           "Threshold value the vertical plane");
     }
     prm.leave_subsection();

     prm.enter_subsection("Discretization");
     {
         prm.declare_entry("Nxyz","10,10,3", Patterns::List(Patterns::Integer(1,1000),3,3,","),
                           "The number of cells along the x, y, z directions, in that order.\n"
                           "In case of 2D the 3rd element is ignored but it should be set > 0.");

         prm.declare_entry("Initial Refinement", "0", Patterns::Integer(0,10),
                         "The number of initial refinements");

         prm.declare_entry("Well Refinement", "0", Patterns::Integer(0,10),
                           "The number of initial refinements around the wells");

         prm.declare_entry("Stream Refinement", "0", Patterns::Integer(0,10),
                           "The number of initial refinements around the Streams");

         prm.declare_entry("Vertical discretization", "", Patterns::Anything(),
                           "A list of numbers between 0 and 1 that correspond\n"
                           "to the vertical distribution of layers separated by "","".\n"
                           "This parameter overrides the 3rd or 2nd element of the Nxyz\n"
                           "in 3D or 2D respecitvely");
     }
     prm.leave_subsection();

     prm.enter_subsection("Debug Parameters");
     {
         prm.declare_entry("Domain Scale X", "5000", Patterns::Double(0,10000),
                           "This should be roughly equal to the maximum dimension\n"
                           "along the X Y");
         prm.declare_entry("Domain Scale Z", "100", Patterns::Double(0,10000),
                           "This should be roughly equal to the maximum dimension\n"
                           "along the Z");
     }
     prm.leave_subsection();
}

template<int dim>
bool CL_arguments<dim>::read_param_file(){

    std::ifstream f(param_file.c_str());
    if (!f.good())
        return false;

    prm.parse_input(param_file);

    //+++++++++++++++++++++++++++++++++++++++++
    // WORKSPACE DIRECTORIES
    //+++++++++++++++++++++++++++++++++++++++++
    prm.enter_subsection("Workspace directories");
    {
        input_dir = prm.get("Input directory");
        output_dir = prm.get("Output directory");
        AQprop.Dirs.input = input_dir;
        AQprop.Dirs.output = output_dir;
    }
    prm.leave_subsection ();


    //+++++++++++++++++++++++++++++++++++++++++
    // GEOMETRY
    //+++++++++++++++++++++++++++++++++++++++++
    prm.enter_subsection("Geometry");
    {
        AQprop.geomtype = prm.get("Geometry Type");

        if (AQprop.geomtype == "BOX"){
            std::vector<std::string> temp = Utilities::split_string_list(prm.get("XYZ dimensions"));
            AQprop.Length = Utilities::string_to_double(temp);

            temp.clear();
            temp = Utilities::split_string_list(prm.get("Left lower point"));
            AQprop.left_lower_point = Utilities::string_to_double(temp);
        } else if (AQprop.geomtype == "FILE") {
            AQprop.input_mesh_file = prm.get("Input mesh file");
            AQprop.input_mesh_file = input_dir + AQprop.input_mesh_file;
        } else {
            std::cerr << AQprop.geomtype << " is Not Valid input" << std::endl;
        }

        std::string temp_name = prm.get("Top elevation function");
        if (is_input_a_scalar((temp_name))){
            AQprop.top_elevation.get_data(temp_name);
        }else{
            AQprop.top_elevation.get_data(input_dir + temp_name);
        }

        temp_name = prm.get("Bottom elevation function");
        if (is_input_a_scalar(temp_name)){
            AQprop.bottom_elevation.get_data(temp_name);
        }
        else{
            AQprop.bottom_elevation.get_data(input_dir + temp_name);
        }

        AQprop.xy_thres = prm.get_double("x-y threshold");
        AQprop.z_thres = prm.get_double("z threshold");
    }
    prm.leave_subsection ();

    prm.enter_subsection("Discretization");
    {
        std::vector<std::string> temp = Utilities::split_string_list(prm.get("Nxyz"));
        AQprop.Nxyz = Utilities::string_to_int(temp);

        AQprop.N_init_refinement = prm.get_integer("Initial Refinement");

        AQprop.N_well_refinement = prm.get_integer("Well Refinement");

        std::string temp_str = prm.get("Vertical discretization");

        if (temp_str != ""){
            std::vector<std::string> temp1 = Utilities::split_string_list(temp_str);
            AQprop.vert_discr = Utilities::string_to_double(temp1);
        }
        else{
            AQprop.vert_discr = linspace(0,1, AQprop.Nxyz[dim-1]);
        }
    }
    prm.leave_subsection ();

    prm.enter_subsection("Debug Parameters");
    {
        AQprop.dbg_scale_x = prm.get_double("Domain Scale X");
        AQprop.dbg_scale_z = prm.get_double("Domain Scale Z");
    }
    prm.leave_subsection ();

    return true;
}

template <int dim>
void CL_arguments<dim>::Debug_Prop(){
    pcout << AQprop.geomtype << std::endl;

}

#endif // USER_INPUT_H
