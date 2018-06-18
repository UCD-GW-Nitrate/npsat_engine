#ifndef USER_INPUT_H
#define USER_INPUT_H

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

#include <deal.II/base/mpi.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/utilities.h>


#include <iostream>
#include <fstream>
#include <list>

#include "dsimstructs.h"
#include "helper_functions.h"
#include "cgal_functions.h"

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
    CL_arguments(/*ine_Tree   stream_tree_in*/);

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

    bool do_gather;

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

        do_gather = false;
        while (args.size()){
            if (args.front() == "-p"){
                args.pop_front();
                if (args.size() == 0){
                    // produce some error
                    std::cerr << "Error: flag '-p' must be followed by the "
                              << "name of a parameter file."
                              << std::endl;
                }
                else{
                    param_file = args.front();
                    args.pop_front();
                    out = true;
                }
            }
            else if (args.front() == "-g"){
                do_gather = true;
                args.pop_front();
                if (args.size() == 0){
                    do_gather = false;
                    std::cerr << "Error: flag '-g' must be followed by the "
                              << "# of processors used during the simulation."
                              << "and the number of chunks."
                              << std::endl;
                }
                else{
                    bool tf = is_input_a_scalar(args.front());
                    if (tf){
                        nproc_solve = dealii::Utilities::string_to_int(args.front());
                    }
                    else{
                        do_gather = false;
                        std::cerr << " -g option expects two integers" << std::endl;
                    }
                    args.pop_front();
                    tf = is_input_a_scalar(args.front());
                    if (tf){
                        nStreamlineChunks = dealii::Utilities::string_to_int(args.front());
                    }
                    else{
                        do_gather = false;
                        std::cerr << " -g option expects two integers" << std::endl;
                    }
                    args.pop_front();
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
          "    ./npsat [-p parameter_file] Simulation input file \n"
          "            [-h] creates a template input file\n"
          "            [-g Nproc Nchunks] Gather particles from a run with Nproc processors\n"
          "                                where the particle tracking has been split into Nchunks\n"
          "                       (The parameter file is also required. You have to provide\n"
          "                        -p and -g options to gather particles)\n"
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

    //+++++++++++++++++++++++++++++++++++++++++
    // WORKSPACE DIRECTORIES
    //+++++++++++++++++++++++++++++++++++++++++
    prm.enter_subsection("A. Workspace directories ============================");
    {
        prm.declare_entry ("a Input directory", "",Patterns::Anything(),
                           "a----------------------------------\n"
                           "The directory with all input data. \n"
                           "All input data must be under the same directory.");

        prm.declare_entry ("b Output directory", "",Patterns::Anything(),
                           "b----------------------------------\n"
                           "The directory where all output data will be saved.");
    }
    prm.leave_subsection();


    //+++++++++++++++++++++++++++++++++++++++++
    // GEOMETRY
    //+++++++++++++++++++++++++++++++++++++++++
    prm.enter_subsection("B. Geometry  ========================================");
    {
        prm.declare_entry("a Geometry Type","BOX", Patterns::Anything(),
                          "a----------------------------------\n"
                          "Geometry Type. Valid entries are ""BOX"" and ""FILE""");

        prm.enter_subsection("A. Case BOX =---=---=---=---=---=---=---=---=---=");
        {
            prm.declare_entry("a XYZ dimensions", "5000,5000,300", Patterns::List(Patterns::Double(1,1000000),3,3,","),
                              "a----------------------------------\n"
                              "The length of the aquifer along the X Y Z directions.\n"
                              "In case of 2D the 3rd element is ignored but it should be present.\n"
                              "This is valid for BOX geometry type");

            prm.declare_entry("b Left lower point","0,0,0", Patterns::List(Patterns::Double(-1000000,1000000),3,3,","),
                              "b----------------------------------\n"
                              "case BOX:\n"
                              "The coordinates of the left lower point of the domain.\n"
                              "In case of 2D the 3rd element is ignored but it should be present.\n"
                              "This is valid for BOX geometry type");
        }
        prm.leave_subsection();

        prm.enter_subsection("B. Case FILE =---=---=---=---=---=---=---=---=---");
        {
            prm.declare_entry ("a Input mesh file", "",Patterns::Anything(),
                               "a----------------------------------\n"
                               "If the geometry type is FILE specify the file name\n"
                               "that contains the initial mesh.\n If the type is box this is ignored.");
        }
        prm.leave_subsection();

        prm.enter_subsection("C. Common parameters =---=---=---=---=---=---=---");
        {
            prm.declare_entry ("a Top elevation function", "",Patterns::Anything(),
                               "a----------------------------------\n"
                               "Top elevation function must be either a single value\n"
                               "or the name of a file");

            prm.declare_entry ("b Bottom elevation function", "",Patterns::Anything(),
                               "b----------------------------------\n"
                               "Bottom elevation function must be either a single value\n"
                               "or the name of a file");

            prm.declare_entry("c x-y threshold", "0.1", Patterns::Double(0,10000),
                              "c----------------------------------\n"
                              "Threshold value along the horizontal plane");

            prm.declare_entry("d z threshold", "0.01", Patterns::Double(0,10000),
                              "d----------------------------------\n"
                              "Threshold value the vertical plane");
        }
        prm.leave_subsection();
    }
    prm.leave_subsection();

    //+++++++++++++++++++++++++++++++++++++++++
    // DISCRETIZATION PARAMETERS
    //+++++++++++++++++++++++++++++++++++++++++
    prm.enter_subsection("C. Discretization ===================================");
    {
        prm.declare_entry("a Nxyz","10,10,3", Patterns::List(Patterns::Integer(1,1000),3,3,","),
                          "a----------------------------------\n"
                          "The number of cells along the x, y, z directions, in that order.\n"
                          "In case of 2D the 3rd element is ignored but it should be present.");

        prm.declare_entry("b Vertical discretization", "", Patterns::Anything(),
                          "b----------------------------------\n"
                          "A list of numbers between 0 and 1 that correspond\n"
                          "to the vertical distribution of layers separated by (,).\n"
                          "This parameter, if present, overrides the 3rd or 2nd element of the Nxyz\n"
                          "in 3D or 2D respectively");

        prm.declare_entry("c Initial Refinement", "0", Patterns::Integer(0,10),
                          "c----------------------------------\n"
                          "The number of initial refinements");

        prm.declare_entry("d Well Refinement", "0", Patterns::Integer(0,10),
                          "d----------------------------------\n"
                          "The number of initial refinements around the wells");

        prm.declare_entry("e Stream Refinement", "0", Patterns::Integer(0,10),
                          "e----------------------------------\n"
                          "The number of initial refinements around the Streams");

    }
    prm.leave_subsection();


    //+++++++++++++++++++++++++++++++++++++++++
    // BOUNDARY CONDITIONS
    //+++++++++++++++++++++++++++++++++++++++++
    prm.enter_subsection("D. Boundary Conditions ==============================");
    {
        prm.declare_entry ("a Dirichlet file name", "",Patterns::Anything(),
                           "a----------------------------------\n"
                           "The name file of multiple files separated by ; \n"
                           "with the constant head boundary conditions");

    }
    prm.leave_subsection();

    //+++++++++++++++++++++++++++++++++++++++++
    // AQUIFER PROPERTIES
    //+++++++++++++++++++++++++++++++++++++++++
    prm.enter_subsection("E Aquifer Properties ================================");
    {
        prm.declare_entry ("a Hydraulic Conductivity KX", "",Patterns::Anything(),
                           "a----------------------------------\n"
                           "Hydraulic conductivity along the x direction.\n"
                           "It must be either a single value or the name of a file");

        prm.declare_entry ("b Hydraulic Conductivity KY", "",Patterns::Anything(),
                           "b----------------------------------\n"
                           "Hydraulic conductivity along the y direction.\n"
                           "It must be either a single value or the name of a file\n"
                           "If KY == KX leave this empty. In 2D this is ignored");

        prm.declare_entry ("c Hydraulic Conductivity KZ", "",Patterns::Anything(),
                           "c----------------------------------\n"
                           "Hydraulic conductivity along the z direction.\n"
                           "It must be either a single value or the name of a file\n"
                           "If KZ == KX leave this empty. In 2D this corresponds to the vertical K");

        prm.declare_entry("d Porosity", "", Patterns::Anything(),
                          "d----------------------------------\n"
                          "Porosity. This is used during particle tracking\n"
                          "A single value or the name of the file");
    }
    prm.leave_subsection();

    //+++++++++++++++++++++++++++++++++++++++++
    // SOURCES & SINKS PARAMETERS
    //+++++++++++++++++++++++++++++++++++++++++
    prm.enter_subsection("F. Sources & Sinks ==================================");
    {
        prm.declare_entry ("a Groundwater recharge", "",Patterns::Anything(),
                           "a----------------------------------\n"
                           "Groundwater recharche function\n"
                           "must be either a single value or the name of a file");

        prm.declare_entry ("b Stream recharge", "",Patterns::Anything(),
                           "b----------------------------------\n"
                           "Stream network\n"
                           "The name of a file that defines the streams and the rates");

        prm.declare_entry ("c Wells", "",Patterns::Anything(),
                           "c----------------------------------\n"
                           "Well network\n"
                           "The name of a file with the well information");
    }
    prm.leave_subsection();


    //+++++++++++++++++++++++++++++++++++++++++
    // SOLVER PARAMETERS
    //+++++++++++++++++++++++++++++++++++++++++
    prm.enter_subsection("G. Solver parameters ================================");{
        prm.declare_entry("a Nonlinear iterations", "10", Patterns::Integer(0,30),
                          "a----------------------------------\n"
                          "Number of nonlinear iterations for solving the unconfined problem\n"
                          "The refinement will occur during the N Max refinements iterations.");

        prm.declare_entry("b Solver tolerance", "1e-8", Patterns::Double(0,0.001),
                          "b----------------------------------\n"
                          "Tolerance of solver");

        prm.declare_entry("c Max iterations", "1000", Patterns::Integer(100,20000),
                          "c----------------------------------\n"
                          "Number of maximum solver iterations");
    }
    prm.leave_subsection();


    //+++++++++++++++++++++++++++++++++++++++++
    // REFINEMENT PARAMETERS
    //+++++++++++++++++++++++++++++++++++++++++
    prm.enter_subsection("H. Refinement Parameters ============================");
    {
        prm.declare_entry("a Max refinements iterations", "5", Patterns::Integer(0,30),
                          "a----------------------------------\n"
                          "Number of maximum refinement cycles during a simulation");

        prm.declare_entry("b Top fraction", "0.15", Patterns::Double(0,1),
                          "b----------------------------------\n"
                          "Top refinement fraction\n"
                          "This defines the percentage of cells to be refined");

        prm.declare_entry("c Bottom fraction", "0.01", Patterns::Double(0,1),
                          "c----------------------------------\n"
                          "Bottom refinement fraction\n"
                          "This defines the percentage of cells to be coarsen");

        prm.declare_entry("d Minimum element size", "1.0", Patterns::Double(0.1,1000),
                          "d----------------------------------\n"
                          "Minimum element size. Cells with lower diameter\n"
                          "will not be further refined. This should be greater than the\n"
                          "thresholds in the geometry subsection");
    }
    prm.leave_subsection();

    //+++++++++++++++++++++++++++++++++++++++++
    // PARTICLE TRACKING PARAMETERS
    //+++++++++++++++++++++++++++++++++++++++++
    prm.enter_subsection("I. Particle tracking ================================");
    {
        prm.declare_entry("a Print entity frequency", "1", Patterns::Integer(0,1000),
                          "a----------------------------------\n"
                          "This will print the streamlines every N Entities (e.g. every 10 wells)");

        prm.declare_entry("b Print streamline frequency", "1", Patterns::Integer(0,1000),
                          "b----------------------------------\n"
                          "For each Entity will print every N streamlines");

        prm.declare_entry("c Stuck iterations", "50", Patterns::Integer(0,1000),
                          "c----------------------------------\n"
                          "If the streamline has not been expanded after N iterations, stop tracing this particle");

        prm.declare_entry("d Outer iterations", "100", Patterns::Integer(0,1000),
                          "d----------------------------------\n"
                          "The number of times that processors are allowed to exchange particles\n"
                          "This will prevent case where a particle moves back and forth between processors");

        prm.declare_entry("e Streamline iterations", "1000", Patterns::Integer(0,10000),
                          "e----------------------------------\n"
                          "The maximum number of steps per streamline");

        prm.declare_entry("f Tracking method", "3", Patterns::Integer(1,3),
                          "f----------------------------------\n"
                          "1-> Euler, 2->RK2, 3->RK4");

        prm.declare_entry("g Step size", "6", Patterns::Double(1,100),
                          "g----------------------------------\n"
                          "The actual step size for each cell is calculated by dividing\n "
                          "the diameter of each cell with the given number.\n"
                          "Essensially this number indicates the average number of steps\n"
                          "of the algorithm within a cell");

        prm.declare_entry("h Search iterations", "3", Patterns::Integer(1,10),
                          "h----------------------------------\n"
                          "How many neighbor to search for the next point if the point has left the current cell");

        prm.declare_entry("i Simplify threshold", "5.5", Patterns::Double(0,100),
                          "i----------------------------------\n"
                          "Simplification threshold used for plotting");

        prm.declare_entry("j Do particle tracking","1", Patterns::Integer(0,1),
                          "j----------------------------------\n"
                          "Set to 0 to deactivate particle tracking. Default is 1");

        prm.declare_entry("k N Particles in parallel", "5000", Patterns::Integer(1,10000),
                          "k----------------------------------\n"
                          "The maximum number of particles that is allowed to run in parallel");

        prm.declare_entry("l Layers per well", "25", Patterns::Integer(1,500),
                          "l----------------------------------\n"
                          "The number of layers that the particles will be distributed around the well");

        prm.declare_entry("m Particles per layer(well)", "4", Patterns::Integer(1,100),
                          "m----------------------------------\n"
                          "The number of particles per layers for the wells");

        prm.declare_entry("n Distance from well", "50.0", Patterns::Double(1,1000),
                          "n----------------------------------\n"
                          "The distance from well that the particles will be releazed");
    }
    prm.leave_subsection ();

    //+++++++++++++++++++++++++++++++++++++++++
    // OUTPUT PARAMETERS
    //+++++++++++++++++++++++++++++++++++++++++
    prm.enter_subsection("J. Output Parameters ================================");
    {
        prm.declare_entry("a Prefix", "", Patterns::Anything(),
                          "a----------------------------------\n"
                          "Prefix is a keyword that is used when printing the various\n"
                          "output files");

        prm.declare_entry("b Domain Scale X", "1000", Patterns::Double(0,10000),
                          "b----------------------------------\n"
                          "This should be roughly equal to the maximum dimension\n"
                          "along the X Y");

        prm.declare_entry("c Domain Scale Z", "500", Patterns::Double(0,10000),
                          "c----------------------------------\n"
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
    prm.enter_subsection("A. Workspace directories ============================");
    {
        input_dir = prm.get("a Input directory");
        output_dir = prm.get("b Output directory");

        AQprop.Dirs.input = input_dir;
        AQprop.Dirs.output = output_dir;
    }
    prm.leave_subsection ();


    //+++++++++++++++++++++++++++++++++++++++++
    // GEOMETRY
    //+++++++++++++++++++++++++++++++++++++++++
    prm.enter_subsection("B. Geometry  ========================================");
    {
        AQprop.geomtype = prm.get("a Geometry Type");

        if (AQprop.geomtype == "BOX"){
            prm.enter_subsection("A. Case BOX =---=---=---=---=---=---=---=---=---=");
            {
                std::vector<std::string> temp = Utilities::split_string_list(prm.get("a XYZ dimensions"));
                AQprop.Length = Utilities::string_to_double(temp);

                temp.clear();
                temp = Utilities::split_string_list(prm.get("b Left lower point"));
                AQprop.left_lower_point = Utilities::string_to_double(temp);
            }
            prm.leave_subsection();
        } else if (AQprop.geomtype == "FILE") {
            prm.enter_subsection("B. Case FILE =---=---=---=---=---=---=---=---=---");
            {
                AQprop.input_mesh_file = prm.get("a Input mesh file");
                AQprop.input_mesh_file = input_dir + AQprop.input_mesh_file;
            }
            prm.leave_subsection();
        } else {
            std::cerr << AQprop.geomtype << " is Not Valid input" << std::endl;
        }

        prm.enter_subsection("C. Common parameters =---=---=---=---=---=---=---");
        {
            std::string temp_name = prm.get("a Top elevation function");
            if (is_input_a_scalar((temp_name))){
                AQprop.top_elevation.get_data(temp_name);
            }else{
                AQprop.top_elevation.get_data(input_dir + temp_name);
            }

            temp_name = prm.get("b Bottom elevation function");
            if (is_input_a_scalar(temp_name)){
                AQprop.bottom_elevation.get_data(temp_name);
            }
            else{
                AQprop.bottom_elevation.get_data(input_dir + temp_name);
            }

            AQprop.xy_thres = prm.get_double("c x-y threshold");
            AQprop.z_thres = prm.get_double("d z threshold");
        }
        prm.leave_subsection ();
    }
    prm.leave_subsection ();


    //+++++++++++++++++++++++++++++++++++++++++
    // DISCRETIZATION
    //+++++++++++++++++++++++++++++++++++++++++
    prm.enter_subsection("C. Discretization ===================================");
    {
        std::vector<std::string> temp = Utilities::split_string_list(prm.get("a Nxyz"));
        AQprop.Nxyz = Utilities::string_to_int(temp);

        std::string temp_str = prm.get("b Vertical discretization");

        AQprop.N_init_refinement = prm.get_integer("c Initial Refinement");

        AQprop.N_well_refinement = prm.get_integer("d Well Refinement");

        AQprop.N_streams_refinement = prm.get_integer("e Stream Refinement");


        if (temp_str != ""){
            std::vector<std::string> temp1 = Utilities::split_string_list(temp_str);
            AQprop.vert_discr = Utilities::string_to_double(temp1);
        }
        else{
            AQprop.vert_discr = linspace(0,1, AQprop.Nxyz[dim-1]);
        }
    }
    prm.leave_subsection ();


    //+++++++++++++++++++++++++++++++++++++++++
    // BOUNDARY CONDITIONS
    //+++++++++++++++++++++++++++++++++++++++++
    prm.enter_subsection("D. Boundary Conditions ==============================");
    {
        AQprop.dirichlet_file_names = input_dir + prm.get("a Dirichlet file name");
    }
    prm.leave_subsection ();


    //+++++++++++++++++++++++++++++++++++++++++
    // AQUIFER PARAMETERS
    //+++++++++++++++++++++++++++++++++++++++++
    prm.enter_subsection("E Aquifer Properties ================================");
    {
        // KX hydraulic conductivity
        AQprop.HydraulicConductivity.resize(dim);
        AQprop.HKuse.resize(dim,false);
        std::string temp_name = prm.get("a Hydraulic Conductivity KX");
        if (is_input_a_scalar(temp_name)){
            AQprop.HydraulicConductivity[0].get_data(temp_name);
        }
        else{
            AQprop.HydraulicConductivity[0].get_data(input_dir + temp_name);
        }
        AQprop.HKuse[0] = true;

        //KZ Hydraulic Conductivity
        std::string KZ_file = prm.get("c Hydraulic Conductivity KZ");
        if (KZ_file != ""){
            if (is_input_a_scalar(KZ_file)){
                AQprop.HydraulicConductivity[dim-1].get_data(KZ_file);
            }
            else{
                AQprop.HydraulicConductivity[dim-1].get_data(input_dir + KZ_file);
            }
            AQprop.HKuse[dim-1] = true;
        }

        if (dim == 3){
            //KY Hydraulic Conductivity
            std::string KY_file = prm.get("b Hydraulic Conductivity KY");
            if (KY_file != ""){
                if(is_input_a_scalar(KY_file)){
                    AQprop.HydraulicConductivity[1].get_data(KY_file);
                }
                else{
                    AQprop.HydraulicConductivity[1].get_data(input_dir + KY_file);
                }
                AQprop.HKuse[1] = true;
            }
        }

        { // Set up the hydraulic conductivity tensor function
            if (dim == 2){
                if (AQprop.HKuse[1] == false)
                    AQprop.HK_function = new MyTensorFunction<dim>(AQprop.HydraulicConductivity[0]);
                else
                   AQprop.HK_function = new MyTensorFunction<dim>(AQprop.HydraulicConductivity[0],
                                                                   AQprop.HydraulicConductivity[1]);
            }
            else if (dim == 3){
                if (AQprop.HKuse[1] == false && AQprop.HKuse[2] == false)
                    AQprop.HK_function = new MyTensorFunction<dim>(AQprop.HydraulicConductivity[0]);
                else if(AQprop.HKuse[1] == false && AQprop.HKuse[2] == true)
                    AQprop.HK_function = new MyTensorFunction<dim>(AQprop.HydraulicConductivity[0],
                                                                   AQprop.HydraulicConductivity[2]);
                else if(AQprop.HKuse[1] == true && AQprop.HKuse[2] == true)
                    AQprop.HK_function = new MyTensorFunction<dim>(AQprop.HydraulicConductivity[0],
                                                                   AQprop.HydraulicConductivity[1],
                                                                   AQprop.HydraulicConductivity[2]);
            }
        }

        //Porosity
        std::string por_file = prm.get("d Porosity");
        if (is_input_a_scalar(por_file)){
            AQprop.Porosity.get_data(por_file);
        }
        else{
            AQprop.Porosity.get_data(input_dir + por_file);
        }
    }
    prm.leave_subsection ();


    //+++++++++++++++++++++++++++++++++++++++++
    // SOURCES & SINKS
    //+++++++++++++++++++++++++++++++++++++++++
    prm.enter_subsection("F. Sources & Sinks ==================================");
    {
        // Diffuse Recharge
        std::string temp_name = prm.get("a Groundwater recharge");
        if (is_input_a_scalar(temp_name)){
            AQprop.GroundwaterRecharge.get_data(temp_name);
        }
        else{
            AQprop.GroundwaterRecharge.get_data(input_dir + temp_name);
        }

        // Read Wells
        std::string well_file = prm.get("c Wells");
        if (well_file != ""){
            well_file = input_dir + well_file;
            AQprop.have_wells = AQprop.wells.read_wells(well_file);
        }

        //Read Rivers
        std::string stream_file = prm.get("b Stream recharge");
        if (stream_file != ""){
            stream_file = input_dir + stream_file;
            AQprop.have_streams = AQprop.streams.read_streams(stream_file);
        }

    }
    prm.leave_subsection();


    //+++++++++++++++++++++++++++++++++++++++++
    // SOLVER PARAMETERS
    //+++++++++++++++++++++++++++++++++++++++++
    prm.enter_subsection("G. Solver parameters ================================");
    {
        AQprop.solver_param.NonLinearIter = prm.get_integer("a Nonlinear iterations");
        AQprop.solver_param.solver_tol = prm.get_double("b Solver tolerance");
        AQprop.solver_param.Maxiter = prm.get_integer("c Max iterations");
    }
    prm.leave_subsection ();


    prm.enter_subsection("H. Refinement Parameters ============================");
    {
        AQprop.refine_param.MaxRefinement = prm.get_integer("a Max refinements iterations");
        AQprop.refine_param.TopFraction = prm.get_double("b Top fraction");
        AQprop.refine_param.BottomFraction = prm.get_double("c Bottom fraction");
        AQprop.refine_param.MinElementSize = prm.get_double("d Minimum element size");
    }
    prm.leave_subsection ();


    //+++++++++++++++++++++++++++++++++++++++++
    // PARTICLE TRACKING
    //+++++++++++++++++++++++++++++++++++++++++
    prm.enter_subsection("I. Particle tracking ================================");
    {
        AQprop.part_param.Entity_freq = prm.get_integer("a Print entity frequency");
        AQprop.part_param.Streaml_freq = prm.get_integer("b Print streamline frequency");
        AQprop.part_param.Stuck_iter = prm.get_integer("c Stuck iterations");
        AQprop.part_param.Outmost_iter = prm.get_integer("d Outer iterations");
        AQprop.part_param.streaml_iter = prm.get_integer("e Streamline iterations");
        AQprop.part_param.method = prm.get_integer("f Tracking method");
        AQprop.part_param.step_size = prm.get_double("g Step size");
        AQprop.part_param.search_iter = prm.get_integer("h Search iterations");
        AQprop.part_param.simplify_thres = prm.get_double("i Simplify threshold");
        AQprop.part_param.bDoParticleTracking = prm.get_integer("j Do particle tracking");
        AQprop.part_param.Nparallel_particles = prm.get_integer("k N Particles in parallel");
        AQprop.part_param.Wells_N_Layers = prm.get_integer("l Layers per well");
        AQprop.part_param.Wells_N_per_layer = prm.get_integer("m Particles per layer(well)");
        AQprop.part_param.radius = prm.get_double("n Distance from well");
    }
    prm.leave_subsection ();


    //+++++++++++++++++++++++++++++++++++++++++
    // OUTPUT PARAMETERS
    //+++++++++++++++++++++++++++++++++++++++++
    prm.enter_subsection("J. Output Parameters ================================");
    {
        AQprop.sim_prefix = prm.get("a Prefix");
        AQprop.dbg_scale_x = prm.get_double("b Domain Scale X");
        AQprop.dbg_scale_z = prm.get_double("c Domain Scale Z");
    }
    prm.leave_subsection ();

    return true;
}

template <int dim>
int CL_arguments<dim>::get_np(){
    return nproc_solve;
}

template <int dim>
int CL_arguments<dim>::get_nSc(){
    return nStreamlineChunks;
}

template <int dim>
void CL_arguments<dim>::Debug_Prop(){
    pcout << AQprop.geomtype << std::endl;

}

#endif // USER_INPUT_H
