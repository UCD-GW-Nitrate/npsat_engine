#ifndef DIRICHLET_BOUNDARY_H
#define DIRICHLET_BOUNDARY_H


#include <deal.II/dofs/function_map.h>
#include <deal.II/distributed/tria.h>

#include "interpinterface.h"
#include "my_functions.h"


namespace BoundaryConditions{

using namespace dealii;


/*! A primitive boundary shape.
 *
 * A primitive shape boundary is a polygon that is associated with a single constant value.
 *
 */
template<int dim>
class BoundPrim{
public:
    //! This holds the X coordinates of the polygon
    std::vector<double> Xcoords;

    //! This holds the Y coordinates of the polygon
    std::vector<double> Ycoords;

    //! This is the dirichlet value
    std::string value;

    //! The type of boundary. It can be any of the keywords TOP, BOT, EDGE
    std::string TYPE;

    //! This is the minimum point of the polygon bounding box
    Point<dim> BBmin;

    //! This is the maximum point of the polygon bounding box
    Point<dim> BBmax;

    //! A method that checks if any point in the list is inside this primitive polygon
    bool is_any_point_insideBB(std::vector<double> x, std::vector<double> y);
};

template <int dim>
bool BoundPrim<dim>::is_any_point_insideBB(std::vector<double> x, std::vector<double> y){
    bool outcome = false;
    for (unsigned int i = 0; i < x.size(); ++i){
        if (x[i] > BBmin[0] && x[i] < BBmax[0] &&
            y[i] > BBmin[1] && y[i] < BBmax[1]){
            outcome = true;
            break;
        }
    }
    return outcome;
}



/*! This is the class that provides methods to read the dirichlet boundary condition data
 * and methods to assign the dirichlet conditions to the triangulation
 */
template<int dim>
class Dirichlet{
public:
    //! The constructor initializes the number of boundraries counter #Nbnd.
    Dirichlet();


    /*!
    * \brief Reads the data from the file
    * \param This is the name of the file that the user passes in the main input file when sets the "Dirichlet file name"
    * \param input_dir is the directory where all input files should be. All input files must be under the same input directory
    *
    * Format of the file:
    *
    * The first line holds the number of boundaries that follows
    *
    * Next for each boundary we print the following
    *
    * TYPE N Value
    *
    * TYPE is any of the following keywords
    * -# TOP
    * -# BOT
    * -# EDGE
    *
    * N is the number of points that discribes the geometry
    *
    * Value is the dirichlet boundary value. This can be a scalar or a file name with interpolation function data.
    *
    * The next N lines contain the coordinates of the boundary primitive shape
    *
    * At the time or writting when the type is EDGE the next value must always be 2.
    *
    */
    void get_from_file(std::string& namefile, std::string& input_dir);

    //! This is a dealii structure that creates a map between face ids and boundaries and it is used
    //! during dirichlet boundary assignement
    typename FunctionMap<dim>::type		function_map;

    /*!
    * \brief assign_dirichlet_to_triangulation assignes the dirichlet boundary conditions.
    * This method should be called every time there is a change in the triangulation such as refinement/coarsening
    * \param triangulation is the triangulation of the domain
    * \param dirichlet_boundary Is the map between triangulation cell ids and boundary primitives
    * \param top_boundary_ids a list of ids that correspond to top boundary. This is an output of the method
    * \param bottom_boundary_ids A list of ids that corresponds to bottom boundary. This is an output of the method
    *
    * First the method clears any ids in the input boundary id lists and adds the default dealii id for the top and bottom boundaries,
    * which for the 3D case are 5 for the top and 4 for the bottom. Next creates the function map.
    *
    * Then the code loops through the triangulation cells. When the cell is locally own loops through the faces of that cell.
    * If any face is at the boundary the code resets its ids to default deal numbering and then loops through the boundary primitives.
    *
    * If the current boundary primitive is TOP and the face boundary indicator corresponds to top face
    * (similarly if the boundary primitive is bottom and the face boundary indicator corresponds to bottom face) the code  tests
    * if the 2D projection of any point of the cell face lays inside the bounding box of the boundary primitive. If this test is
    * true it tests whether there is an intersection between the boundary primitive and the face polygon. If there is such an intersection
    * the id of this face is set to the corresponing id on function map. (This above logic assumes that the cell size is at
    * least of the same order or smaller than the boundary primitive. If the boundary primitive is fully withing the cell face the code will
    * fail to identify that boundary)
    *
    * If the boundary TYPE is EDGE and the boundary indicator is (for the 3D case) 0, 1, 2 or 3 we test if this face is part of the boundary edge.
    * First we compute the distance between the boundary line and the vertices of the face. Since those faces are the vertical ones we can simply test
    * the first two vertices. If the distance is less than a specified threshold we used CGAL munctions to test if there are
    * overlapping parts between the two segments.
    */
    void assign_dirichlet_to_triangulation(parallel::distributed::Triangulation<dim>& triangulation,
                   typename FunctionMap<dim>::type&	dirichlet_boundary,
                   std::vector<int>& top_boundary_ids,
                   std::vector<int>& bottom_boundary_ids);

    /*!
     * \brief add_id Pushes the #id into the #id_list only if the #id is not already in the list
     * \param id_list
     * \param id
     */
    void add_id(std::vector<int>& id_list, int id);


    std::string namefile;
    std::vector<BoundPrim<dim> > boundary_parts;
    std::vector<InterpInterface<dim-1>> interp_funct;
    std::vector<MyFunction<dim,dim-1>> DirFunctions;
    int Nbnd;

};

template <int dim>
Dirichlet<dim>::Dirichlet(){
    Nbnd = 0;
}

template<int dim>
void Dirichlet<dim>::get_from_file(std::string& filename, std::string& input_dir){
    std::ifstream  datafile(filename.c_str());
    if (!datafile.good()){
        std::cout << "Can't open " << filename << std::endl;
    }
    else{
        char buffer[512];
        datafile.getline(buffer,512);
        std::istringstream inp(buffer);
        int N_bnd;
        inp >> N_bnd;
        boundary_parts.resize(N_bnd + boundary_parts.size());
        interp_funct.resize(N_bnd + boundary_parts.size());

        std::string type;
        for (unsigned int i = 0; i < N_bnd; ++i){
            datafile.getline(buffer,512);
            std::istringstream inp(buffer);
            inp >> boundary_parts[i].TYPE;
            //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        }

    }

}


template <int dim>
void Dirichlet<dim>::add_id(std::vector<int>& id_list, int id){
    bool id_found = false;
    for (unsigned int i = 0; i < id_list.size(); ++i){
        if (id_list[i] == id){
            id_found = true;
            break;
        }
    }

    if (!id_found)
        id_list.push_back(id);
}

}

#endif // DIRICHLET_BOUNDARY_H
