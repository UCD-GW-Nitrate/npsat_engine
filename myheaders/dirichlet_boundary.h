#ifndef DIRICHLET_BOUNDARY_H
#define DIRICHLET_BOUNDARY_H


//#include <deal.II/dofs/function_map.h>
#include <deal.II/distributed/tria.h>

#include "interpinterface.h"
#include "my_functions.h"
//#include "cgal_functions.h"
#include "helper_functions.h"
#include "boost_functions.h"


namespace BoundaryConditions{

using namespace dealii;

enum class BoundaryType{BOT,VER};

/*! A primitive boundary shape.
 *
 * A primitive shape boundary is a polygon that is associated with a single constant value.
 *
 */

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
    Point<2> BBmin;

    //! This is the maximum point of the polygon bounding box
    Point<2> BBmax;

    //! A method that checks if any point in the list is inside the bounding box of the primitive polygon
    bool is_any_point_insideBB(std::vector<double> x, std::vector<double> y);

    bool is_any_point_insideBB(double x, double y);

    bool Point_in_BB(double xmin, double ymin, double xmax, double ymax, double x, double y);

    bool is_point_inside_poly(double x, double y);

    boost_polygon poly;
};

bool BoundPrim::is_any_point_insideBB(std::vector<double> x, std::vector<double> y){
    bool outcome = false;
    double xmin = 99999999999;
    double ymin = 99999999999;
    double xmax = -99999999999;
    double ymax = -99999999999;
    for (unsigned int i = 0; i < x.size(); ++i){
        if (xmin > x[i]) xmin = x[i];
        if (ymin > y[i]) ymin = y[i];
        if (xmax < x[i]) xmax = x[i];
        if (ymax < y[i]) ymax = y[i];
        if (Point_in_BB(BBmin[0], BBmin[1], BBmax[0], BBmax[1], x[i], y[i])){
            outcome = true;
            break;
        }
    }

    if (!outcome){
        if (Point_in_BB(xmin, ymin, xmax, ymax, BBmin[0], BBmin[1]))
            outcome = true;
        else{
            if (Point_in_BB(xmin, ymin, xmax, ymax, BBmax[0], BBmax[1]))
                outcome = true;
            else{
                if (Point_in_BB(xmin, ymin, xmax, ymax, BBmin[0], BBmax[1]))
                    outcome = true;
                else{
                    if (Point_in_BB(xmin, ymin, xmax, ymax, BBmax[0], BBmin[1]))
                        outcome = true;
                }
            }
        }
    }
    return outcome;
}

bool BoundPrim::Point_in_BB(double xmin, double ymin, double xmax, double ymax, double x, double y){
    if (x > xmin && x < xmax && y > ymin && y < ymax)
        return true;
    else
        return false;
}

bool BoundPrim::is_any_point_insideBB(double x, double y){
    if (Point_in_BB(BBmin[0], BBmin[1], BBmax[0], BBmax[1], x, y)){
        return true;
    }
    return false;
}

bool BoundPrim::is_point_inside_poly(double x, double y){
    boost_point pnt(x,y);
    return boost::geometry::within(pnt, poly);
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
    * The first line holds the number of boundaries that follow
    *
    * Next for each boundary we print the following info
    *
    * TYPE N Value
    *
    * TYPE is any of the following keywords
    * -# TOP
    * -# BOT
    * -# EDGE
    * -# EDGETOP
    *
    * N is the number of points that discribes the geometry
    *
    * Value is the dirichlet boundary value. This can be a scalar or a file name with interpolation function data.
    *
    * The next N lines contain the coordinates of the boundary primitive shape
    *
    * At the moment when the type is EDGE the next value must always be 2.
    *
    */
    void get_from_file(std::string& namefile, std::string& input_dir);

    void read_master_file(std::string& namefile, std::string& input_dir);

    //! This is a dealii structure that creates a map between face ids and boundaries and it is used
    //! during dirichlet boundary assignement
    std::map<types::boundary_id, const Function<dim>* > function_map;
    //typename FunctionMap<dim>::type		function_map;

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
    * the first two vertices. If the distance is less than a specified threshold we used CGAL functions to test if there are
    * overlapping parts between the two segments.
    */
    void assign_dirichlet_to_triangulation(parallel::distributed::Triangulation<dim>& triangulation,
                    std::map<types::boundary_id, const Function<dim>* >& dirichlet_boundary,
                    /*typename FunctionMap<dim>::type&	dirichlet_boundary,*/
                    std::vector<int>& top_boundary_ids,
                    std::vector<int>& bottom_boundary_ids);

    /*!
     * \brief add_id Pushes the #id into the #id_list only if the #id is not already in the list
     * \param id_list
     * \param id
     */
    void add_id(std::vector<int>& id_list, int id);


    std::string namefile;
    std::vector<BoundPrim> boundary_parts;
    std::vector<InterpInterface<dim>> interp_funct;
    std::vector<MyFunction<dim,dim>> DirFunctions;
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
        memset (buffer,' ',512);
        datafile.getline(buffer,512);
        std::istringstream inp(buffer);
        int N_bnd;
        inp >> N_bnd;
        int new_size = N_bnd + boundary_parts.size();
        boundary_parts.resize(new_size);
        interp_funct.resize(new_size);
        DirFunctions.resize(new_size);

        std::string type;
        for (int i = 0; i < N_bnd; ++i){
            memset (buffer,' ',512);
            datafile.getline(buffer,512);
            std::istringstream inp(buffer);
            inp >> boundary_parts[i].TYPE;
            if (dim == 2){
                if (boundary_parts[i].TYPE == "EDGE" || boundary_parts[i].TYPE == "EDGETOP"){// in 2D an EDGE is represented by one point
                    double x;
                    inp >> x; boundary_parts[i].Xcoords.push_back(x);
                    boundary_parts[i].Ycoords.push_back(0.0);
                }
                else if (boundary_parts[i].TYPE == "TOP" || boundary_parts[i].TYPE == "BOT"){ // in 2D the top and bottom is represented by 2 points
                    double x;
                    inp >> x; boundary_parts[i].Xcoords.push_back(x); boundary_parts[i].Ycoords.push_back(0.0);
                    inp >> x; boundary_parts[i].Xcoords.push_back(x); boundary_parts[i].Ycoords.push_back(0.0);
                }
                inp >> boundary_parts[i].value;
                interp_funct[i].get_data(boundary_parts[i].value);
                DirFunctions[i].set_interpolant(interp_funct[i]);
                Nbnd++;
            }
            else if (dim == 3){
                int N;  // Number of points that define the polygon
                inp >> N;
                // The value for each boundary can be a scalar value or a file that it represents a 2D variable field
                std::string temp_str;
                inp >> temp_str;
                if (is_input_a_scalar(temp_str)){
                    boundary_parts[i].value = temp_str;
                }
                else{
                    boundary_parts[i].value = input_dir + temp_str;
                }

                if (boundary_parts[i].TYPE == "TOP" || boundary_parts[i].TYPE == "BOT"){
                    double x;
                    boundary_parts[i].BBmin[0] = 99999999999;
                    boundary_parts[i].BBmin[1] = 99999999999;
                    boundary_parts[i].BBmax[0] = -99999999999;
                    boundary_parts[i].BBmax[1] = -99999999999;
                    for (int iv = 0; iv < N; ++iv){
                        datafile.getline(buffer,512);
                        std::istringstream inp(buffer);
                        inp >> x;
                        // X coordinate
                        if (x < boundary_parts[i].BBmin[0])
                            boundary_parts[i].BBmin[0] = x;
                         if (x > boundary_parts[i].BBmax[0])
                             boundary_parts[i].BBmax[0] = x;
                         boundary_parts[i].Xcoords.push_back(x);

                         // Y coordinate
                         inp >> x;
                         if (x < boundary_parts[i].BBmin[1])
                             boundary_parts[i].BBmin[1] = x;
                         if (x > boundary_parts[i].BBmax[1])
                             boundary_parts[i].BBmax[1] = x;
                         boundary_parts[i].Ycoords.push_back(x);
                    }
                    interp_funct[i].get_data(boundary_parts[i].value);
                    DirFunctions[i].set_interpolant(interp_funct[i]);
                    Nbnd++;
                }
                else if (boundary_parts[i].TYPE == "EDGE" || boundary_parts[i].TYPE == "EDGETOP"){// we read the value associated with the edge
                    if (N>=2){
                        double x;
                        for (unsigned int iv = 0; iv < 2; ++iv){
                            datafile.getline(buffer,512);
                            std::istringstream inp(buffer);
                            inp >> x;
                            boundary_parts[i].Xcoords.push_back(x);
                            inp >> x;
                            boundary_parts[i].Ycoords.push_back(x);
                        }

                        interp_funct[i].get_data(boundary_parts[i].value);

                        if (interp_funct[i].get_type() == 1){
                            std::cout << "Debug why the set_SCI_EDGE_points was used in the first place" << std::endl;
                            Point<dim> a,b;
                            a[0] = boundary_parts[i].Xcoords[0];
                            a[1] = boundary_parts[i].Ycoords[0];
                            a[2] = 0;
                            b[0] = boundary_parts[i].Xcoords[1];
                            b[1] = boundary_parts[i].Ycoords[1];
                            b[2] = 0;
                        }
                        DirFunctions[i].set_interpolant(interp_funct[i]);
                        Nbnd++;
                    }
                    else{
                        interp_funct[i].get_data(boundary_parts[i].value);
                        DirFunctions[i].set_interpolant(interp_funct[i]);
                        Nbnd++;
                    }
                }
            }
        }
    }
    datafile.close();
}

template <int dim>
void Dirichlet<dim>::assign_dirichlet_to_triangulation(parallel::distributed::Triangulation<dim>& triangulation,
                                                        std::map<types::boundary_id, const Function<dim>* >& dirichlet_boundary,
                                                       /*typename FunctionMap<dim>::type&	dirichlet_boundary,*/
                                                       std::vector<int>& top_boundary_ids,
                                                       std::vector<int>& bottom_boundary_ids){
    top_boundary_ids.clear();
    bottom_boundary_ids.clear();
    top_boundary_ids.push_back(GeometryInfo<dim>::faces_per_cell-1);
    bottom_boundary_ids.push_back(GeometryInfo<dim>::faces_per_cell-2);

    const int JJ = 17;
    for (unsigned int i = 0; i < DirFunctions.size(); ++i){
        dirichlet_boundary[JJ + i] = &DirFunctions[i];
    }

    typename parallel::distributed::Triangulation<dim>::active_cell_iterator
    cell = triangulation.begin_active(),
    endc = triangulation.end();
    for (; cell!=endc; ++cell){
        if (cell->is_locally_owned() || cell->is_ghost()){
            for (unsigned int iface = 0; iface < GeometryInfo<dim>::faces_per_cell; ++iface){
                if (cell->face(iface)->at_boundary()){
                    // Here we reset the boundary indicators
                    cell->face(iface)->set_all_boundary_ids(iface);
                    cell->face(iface)->set_user_index(0);
                    for (unsigned int i = 0; i < boundary_parts.size(); ++i){
                        if (dim == 2){
                            if (boundary_parts[i].TYPE == "EDGE" && (cell->face(iface)->boundary_id() == 0 || cell->face(iface)->boundary_id() == 1) ){
                                Point<dim> x1 = cell->face(iface)->vertex(0);
                                if (abs(x1[0] - boundary_parts[i].Xcoords[0]) < 0.01){
                                    cell->face(iface)->set_all_boundary_ids(JJ+i);
                                    cell->face(iface)->set_user_index(1);
                                    break;
                                }
                            }
                            else if (boundary_parts[i].TYPE == "TOP" && cell->face(iface)->boundary_id() == 3){
                                add_id(top_boundary_ids, 3);
                                Point<dim> x1 = cell->face(iface)->vertex(0);
                                Point<dim> x2 = cell->face(iface)->vertex(1);
                                double xp1 = boundary_parts[i].Xcoords[0];
                                double xp2 = boundary_parts[i].Xcoords[1];
                                bool assign_this = false;
                                if (xp1 > x1[0] && xp1 < x2[0])
                                    assign_this = true;
                                else if (xp2 > x1[0] && xp2 < x2[0])
                                    assign_this = true;
                                else if (x1[0] > xp1 && x1[0] < xp2)
                                    assign_this = true;
                                else if (x2[0] > xp1 && x2[0] < xp2)
                                    assign_this = true;

                                if (assign_this){
                                    cell->face(iface)->set_all_boundary_ids(JJ+i);
                                    cell->face(iface)->set_user_index(1);
                                    add_id(top_boundary_ids, JJ+i);
                                    break;
                                }
                            }
                            else if (boundary_parts[i].TYPE == "BOT" && cell->face(iface)->boundary_id() == 2){
                                add_id(bottom_boundary_ids, 2);
                                Point<dim> x1 = cell->face(iface)->vertex(0);
                                Point<dim> x2 = cell->face(iface)->vertex(1);
                                double xp1 = boundary_parts[i].Xcoords[0];
                                double xp2 = boundary_parts[i].Xcoords[1];
                                bool assign_this = false;
                                if (xp1 > x1[0] && xp1 < x2[0])
                                    assign_this = true;
                                else if (xp2 > x1[0] && xp2 < x2[0])
                                    assign_this = true;
                                else if (x1[0] > xp1 && x1[0] < xp2)
                                    assign_this = true;
                                else if (x2[0] > xp1 && x2[0] < xp2)
                                    assign_this = true;

                                if (assign_this){
                                    cell->face(iface)->set_all_boundary_ids(JJ+i);
                                    cell->face(iface)->set_user_index(1);
                                    add_id(bottom_boundary_ids, JJ+i);
                                    break;
                                }
                            }
                        }
                        else if (dim == 3){
                            if ((boundary_parts[i].TYPE == "TOP" && (cell->face(iface)->boundary_id() == 5 || iface == 5)) ||
                                (boundary_parts[i].TYPE == "BOT" && (cell->face(iface)->boundary_id() == 4 || iface == 4))  ){
                                std::vector<double> xface, yface;
                                for (unsigned int ivert = 0; ivert < GeometryInfo<dim>::vertices_per_face; ++ivert){
                                    xface.push_back(cell->face(iface)->vertex(ivert)[0]);
                                    yface.push_back(cell->face(iface)->vertex(ivert)[1]);
                                }

                                if (boundary_parts[i].is_any_point_insideBB(xface, yface) == false)
                                    continue;

                                // re-orient the cell coordinates
                                double tempd = xface[2]; xface[2] = xface[3]; xface[3] = tempd;
                                tempd = yface[2]; yface[2] = yface[3]; yface[3] = tempd;

                                bool do_intersect = polyXpoly(boundary_parts[i].Xcoords, boundary_parts[i].Ycoords, xface, yface);
                                if (do_intersect){
                                    cell->face(iface)->set_all_boundary_ids(JJ+i);
                                    cell->face(iface)->set_user_index(1);
                                    if (boundary_parts[i].TYPE == "TOP")
                                        add_id(top_boundary_ids, JJ+i);
                                    else if (boundary_parts[i].TYPE == "BOT")
                                        add_id(bottom_boundary_ids, JJ+i);
                                    break;
                                }
                            }
                            else if ((boundary_parts[i].TYPE == "EDGE" || boundary_parts[i].TYPE == "EDGETOP") && (
                                         cell->face(iface)->boundary_id() == 0 ||
                                         cell->face(iface)->boundary_id() == 1 ||
                                         cell->face(iface)->boundary_id() == 2 ||
                                         cell->face(iface)->boundary_id() == 3) )
                            {
                                if (boundary_parts[i].TYPE == "EDGETOP" && !cell->face(5)->at_boundary())
                                    continue;

                                double cx3,cy3,cx4,cy4; // variables for storing the cell face coordinates
                                //double cz3, cz4;// z variables are used only for debugging
                                cx3 = cell->face(iface)->vertex(0)[0]; cy3 = cell->face(iface)->vertex(0)[1]; //cz3 = cell->face(iface)->vertex(0)[2];
                                cx4 = cell->face(iface)->vertex(1)[0]; cy4 = cell->face(iface)->vertex(1)[1]; //cz4 = cell->face(iface)->vertex(1)[2];

                                // Sometimes the faces are oriented in such a way that vertices 0 and 1 have the same x and y coordinates
                                // In such cases we use the 2nd point
                                if (Point<2>(cx3,cy3).distance(Point<2>(cx4,cy4)) < 0.1){
                                    cx4 = cell->face(iface)->vertex(2)[0]; cy4 = cell->face(iface)->vertex(2)[1]; //cz4 = cell->face(iface)->vertex(2)[2];
                                }

                                /*{// Debug code
                                    Point<2> dbg(32374.5,1359430);
                                    if (dbg.distance(Point<2>(cx3,cy3)) < 100 || dbg.distance(Point<2>(cx4,cy4)) < 100){
                                        int iii = 0;
                                        dummy_function(true,iii);
                                    }
                                }*/


                                if (boundary_parts[i].Xcoords.size() == 2){
                                    double lx1,ly1,lx2,ly2; // variables for storing the boundary coordinates

                                    lx1 = boundary_parts[i].Xcoords[0]; ly1 = boundary_parts[i].Ycoords[0];
                                    lx2 = boundary_parts[i].Xcoords[1]; ly2 = boundary_parts[i].Ycoords[1];
                                    //double L = sqrt(pow(lx2 - lx1, 2) + pow(ly2 - ly1, 2));


                                    // Next we will calculate the distance of the two cell points from the boundary line
                                    double dst1 = distance_point_line(cx3,cy3,lx1,ly1,lx2,ly2);
                                    double dst2 = distance_point_line(cx4,cy4,lx1,ly1,lx2,ly2);

                                    // The cell face is collinear with the boundary line if the distances is very close to zero
                                    // and one of the two distances is positive.
                                    bool are_collinear = false;
                                    if (std::abs(dst1) < 20 && std::abs(dst2) < 20){
                                        if ( !(dst1 < 0) || !(dst2 < 0)){
                                            are_collinear = true;
                                        }
                                        else{ // It maybe possible due to numerical errors that the distances are both negative
                                            // This can happen under two circumstances.
                                            // 1) The boundary line is smaller than the cell face. This means that the boundary condition
                                            //    lines have not been set correctly.
                                            //    FUTURE VERSION OF THE CODE SHOULD ADDRESS THIS CASE
                                            // 2) The boundary segment is identical with the cell face. Then it is possible that both points
                                            //    of the cell may appear outside of the boundary by very small amount.

                                            //====== Case 2 ======
                                            {
                                                double min_dst1 = std::min(distance_2_points(cx3,cy3,lx1,ly1),distance_2_points(cx3,cy3,lx2,ly2));
                                                double min_dst2 = std::min(distance_2_points(cx4,cy4,lx1,ly1),distance_2_points(cx4,cy4,lx2,ly2));
                                                if (min_dst1 < 0.1 && min_dst2 < 0.1){
                                                    are_collinear = true;
                                                }
                                            }
                                        }
                                        if (are_collinear){
                                            // the face is colinear with the boundary however we will do an extra check using cgal methods
                                            //CGAL::Segment_2< exa_Kernel > segm(exa_Point2(lx1,ly1),exa_Point2(lx2,ly2));
                                            //if (segm.collinear_has_on(exa_Point2(cx3,cy3)) || segm.collinear_has_on(exa_Point2(cx4,cy4))){
                                            cell->face(iface)->set_all_boundary_ids(JJ+i);
                                            cell->face(iface)->set_user_index(1);
                                            //print_cell_face_matlab<dim>(cell,iface);
                                            break;
                                            //}
                                        }
                                    }
                                }
                                else if (boundary_parts[i].Xcoords.size() == 0){
                                    // If there are no coordinates defined in the boundary then it must be an interpolation function of type
                                    // boundary line
                                    Point<dim> A, B;
                                    A[0] = cx3; A[1] = cy3;
                                    B[0] = cx4; B[1] = cy4;
                                    if (interp_funct[i].is_face_part_of_BND(A,B)){
                                        cell->face(iface)->set_all_boundary_ids(JJ+i);
                                        cell->face(iface)->set_user_index(1);
                                        break;
                                    }
                                }
                                else
                                    std::cerr << "This boundary with id " << i << " has not 2 or 0 coordinates" << std::endl;
                            }
                        }
                    }
                }
            }
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



template<int dim>
class Neumann{
public:
    //! A constructor that does nothing
    Neumann();

    double interpolate(Point<dim> p, int ibnd);
    bool getData(std::string filename);
    int Nbnd();
    BoundaryType getType(int i);

private:
    int Nb = 0;
    std::vector<BoundPrim> boundary_parts;
    std::vector<InterpInterface<dim> > interp_funct;
    std::vector<BoundaryType> bnd_type;
    std::vector<double> multiplier;

};

template<int dim>
Neumann<dim>::Neumann(){
    Nb = 0;
}

template<int dim>
int Neumann<dim>::Nbnd(){
    return Nb;
}

template<int dim>
BoundaryType Neumann<dim>::getType(int i){
    return bnd_type[i];
}

template<int dim>
double Neumann<dim>::interpolate(Point<dim> p, int ibnd){

    double out = 0.0;
    if (dim == 2){
        std::cout << "Neuman conditions in 2D are not impemented yet" << std::endl;
    }
    else if (dim == 3){
        if (bnd_type[ibnd] == BoundaryType::BOT){
            if (boundary_parts[ibnd].is_any_point_insideBB(p[0], p[1])){
                if (boundary_parts[ibnd].is_point_inside_poly(p[0], p[1])){
                    out = interp_funct[ibnd].interpolate(p) * multiplier[ibnd];
                }
            }
        }
        else if (bnd_type[ibnd] == BoundaryType::VER){
            std::cout << "Vertical Neuman conditions in 3D are not impemented yet" << std::endl;
        }
    }
    return out;
}

template<int dim>
bool Neumann<dim>::getData(std::string filename){
    std::ifstream  datafile(filename.c_str());
    if (!datafile.good()){
        std::cout << "Can't open " << filename << std::endl;
        return false;
    }
    else{
        std::string line, tmp;
        {// NBND number of boundaries
            getline(datafile, line);
            std::istringstream inp(line.c_str());
            inp >> Nb;
            boundary_parts.resize(Nb);
            interp_funct.resize(Nb);
            bnd_type.resize(Nb);
            multiplier.resize(Nb);
        }
        for (int i = 0; i < Nb; ++i){
            std::string type, value;
            int np;
            double mult;

            {// TYPE N VALUE (For each boundary)
                getline(datafile, line);
                std::istringstream inp(line.c_str());
                inp >>type;
                if (type.compare("BOT") == 0){
                    bnd_type[i] = BoundaryType::BOT;
                }
                else if (type.compare("VER") == 0){
                    bnd_type[i] = BoundaryType::VER;
                }
                else{
                    std::cout << "Unknown type of Neumann condition: " << type << std::endl;
                }
                inp >> np;
                inp >> mult;
                multiplier[i] = mult;
                inp >> value;
            }
            if (dim == 2){
                std::cout << "Neumann readers are not implemented yet for 2D" << std::endl;
                return false;
            }
            else if (dim == 3){
                switch (bnd_type[i]) {
                case BoundaryType::BOT:
                {// in BOT case the geometry is a polygon
                    double x, y;
                    std::vector<boost_point> tmp_pnts;
                    boundary_parts[i].BBmin[0] = 99999999999;
                    boundary_parts[i].BBmin[1] = 99999999999;
                    boundary_parts[i].BBmax[0] = -99999999999;
                    boundary_parts[i].BBmax[1] = -99999999999;
                    for (int iv = 0; iv < np; ++iv){
                        getline(datafile, line);
                        std::istringstream inp(line.c_str());
                        inp >> x;
                        // X coordinate
                        if (x < boundary_parts[i].BBmin[0])
                            boundary_parts[i].BBmin[0] = x;
                        if (x > boundary_parts[i].BBmax[0])
                            boundary_parts[i].BBmax[0] = x;
                         boundary_parts[i].Xcoords.push_back(x);

                         // Y coordinate
                         inp >> y;
                         if (y < boundary_parts[i].BBmin[1])
                             boundary_parts[i].BBmin[1] = y;
                         if (y > boundary_parts[i].BBmax[1])
                             boundary_parts[i].BBmax[1] = y;
                         boundary_parts[i].Ycoords.push_back(y);
                         tmp_pnts.push_back(boost_point(x,y));

                    }
                    boost::geometry::assign_points(boundary_parts[i].poly, tmp_pnts);
                    boost::geometry::correct(boundary_parts[i].poly);
                    interp_funct[i].get_data(value);
                }
                break;
                case BoundaryType::VER:{
                    std::cout << "Neumann readers are not implemented yet for VER type" << std::endl;
                }
                break;
                }
            }
        }
    }
    datafile.close();
    return true;
}

}

#endif // DIRICHLET_BOUNDARY_H
