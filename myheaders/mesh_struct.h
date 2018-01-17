#ifndef MESH_STRUCT_H
#define MESH_STRUCT_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/base/conditional_ostream.h>

#include "zinfo.h"
#include "pnt_info.h"
#include "cgal_functions.h"
#include "my_functions.h"

//! Returns true if any neighbor element is ghost
template <int dim>
bool any_ghost_neighbor(typename DoFHandler<dim>::active_cell_iterator cell){
    bool out = false;
    for (unsigned int iface = 0; iface < GeometryInfo<dim>::faces_per_cell; ++iface){
        if (cell->at_boundary(iface)) continue;

        if (cell->neighbor(iface)->active()){
            if (cell->neighbor(iface)->is_ghost()){
                out = true;
                break;
            }
        }
        else{
            for (unsigned int ichild = 0; ichild < cell->face(iface)->n_children();  ++ichild){
                if (cell->neighbor_child_on_subface(iface,ichild)->is_ghost()){
                    out = true;
                    break;
                }
            }
            if (out)
                break;
        }
    }
    return out;
}

/*!
 * \brief The Mesh_struct class contains the coordinates of the entire mesh grouped into lists of dim-1 points,
 * where each point contains a list of the z node elevation.
 */
template <int dim>
class Mesh_struct{
public:

    /*!
     * \brief Mesh_struct The constructor just sets the threshold values
     * \param xy_thr this is the threshold for the x-y coordinates. Two point with distance smaller that xy_thr
     * are considered identical. Typically the xy threshold has larger values compared to the z threshold.
     * \param z_thr this is the treshold for the z coordinates. Two nodes with the same x-y coordinates
     * are considered identical if their elevation is smaller that the z_thr
     */
    Mesh_struct(double xy_thr, double z_thr);

    //! The threshold along the x-y coordinates
    double xy_thres;
    //! The threshold along the z coordinates
    double z_thres;

    //! This is a counter for the points in the #PointsMap
    int _counter;

    //! This map associates each point with a unique id (#_counter)
    std::map<int , PntsInfo<dim> > PointsMap;

    //! @@@@@@@@@@@@@@@@@@@@@@@@@@@@@WHAT IS THIS SUPPOSED TO DO????
    std::map<int,std::pair<int,int> > dof_ij;

    //! this is a cgal container of the points of this class stored in an optimized way for spatial queries
    PointSet2 CGALset;

    //! Adds a new point in the structure. If the point exists adds the z coordinate only and returns
    //! the id of the existing point. if the point doesnt exist creates a new point and returns the new id.
    int add_new_point(Point<dim-1>, Zinfo zinfo);

    //! Checks if the point already exists in the mesh structure
    //! If the point exists it returns the id of the point in the #CGALset
    //! otherwise returns -9;
    int check_if_point_exists(Point<dim-1> p);

    /*!
     * \brief updateMeshstruct is the heart of this class. for a given parallel triangulation updates the existing
     * points or creates new ones.
     *
     * The method first loops through the locally owned cells and extracts the coordinates and dof for each coordinate
     * which stores it to #distributed_mesh_vertices.
     * \param xy_thr this is the threshold for the x-y coordinates. Two point with distance smaller that xy_thr
     * are considered identical. Typically the xy threshold has larger values compared to the z threshold.
     * \param distributed_mesh_vertices is a vector of size #dim x (Number of triangulation vertices).
     * Essentially we treat all the coordinates as unknowns yet only the vertical component is the one we are going
     * to change
     */
    void updateMeshStruct(DoFHandler<dim>& mesh_dof_handler,
                         FESystem<dim>& mesh_fe,
                         ConstraintMatrix& mesh_constraints,
                         IndexSet& mesh_locally_owned,
                         IndexSet& mesh_locally_relevant,
                         TrilinosWrappers::MPI::Vector& mesh_vertices,
                         TrilinosWrappers::MPI::Vector& distributed_mesh_vertices,
                         MPI_Comm&  mpi_communicator,
                         ConditionalOStream pcout);

    //! resets all the information that is contained except the coordinates and the level of the points
    void reset();

    //! Prints to screen the number of vertices the #myrank processor has.
    //! It is used primarily for debuging
    void n_vertices(int myrank);

    //!
    void dbg_set_scales(double xscale, double zscale);
private:
    void dbg_meshStructInfo2D(unsigned int n_proc);
    void dbg_meshStructInfo3D(unsigned int n_proc);
    double dbg_scale_x;
    double dbg_scale_z;


};

template <int dim>
Mesh_struct<dim>::Mesh_struct(double xy_thr, double z_thr){
    xy_thres = xy_thr;
    z_thres = z_thr;
    _counter = 0;
}

template <int dim>
int Mesh_struct<dim>::add_new_point(Point<dim-1>p, Zinfo zinfo){
    int outcome = -99;

    // First search for the XY location in the structure
    int id = check_if_point_exists(p);
    if ( id < 0 ){
        // this is a new point and we add it to the map
        PntsInfo<dim> tempPnt(p, zinfo);
        PointsMap[_counter] = tempPnt;

        //... to the Cgal structure
        std::vector< std::pair<ine_Point2,unsigned> > pair_point_id;
        double x,y;
        if (dim == 2){
            x = p[0];
            y = 0;
        }else if (dim == 3){
            x = p[0];
            y = p[1];
        }
        pair_point_id.push_back(std::make_pair(ine_Point2(x, y), _counter));
        CGALset.insert(pair_point_id.begin(), pair_point_id.end());
        outcome = _counter;
        _counter++;
    }else if (id >=0){
        typename std::map<int, PntsInfo<dim> >::iterator it = PointsMap.find(id);
        it->second.add_Zcoord(zinfo, z_thres);
        outcome = it->first;
    }

    return outcome;

}

template <int dim>
int Mesh_struct<dim>::check_if_point_exists(Point<dim-1> p){
    int out = -9;
    double x,y;
    if (dim == 2){
        x = p[0];
        y = 0;
    }else if (dim == 3){
        x = p[0];
        y = p[1];
    }

    std::vector<int> ids = circle_search_in_2DSet(CGALset, ine_Point3(x, y, 0.0) , xy_thres);

    if (ids.size() > 1)
        std::cerr << "More than one points around x: " << x << ", y: " << y << "found within the specified tolerance" << std::endl;
    else if(ids.size() == 1) {
         out = ids[0];
    }
    else{
         out = -9;
    }

    return out;
}

template <int dim>
void Mesh_struct<dim>::updateMeshStruct(DoFHandler<dim>& mesh_dof_handler,
                                       FESystem<dim>& mesh_fe,
                                       ConstraintMatrix& mesh_constraints,
                                       IndexSet& mesh_locally_owned,
                                       IndexSet& mesh_locally_relevant,
                                       TrilinosWrappers::MPI::Vector& mesh_vertices,
                                       TrilinosWrappers::MPI::Vector& distributed_mesh_vertices,
                                       MPI_Comm&  mpi_communicator,
                                       ConditionalOStream pcout){
    // Use this to time the operation. Note that this is a very expensive operation but nessecary
    std::clock_t begin_t = std::clock();
    // get the rank and processor id just for output display
    unsigned int my_rank = Utilities::MPI::this_mpi_process(mpi_communicator);
    unsigned int n_proc = Utilities::MPI::n_mpi_processes(mpi_communicator);

    // make sure all processors start together
    MPI_Barrier(mpi_communicator);
    reset(); // delete all the information except the coordinates
    MPI_Barrier(mpi_communicator);

    const MappingQ1<dim> mapping;

    pcout << "Distribute mesh dofs..." << std::endl << std::flush;
    mesh_dof_handler.distribute_dofs(mesh_fe);
    mesh_locally_owned = mesh_dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs (mesh_dof_handler, mesh_locally_relevant);
    mesh_vertices.reinit (mesh_locally_owned, mesh_locally_relevant, mpi_communicator);
    distributed_mesh_vertices.reinit(mesh_locally_owned, mpi_communicator);

    const std::vector<Point<dim> > mesh_support_points
                                  = mesh_fe.base_element(0).get_unit_support_points();

    FEValues<dim> fe_mesh_points (mapping,
                                  mesh_fe,
                                  mesh_support_points,
                                  update_quadrature_points);

    mesh_constraints.clear();
    mesh_constraints.reinit(mesh_locally_relevant);
    DoFTools::make_hanging_node_constraints(mesh_dof_handler, mesh_constraints);
    mesh_constraints.close();

    // to avoid duplicate executions we will maintain a map with the dofs that have been
    // already processed
    std::map<int,int> dof_local;// the key are the dof and the value is the _counter
    std::map<int,int>::iterator itint;
    MPI_Barrier(mpi_communicator);

    pcout << "Update XYZ structure..." << std::endl << std::flush;
    std::vector<unsigned int> cell_dof_indices (mesh_fe.dofs_per_cell);
    typename DoFHandler<dim>::active_cell_iterator
        cell = mesh_dof_handler.begin_active(),
        endc = mesh_dof_handler.end();
    for (; cell != endc; ++cell){
        if (cell->is_locally_owned()){
            fe_mesh_points.reinit(cell);
            cell->get_dof_indices (cell_dof_indices);
            for (unsigned int idof = 0; idof < mesh_fe.base_element(0).dofs_per_cell; ++idof){
                // for each dof of this cell we extract the coordinates and the dofs
                Point <dim> current_node;
                std::vector<int> current_dofs(dim);
                for (unsigned int dir = 0; dir < dim; ++dir){
                    // for each cell support_point_index spans from 0 to dim*Nvert_per_cell-1
                    // eg for dim =2 spans from 0-7
                    // The first dim indices correspond to x,y,z of the first vertex of triangulation
                    // The current_dofs contains the dof index for each coordinate.
                    // The current_node containts the x,y,z coordinates
                    // The distributed_mesh_vertices is a vector of size Nvertices*dim
                    // essentially we are treating all xyz coordinates as variables although we are going to
                    // change only the vertical component of it (In 2D this is the y).
                    unsigned int support_point_index = mesh_fe.component_to_system_index(dir, idof );
                    current_dofs[dir] = static_cast<int>(cell_dof_indices[support_point_index]);
                    current_node[dir] = fe_mesh_points.quadrature_point(idof)[dir];
                    distributed_mesh_vertices[cell_dof_indices[support_point_index]] = current_node[dir];

                    pcout << "dir:" << dir << ", idof:" << idof << ", cur_dof:" << current_dofs[dir]
                          <<   ", cur_nd:" << current_node[dir] << ", spi:" << support_point_index << std::endl;
                }

                // find if we have already check this point
                itint = dof_local.find(current_dofs[dim-1]);
                if (itint != dof_local.end()){
                    // MAYBE THATS NOT NEEDED
                    // if the point is already inside the map we just need to check if this cell
                    // has any ghost neighbors. If yes then we flag the XY location for sharing.
                    // We do that of we have not set the sharing flag before
                }else{
                    // we have to add this point to the structure.
                    Zinfo zinfo(current_node[dim-1], current_dofs[dim-1], cell->level(),
                            mesh_constraints.is_constrained(current_dofs[dim-1]));
                    // ADD THE CODE ABOUT THE BOUNDARY INFO

                    Point<dim-1> ptemp;
                    for (unsigned int d = 0; d < dim-1; ++d)
                        ptemp[d] = current_node[d];

                    int id_in_map = add_new_point(ptemp, zinfo);
                    if (id_in_map < 0)
                        std::cerr << "Something went really wrong while trying to insert a new point into the mesh struct" << std::endl;
                    else{
                        bool tf = any_ghost_neighbor<dim>(cell);
                        if (tf)
                            PointsMap[id_in_map].have_to_send = 1;
                    }
                    dof_local[current_dofs[dim-1]] = id_in_map;
                }
            }
        }
        pcout << "----------------------------------------------------------------" << std::endl;
    }

    MPI_Barrier(mpi_communicator);
    distributed_mesh_vertices.compress(VectorOperation::insert);
    MPI_Barrier(mpi_communicator);

    dbg_meshStructInfo2D(my_rank);
    dbg_meshStructInfo3D(my_rank);


    if (n_proc > 1){
        pcout << "exchange vertices between processors..." << std::endl << std::flush;
        // All vertices have been added to the PointsMap structure.
        // we loop through each vertex and store to a separate vector
        // those that require communication and they are actively used

        std::vector<std::vector<double> > Xcoord(n_proc);
        std::vector<std::vector<double> > Ycoord(n_proc);
        std::vector<std::vector<double> > Zcoord(n_proc);
        std::vector<std::vector<int> > id_above(n_proc);// this are not needed at this time
        std::vector<std::vector<int> > id_below(n_proc);// they should be -9 but have to double check within a loop
        std::vector<std::vector<int> > is_hanging(n_proc);
        std::vector<std::vector<int> > dof(n_proc);
        std::vector<std::vector<int> > level(n_proc);
        std::vector<std::vector<int> > istart(n_proc);
        std::vector<std::vector<int> > iend(n_proc);

        typename std::map<int ,  PntsInfo<dim> >::iterator it;
        for (it = PointsMap.begin(); it != PointsMap.end(); ++it){
            if (it->second.have_to_send == 1){
                if (it->second.number_of_positive_dofs() > 0){
                    Xcoord[my_rank].push_back(it->second.PNT[0]);
                    if (dim ==3)
                        Ycoord[my_rank].push_back(it->second.PNT[1]);
                    istart[my_rank].push_back(Zcoord[my_rank].size());
                    std::vector<Zinfo>::iterator itz = it->second.Zlist.begin();
                    for (; itz != it->second.Zlist.end(); ++itz){
                        // we will send only those with meaningful dof
                        if (itz->dof >= 0){
                            Zcoord[my_rank].push_back(itz->z);
                            id_above[my_rank].push_back(itz->id_above);// maybe not needed at this time
                            id_below[my_rank].push_back(itz->id_below);// maybe not needed at this time
                            is_hanging[my_rank].push_back(itz->hanging);
                            dof[my_rank].push_back(itz->dof);
                            level[my_rank].push_back(itz->level);
                        }
                    }
                    iend[my_rank].push_back(Zcoord[my_rank].size()-1);
                }
            }
        }




    }//if (n_proc > 1)











}

template <int dim>
void Mesh_struct<dim>::reset(){
    typename std::map<int , PntsInfo<dim> >::iterator it;
    for (it = PointsMap.begin(); it != PointsMap.end(); ++it){
        it->second.reset();
    }
    dof_ij.clear();
}

template <int dim>
void Mesh_struct<dim>::n_vertices(int myrank){
    int Nxy = PointsMap.size();
    int Nz = 0;
    typename std::map<int , PntsInfo<dim> >::iterator it;
    for (it = PointsMap.begin(); it != PointsMap.end(); ++it){
        Nz += it->second.Zlist.size();
    }
    std::cout << "I'm " << myrank << ", Nxy = " << Nxy << ", Nz = " << Nz << std::endl;
}

template <int dim>
void Mesh_struct<dim>::dbg_meshStructInfo2D(unsigned int my_rank){
    const std::string log_file_name = ("MshStruct2D"	+
                                       Utilities::int_to_string(my_rank, 4) +
                                       ".txt");
     std::ofstream log_file;
     log_file.open(log_file_name.c_str());
     typename std::map<int , PntsInfo<dim> >::iterator it;
     for (it = PointsMap.begin(); it != PointsMap.end(); ++it){
         double x,y,z;
         x = it->second.PNT[0]/dbg_scale_x;
         if (dim == 3) z = it->second.PNT[1]/dbg_scale_x;
         else z = 0;
         y = 0;
         log_file << std::setprecision(3)
                  << std::fixed
                  << std::setw(15) << x << ", "
                  << std::setw(15) << y << ", "
                  << std::setw(15) << z << ", "
                  << std::setw(15) << it->second.T << ", "
                  << std::setw(15) << it->second.B << ", "
                  << std::setw(5) << it->second.have_to_send
                  << std::endl;
     }
     log_file.close();

}

template <int dim>
void Mesh_struct<dim>::dbg_meshStructInfo3D(unsigned int my_rank){
    const std::string log_file_name = ("MshStruct3D"	+
                                       Utilities::int_to_string(my_rank, 4) +
                                       ".txt");
     std::ofstream log_file;
     log_file.open(log_file_name.c_str());
     typename std::map<int , PntsInfo<dim> >::iterator it;
     for (it = PointsMap.begin(); it != PointsMap.end(); ++it){
         std::vector<Zinfo>::iterator itz = it->second.Zlist.begin();
         for (; itz != it->second.Zlist.end(); ++itz){
             double x,y,z;
             x = it->second.PNT[0]/dbg_scale_x;
             if (dim == 3) z = it->second.PNT[1];
             else z = 0;
             y = itz->z;
             log_file << std::setprecision(3)
                      << std::fixed
                      << std::setw(15) << x << ", "
                      << std::setw(15) << y << ", "
                      << std::setw(15) << z << ", "
                      << std::setw(15) << itz->dof << ", "
                      << std::setw(15) << itz->level << ", "
                      << std::setw(15) << itz->id_above  << ", "
                      << std::setw(15) << itz->id_below << ", "
                      << std::setw(15) << itz->level << ", "
                      << std::setw(15) << itz->id_top  << ", "
                      << std::setw(15) << itz->id_bot << ", "
                      << std::setw(15) << itz->rel_pos  << ", "
                      << std::setw(15) << itz->hanging << ", "
                      << std::setw(15) << it->second.T << ", "
                      << std::setw(15) << it->second.B << ", "
                      << std::setw(5) << it->second.have_to_send
                      << std::endl;

         }
     }
     log_file.close();
}

template <int dim>
void Mesh_struct<dim>::dbg_set_scales(double xscale, double zscale){
    dbg_scale_x = xscale;
    dbg_scale_z = zscale;
}

#endif // MESH_STRUCT_H
