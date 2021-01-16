#ifndef STREAMS_H
#define STREAMS_H

#include <deal.II/base/point.h>
#include <deal.II/distributed/tria.h>
//#include <deal.II/grid/tria_accessor.h>
//#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/constraint_matrix.h>

//#include "my_functions.h"
#include "cgal_functions.h"
#include "helper_functions.h"
#include "boost_functions.h"
#include "mpi_help.h"

using namespace dealii;

/*!
 * \brief The Streams class provides the required functionality to deal with stream sources.
 *
 * Although the class is templated it make sence to use it only in 3D. In fact the code is not designed for other than 3D.
 *
 * Streams are assumed to be polygon entities that exist on the top of the aquifer.
 * The user defines the streams as line segments where each segment is associated with a stream rate and width.
 * The program converts the line segments into orthogonal four sided polygons. These polygons are then converted
 * into triangles because all the spatial queries are designed to work with triangles. This will make easier to adapt
 * the code to a case where the rivers are actual polygons.
 *
 *
 *
 *
 */
template <int dim>
class Streams{
public:
    //! This constructor prepares the necessary data structures
    Streams();

    /*! \brief Reads the stream input file
     * \param namefile is the name of the stream input file
     * \return True upon successful reading
     *
     * The format of the file is
     * N_seg = the number of river segments
     *
     * Repeat N_seg times the following:
     *
     * N_points Q_rate [Width]
     *
     * where N_points is the number of points associated with the segment,
     * Q_rate is the recharge rate of water that is dischagred/recharged in the aquifer [L/T]
     * with is used only if N_points is 2, and represents the width of the river
     *
     * The next N_points lines correspond to the coordinates of the rivel segment
     *
     * X_node Y_node
     * .        .
     * .        .
     *
     */

    bool read_streams(std::string namefile);
    //! A list with the coordinates of the starting points of the stream.
    //! Note that it is not important which end is set as starting or ending pooint
    //std::vector<Point<dim-1>>    A;
    //! A list with the coordinates of the ending points of the stream.
    //std::vector<Point<dim-1> >    B;
    //! A list of the recharge or discharge rates
    std::vector<double>         Q_rate;
    //! A list of the stream line lenghts. These calculated by the program
    std::vector<double>         length;
    //! A list of the stream line widths.
    //std::vector<double>         width;
    //! A list of triangles
    ineTriangle_list            stream_triangles;
    //! A list of stream ids. The stream id depends by the order they are listed in the input file. It set by the program
    std::vector<int>            stream_ids;
    //! A tree structures which holds the streams.
    //! (This is a pointer because I was getting compile errors due to calling a private constructor of class.
    //! Setting this as a point and adding a construction code on the class constructor made the code at least to compile
    //ine_Tree*                    stream_tree;
    //! The number of line segments
    unsigned int N_seg;
    //! A list of stream outlines. Each stream outline consists of a number of points which define the shape of the stream.
    //! Currently 4-point outlines is used i.e.
    //! Future version may allow users to define the shape of the rivers as polygons
    //std::vector<std::vector<Point<dim-1>>> river_BBox;
    //std::vector<std::vector<Point<dim-1>>> river_outline;
    //! This is a list of the maximum x point of each stream outline.
    //! It is used to define a bounding box for each stream segment to avoid unecessary computations.
    std::vector<double> Xmax;
    //! A list of minimum x point of each stream outline
    std::vector<double> Xmin;
    //! A list of minimum y point of each stream outline
    std::vector<double> Ymin;
    //! A list of maximum y point of each stream outline
    std::vector<double> Ymax;

    //! A list of the X coordinates of the stream outlines
    //! std::vector<std::vector<double> > Xoutline;
    std::vector<std::vector<double> > Xpoly;
    //! A list of the Y coordinates of the stream outlines
    //std::vector<std::vector<double> > Youtline;
    std::vector<std::vector<double> > Ypoly;

    //! Calculate the stream rate that corresponds to the point p
    //!
    //! The function loops through the Streams#N_seg and checks if the point in question
    //! is inside the stream bounding box. If yes use the CellAccessor< dim, spacedim >::point_inside
    //! method to check whether the point is inside the river outline. Before that modifies the cell used by the CellAccessor
    //! to match the outline of the river segment.
    //!
    //! \param p The point to calculate the rate.
    //! \return the rate that is associated with the line segments times the intersection area, if the point is found inside the stream outline.
    //double get_stream_rate(Point<dim> p)const;

    /*! Checks if there is any intersection of a triangulation cell with any stream segment
     *
     * First it uses a CGAL method to identify which river triangles, that are stored into the #stream_tree
     * are intersected with the triangulation cell. The triangulation cell is the top face of a 3D cell, that is also a boundary.
     * The vectors #xp, #yp contain the coordinates of the tested cell. The order of coordinates should be counterclockwise,
     * and not in the order that dealii uses because it creates self intersecting polygons.
     *
     * Once we have a list of triangles that overlap with the given cell we loop through them.
     * In the case that rivers are defined as lines which are converted into polygons each polygon is split into
     * 2 triangles. Therefore, from the list of intersected triangles we make a list of unique river polygons that
     * intersect the given cell face. Then for each polygon we use boost methods to do a detailed intersection between
     * the river polygon and the cell face. This will return the centroid of the intersection and the area.
     * We multiply the intersected area with the stream rate to calculate the stream volume of this river intersection.
     *
     *
    * \param xc is a list of the x coordinates of the centroid of the intersected area
    * \param yc is a list of the y coordinates of the centroid of the intersected area
    * \param Q is the recharge rate of intersected segment
    * \param xp is a list of the x coordinates of the triangulation cell
    * \param yp is a list of the y coordinates of the triangulation cell
    * \param stream_tree A list of triangles that represent the river network
    * \return True if this cell has at least one intersection with any stream.
    */
    bool get_stream_recharge(std::vector<double>& xc,
                             std::vector<double>& yc,
                             std::vector<double>& Q,
                             std::vector<double> xp,
                             std::vector<double> yp,
                             ine_Tree &stream_tree,
                             double& ar);

    /*! Calculate the contributions to the Right Hand side vector from the streams
    *
    * The function loops through the triangulation cells. For the cells that the top face is boundary
    * the method Streams#get_stream_recharge is used to check whether the current cell intersects any stream segment.
    * The method returns also the coordinates of the centroids of each river intersection with the tested face
    * abd the rate associated with it
    *
    * For each intersection an 1-point quadrature (on the centroid) formula is defined and the contrubition for each stream intersection
    * is added accordinlgy to the RHS vector system_rhs
    *
    *
    * \param system_rhs Right hand side vector
    * \param dof_handler is the typical deal dof_handler
    * \param fe is the typical deal FE_Q
    * \param constraints is the typical deal constraint matrix
    * \param top_boundary_ids is a list of ids that correspond to the top faces
    */
    void add_contributions(TrilinosWrappers::MPI::Vector& system_rhs,
                           const DoFHandler<dim>& dof_handler,
                           const FE_Q<dim>& fe,
                           const AffineConstraints<double>& constraints,
                           std::vector<int> top_boundary_ids,
                           int my_rank, int n_proc, MPI_Comm mpi_communicator);

    //! This function loops through the cells and flags for refinement those that their top face intersects with a stream
    void flag_cells_for_refinement(parallel::distributed::Triangulation<dim>& 	triangulation);

    double stream_multiplier = 1.0;

private:
    //! a temporary 1-cell triangulation which is used as way to access methods that operate in 2 dimensions
    //Triangulation<dim-1> tria;
    //! a temporary 1-cell triangulation which is used as way to access method that operate in 2 dimensions
    //Triangulation<dim-1> river_rect;
    //! This method modifies the shape of the Streams#river_rect triangulation to match the shape of the input outline
    //! \param outline is a list of points which describe a stream segment. The number of points has to be 4
    void setup_river_rect(std::vector<Point<dim>> outline, Triangulation<dim-1> &river_rect )const;
    //! Changes the shape of the cell of the Streams#tria triangulation according to the shape of the top_face
    //! \param top_face is the face of the cell we need to convert into a 2D cell to obtain access to several cell methods
    void setup_cell(typename DoFHandler<dim>::face_iterator top_face, Triangulation<dim-1> &tria);
    //! Converts the stream line segments into rectangular areas
    //!  \param xx are the x coordinates of the rectangular stream
    //! \param yy are the y coordinates of the rectangular stream
    //! \param A is one end of the line segment
    //! \param B is the other end of the line segment
    //! \param width the width of the stream.
    void create_river_outline(std::vector<double>& xx,
                              std::vector<double>& yy,
                              Point<dim-1> A, Point<dim-1> B, double width);
};

template <int dim>
Streams<dim>::Streams()
{
    N_seg = 0;
}


template <int dim>
void Streams<dim>::setup_cell(typename DoFHandler<dim>::face_iterator top_face, Triangulation<dim-1> &tria){
    typename Triangulation<dim-1 >::active_cell_iterator cell = tria.begin_active();
    for (unsigned int i=0; i<GeometryInfo<dim-1>::vertices_per_cell; ++i){
        Point<dim-1> &v = cell->vertex(i);
        v[0] = top_face->vertex(i)[0];
        v[1] = top_face->vertex(i)[1];
    }
}

template <int dim>
bool Streams<dim>::Streams::read_streams(std::string namefile){
    std::ifstream  datafile(namefile.c_str());
    char buffer[512];
    if (!datafile.good()){
        std::cout << "Can't open the file" << namefile << std::endl;
        return false;
    }
    else{
        {	// read the number of river segments
            datafile.getline(buffer,512);
            std::istringstream inp(buffer);
            inp >> N_seg;
        }
        {// read the river segments info
            //A.resize(N_seg);
            //B.resize(N_seg);
            length.resize(N_seg);
            Q_rate.resize(N_seg);
            //width.resize(N_seg);
            Xpoly.resize(N_seg);
            Ypoly.resize(N_seg);
            Xmin.resize(N_seg);
            Xmax.resize(N_seg);
            Ymin.resize(N_seg);
            Ymax.resize(N_seg);

            double x, y, q, w;
            for (unsigned int i = 0; i < N_seg; ++i){
                unsigned int N_points;
                {
                    datafile.getline(buffer,512);
                    std::istringstream inp(buffer);
                    inp >> N_points;
                    if (N_points < 2 || N_points > 4 )
                        std::cerr << "The stream segment " << i << " consists of " << N_points << " points." << std::endl;
                    inp >> q;
                    Q_rate[i] = q;
                    if (N_points == 2)
                        inp >> w;
                }

                std::vector<double> xx;
                std::vector<double> yy;

                if (N_points == 2){
                    Point<dim-1> A, B;
                    {//Point A
                        datafile.getline(buffer,512);
                        std::istringstream inp(buffer);
                        inp >> x; inp>> y;
                        if (dim == 3){
                            A[0] = x; A[1] = y;
                        }
                        else{
                            std::cout << "Streams cannot be defined in problems other than 3D" << std::endl;
                        }
                    }
                    {//Point B
                        datafile.getline(buffer,512);
                        std::istringstream inp(buffer);
                        inp >> x; inp>> y;
                        if (dim == 3){
                            B[0] = x; B[1] = y;
                        }
                        else{
                            std::cout << "Streams cannot be defined in problems other than 3D" << std::endl;
                        }
                    }
                    create_river_outline(xx, yy, A, B, w);
                }
                else{
                    for (unsigned int j = 0; j < N_points; ++j){
                        datafile.getline(buffer,512);
                        std::istringstream inp(buffer);
                        inp >> x; inp>> y;
                        xx.push_back(x);
                        yy.push_back(y);
                    }
                }
                Xpoly[i] = xx;
                Ypoly[i] = yy;
                Xmin[i] = 100000000; Xmax[i] = -100000000;
                Ymin[i] = 100000000; Ymax[i] = -100000000;
                for (unsigned j = 0; j < xx.size(); ++j){
                    if (xx[j] > Xmax[i])
                        Xmax[i] = xx[j];
                    if (xx[j] < Xmin[i])
                        Xmin[i] = xx[j];
                    if (yy[j] > Ymax[i])
                        Ymax[i] = yy[j];
                    if (yy[j] < Ymin[i])
                        Ymin[i] = yy[j];
                }

                stream_triangles.push_back(ine_Triangle(ine_Point3(xx[0], yy[0], 0.0),
                                                        ine_Point3(xx[1], yy[1], 0.0),
                                                        ine_Point3(xx[2], yy[2], 0.0)));
                //std::cout << "plot([" << xx[0] << " " << xx[1] << " " << xx[2] << " " <<xx[0] << "],[";
                //std::cout << yy[0] << " " << yy[1] << " " << yy[2] << " " << yy[0] << "])" << std::endl;


                stream_ids.push_back(i);
                if (N_points  == 4){
                    stream_triangles.push_back(ine_Triangle(ine_Point3(xx[2], yy[2], 0.0),
                                               ine_Point3(xx[3], yy[3], 0.0),
                                               ine_Point3(xx[0], yy[0], 0.0)));
                    //std::cout << "plot([" << xx[2] << " " << xx[3] << " " << xx[0] << " " <<xx[2] << "],[";
                    //std::cout << yy[2] << " " << yy[3] << " " << yy[0] << " " << yy[2] << "])" << std::endl;
                    stream_ids.push_back(i);
                }
            }
        }
        //stream_tree->insert(stream_triangles.begin(), stream_triangles.end());
        return true;
    }
}

//template <int dim>
//double Streams<dim>::get_stream_rate(Point<dim> p)const{
//    double stream_rate = 0;
//    // loop through the stream segments
//    for (unsigned int i_seg = 0; i_seg < N_seg; ++i_seg){
//        // if the point in question is inside of the bounding box of the stream
//        // then check whether is inside this stream
//        if (p[0] >= Xmin[i_seg] && p[0] <= Xmax[i_seg] &&
//                p[1] >= Ymin[i_seg] && p[1] <= Ymax[i_seg]){
//            setup_river_rect(river_outline[i_seg], river_rect);
//            typename Triangulation<dim-1>::active_cell_iterator cell2D = river_rect.begin_active();
//            bool test = cell2D->point_inside(Point<dim-1>(p[0], p[1]));
//            if (test){
//                stream_rate = Q_rate[i_seg];
//                break;
//            }
//        }
//    }
//    return stream_rate;
//}

template <int dim>
void Streams<dim>::setup_river_rect(std::vector<Point<dim> > outline, Triangulation<dim-1> &river_rect)const{
    typename Triangulation<dim-1>::active_cell_iterator cell = river_rect.begin_active();
    for (unsigned int i=0; i<GeometryInfo<dim-1>::vertices_per_cell; ++i){
        Point<dim-1> &v = cell->vertex(i);
        v[0] = outline[i][0];
        v[1] = outline[i][1];
    }
}

template <int dim>
void Streams<dim>::create_river_outline(std::vector<double>& xx,
                                        std::vector<double>& yy,
                                        Point<dim-1> A, Point<dim-1> B, double width){
    xx.clear();
    yy.clear();

    if (abs(A[0] - B[0]) < 0.1 && abs(A[1] - B[1]) < 0.1){
        std::cerr << "There are river segments with almost 0 length" << std::endl;
    }
    else if(abs(A[0] - B[0]) < 0.1){// if the river is vertical
        xx.push_back(A[0]-width);    yy.push_back(A[1]);
        xx.push_back(A[0]-width);    yy.push_back(B[1]);
        xx.push_back(B[0]+width);    yy.push_back(B[1]);
        xx.push_back(B[0]+width);    yy.push_back(A[1]);
    }
    else if (abs(A[1] - B[1]) < 0.1) {
        xx.push_back(A[0]);    yy.push_back(A[1]-width);
        xx.push_back(A[0]);    yy.push_back(A[1]+width);
        xx.push_back(B[0]);    yy.push_back(B[1]+width);
        xx.push_back(B[0]);    yy.push_back(B[1]-width);
    }
    else{
        // find slope and intercept
        double m = (B[1] - A[1])/(B[0] - A[0]);
        double b = A[1] - m*A[0];
        // find the intercepts of the two parallel lines at distance width
        double b1 = b - width*sqrt(pow(m,2) + 1);
        double b2 = width*sqrt(pow(m,2) + 1) + b;

        // find the slope and intercept of the perpendicular lines at points A and B
        double m_p = -(1.0/m);
        double b_a = A[1] - m_p*A[0];
        double b_b = B[1] - m_p*B[0];

        double x, y;
        if(line_line_intersection(b_a, m_p, b1, m, x, y)){
            xx.push_back(x);    yy.push_back(y);
        }
        if(line_line_intersection(b_a, m_p, b2, m, x, y)){
            xx.push_back(x);    yy.push_back(y);
        }
        if(line_line_intersection(b_b, m_p, b2, m, x, y)){
            xx.push_back(x);    yy.push_back(y);
        }
        if (line_line_intersection(b_b, m_p, b1, m, x, y)){
            xx.push_back(x);    yy.push_back(y);
        }
    }
}

template <int dim>
bool Streams<dim>::get_stream_recharge(std::vector<double>& xc, std::vector<double>& yc, std::vector<double> &Q,
                                       std::vector<double> xp,
                                       std::vector<double> yp,
                                       ine_Tree& stream_tree,
                                       double& ar){
    std::vector<int> ids;
    xc.clear();
    yc.clear();
    Q.clear();

    bool tf = find_intersection_in_AABB_TREE(stream_tree,
                                             stream_triangles,
                                             xp, yp, ids);
    ar = 0;
    if (tf){
        std::map<int,int> unique_ids;
        // make a unique list of river segment ids
        for (unsigned int i = 0; i < ids.size(); ++i){
            unique_ids[stream_ids[ids[i]]] = stream_ids[ids[i]];
        }
        std::map<int,int>::iterator it = unique_ids.begin();
        for (; it != unique_ids.end(); ++it){
            double d_xc, d_yc;
            try {
                double area = polyXpoly(xp, yp, Xpoly[it->first], Ypoly[it->first], d_xc, d_yc);
                ar +=area;
                //std::cout << "Cell: ";
                //print_poly_matlab(xp,yp);
                //std::cout << "River: ";
                //print_poly_matlab(Xoutline[it->first], Youtline[it->first]);
                //std::cout << /* "Intersected area: " << area << */ "plot(" << d_xc << ", " << d_yc << ", 'x')" << std::endl;

                if (area < 0.1){
                    int aa = 0;
                    aa++;
                    //std::cout << "The area " << area << " is too small" << std::endl;
                }
                xc.push_back(d_xc);
                yc.push_back(d_yc);
                Q.push_back(area * Q_rate[it->first]);
            } catch (...) {
                std::cout << "Boost failed to find intersection" << std::endl;
            }
        }
    }
    return tf;
}


template <int dim>
void Streams<dim>::add_contributions(TrilinosWrappers::MPI::Vector& system_rhs,
                                     const DoFHandler<dim>& dof_handler,
                                     const FE_Q<dim>& fe,
                                     const AffineConstraints<double>& constraints,
                                     std::vector<int> top_boundary_ids,
                                     int my_rank, int n_proc, MPI_Comm mpi_communicator){
    if (N_seg == 0)
        return;

    std::vector<int> v_nmb;
    if (dim == 3){
        v_nmb.push_back(0);
        v_nmb.push_back(1);
        v_nmb.push_back(3);
        v_nmb.push_back(2);
    }
    else{
        std::cout << "You cannot use Streams in other than 3D problem" << std::endl;
        return;
    }

    Triangulation<dim-1> tria;
    initTria<dim>(tria);
    ine_Tree stream_tree;
    stream_tree.insert(stream_triangles.begin(), stream_triangles.end());



    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    Vector<double>       cell_rhs_streams (dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    double QSTRM = 0;
    double SUMA_AREA = 0;
    //int dbg_coounter = 0;
    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (; cell!=endc; ++cell){
        if (cell->is_locally_owned()){
            for (unsigned int i_face=0; i_face < GeometryInfo<dim>::faces_per_cell; ++i_face){
                if(cell->face(i_face)->at_boundary()){
                    bool isfacetop = false;
                    for (unsigned int jj = 0; jj < top_boundary_ids.size(); ++jj){
                        if (top_boundary_ids[jj] == static_cast<int>(cell->face(i_face)->boundary_id())){
                            isfacetop = true;
                            break;
                        }
                    }

                    if (isfacetop){
                        const MappingQ1<dim-1> mapping;
                        setup_cell(cell->face(i_face), tria);
                        std::vector<double> xc, yc, Qface;
                        std::vector<double> xface(GeometryInfo<dim>::vertices_per_face);
                        std::vector<double> yface(GeometryInfo<dim>::vertices_per_face);
                        // We have to use the correct order of vertices which is not like dealii vertex numbering

                        for (unsigned int jj = 0; jj < GeometryInfo<dim>::vertices_per_face; ++jj){
                            xface[jj] = cell->face(i_face)->vertex(v_nmb[jj])[0];
                            yface[jj] = cell->face(i_face)->vertex(v_nmb[jj])[1];
                        }

                        //print_poly_matlab(xface,yface);
                        //std::cout << dbg_coounter++ << std::endl;
                        double ar;
                        bool tf = get_stream_recharge(xc,yc,Qface,xface,yface, stream_tree, ar);
                        SUMA_AREA += ar;
                        if (tf){ // if this cell intersects a stream
                            cell_rhs_streams = 0;
                            // construct one point quadrature using the centroid of the intersected area
                            for (unsigned int k = 0; k < xc.size(); ++k){
                                Point<dim-1> quad_point;
                                quad_point[0] = xc[k]; quad_point[1] = yc[k];
                                Point<dim-1> unit_mid_point;
                                bool mapping_done = try_mapping<dim-1>(quad_point, unit_mid_point,
                                                                       tria.begin_active(), mapping);
                                if (mapping_done){
                                    double weight = 1.0;
                                    Quadrature<dim-1>stream_quad(unit_mid_point);
                                    FEFaceValues<dim>fe_face_stream_values(fe, stream_quad, update_values | update_quadrature_points);
                                    fe_face_stream_values.reinit(cell, i_face);
                                    for (unsigned int q_point = 0; q_point < stream_quad.size(); ++q_point){
                                        for (unsigned int j = 0; j < dofs_per_cell; ++j){
                                            double Q_temp = weight*Qface[k]*fe_face_stream_values.shape_value(j,q_point);
                                            cell_rhs_streams(j) += Q_temp * stream_multiplier;
                                            QSTRM += Q_temp * stream_multiplier;
                                        }
                                    }
                                }
                            }
                            cell->get_dof_indices (local_dof_indices);
                            constraints.distribute_local_to_global(cell_rhs_streams,
                                                                   local_dof_indices,
                                                                   system_rhs);
                        }
                    }
                }
            }
        }
    }

    sum_scalar<double>(QSTRM, n_proc, mpi_communicator, MPI_DOUBLE);
    sum_scalar<double>(SUMA_AREA, n_proc, mpi_communicator, MPI_DOUBLE);
    if (my_rank == 0)
        std::cout << "\t Q_streams = " << QSTRM << " and AREA = " << SUMA_AREA << std::endl;
}

template <int dim>
void Streams<dim>::flag_cells_for_refinement(parallel::distributed::Triangulation<dim>& triangulation){

    ine_Tree stream_tree;
    stream_tree.insert(stream_triangles.begin(), stream_triangles.end());

    typename parallel::distributed::Triangulation<dim>::active_cell_iterator
    cell = triangulation.begin_active(),
    endc = triangulation.end();
    for (; cell!=endc; ++cell){
        if (cell->is_locally_owned()){
            for (unsigned int i_face=0; i_face < GeometryInfo<dim>::faces_per_cell; ++i_face){
                if(cell->face(i_face)->at_boundary() && cell->face(i_face)->boundary_id() == GeometryInfo<dim>::faces_per_cell-1){
                    std::vector<double> xface(GeometryInfo<dim>::vertices_per_face);
                    std::vector<double> yface(GeometryInfo<dim>::vertices_per_face);
                    for (unsigned int jj = 0; jj < GeometryInfo<dim>::vertices_per_face; ++jj){
                        xface[jj] = cell->face(i_face)->vertex(jj)[0];
                        yface[jj] = cell->face(i_face)->vertex(jj)[1];
                    }

                    std::vector<int> ids;
                    bool tf = find_intersection_in_AABB_TREE(stream_tree,
                                                             stream_triangles,
                                                             xface, yface, ids);
                    if (tf){
                        cell->set_refine_flag ();
                    }
                }
            }
        }
    }
}





#endif // STREAMS_H
