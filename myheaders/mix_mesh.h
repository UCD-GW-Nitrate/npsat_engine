#ifndef MIX_MESH_H
#define MIX_MESH_H

#include <deal.II/grid/tria.h>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>

typedef boost::geometry::model::d2::point_xy<double> b_point;
typedef boost::geometry::model::polygon<b_point> b_polygon;

#include "helper_functions.h"
#include "cgal_functions.h"

using namespace dealii;

//! This is a helper structure for sorting points.
struct bary_dst{
    double dst;
    int id;
};

//! The is the sorting criterion.
bool sort_distance (bary_dst a, bary_dst b) { return (a.dst< b.dst); }

template <int dim>
class mix_mesh{
public:
    mix_mesh();

//    /*! Return the id of the element that containts the point p.
//     * If the returned value is negative then the point is outside of the mesh.
//     * If the element is found the variable t has the parametric
//     * coordinates of the triangle that contains the point. If the element is
//     * quadrilateral the algorithm splits the element into 2 triangles and
//     * the quad will indicate which subtriangle contains the point. quad takes either
//     * 1 or 2. 1 means that the triangle with indices 0 1 2 containts the point. 2 means
//     * that the triangle with indices 1 2 3 containts the point
//     */
    //int find_elem_id(Point<dim> p, Point<3> &t, int &quad);

    //! A vector that contains the vertices of the mesh
    std::vector<Point<dim> >		P;

    //! A vector with the indices of the elements
    std::vector<std::vector<int> > 	MSH;

    //! A vector with the barycenters of the elements
    std::vector<Point<dim> > 		bary;

    //! The number of points
    unsigned int Np;

    //! The number of elements
    unsigned int Nel;

    //! A vector with properties defined on nodes
    std::vector<std::vector<double> > data_point;

    //! A vector with properties defined on elements
    std::vector<std::vector<double> > data_elem;

    bool add_element(std::vector<int> element_ids);

    //! add a point in the mesh without checking if hte point already exists.
    //! The check should be done before this. This routine, in addition to pushing
    //! back the point into vector #P recalculates the bounding box
    void add_point(Point<dim> p);

    //!returns the id of the mesh node which is closer to point p.
    //int find_nearest_node(Point<dim> p);

    bool interpolate_on_nodes(Point<dim> p, std::vector<double>& values);

    void reset();


private:
    bool is_point_inside(Point<dim> p, int el_id);
    Point<dim> MIN;
    Point<dim> MAX;
};

template <int dim>
mix_mesh<dim>::mix_mesh(){
    for (unsigned int i = 0; i < dim; ++i){
        MIN[i] =  999999999999;
        MAX[i] = -999999999999;
    }
}

template <int dim>
void mix_mesh<dim>::reset(){
    P.clear();
    MSH.clear();
    data_elem.clear();
    data_point.clear();
    bary.clear();
}

template <int dim>
bool mix_mesh<dim>::add_element(std::vector<int> element_ids){
    for (unsigned int i = 0; i < element_ids.size(); ++i){
        if (element_ids[i] < 0 || element_ids[i] > P.size()-1){
            std::cerr << "The element index " << element_ids[i] << " does not exists in the P vector." << std::endl;
            return false;

        }
        for (unsigned int j = i + 1; j < element_ids.size(); ++j){
            if (element_ids[i] == element_ids[j]){
                std::cerr << "This element appears to have the same id in two corner" << std::endl;
                return false;
            }
        }
    }

    // if all the ids are within a valid range add the element and calculate the barycenter
    MSH.push_back(element_ids);
    // calculate the element barycenter
    Point<dim> barycnt;
    for (unsigned int j = 0; j < element_ids.size(); ++j){
        for (unsigned int k = 0; k < dim; ++k)
            barycnt[k] += P[element_ids[j]][k];
    }

    for (unsigned int k = 0; k < dim; ++k)
        barycnt[k] = barycnt[k]/element_ids.size();

    bary.push_back(barycnt);
    return true;
}

/*
template <int dim>
int mix_mesh<dim>::find_nearest_node(Point<dim> p){
    int id_node=-1;
    double dst = 10000000000;
    double temp;
    for (unsigned int i = 0; i < P.size(); ++i){
        temp = p.distance(P[i]);
        if (temp < dst){
            id_node = i;
            dst = temp;
        }
    }
    return id_node;
}
*/


template <int dim>
bool mix_mesh<dim>::interpolate_on_nodes(Point<dim> p, std::vector<double>& values){

    for (unsigned int i = 0; i < dim; ++i){
        if (p[i] < MIN[i] || p[i] > MAX[i]){
            return false;
        }
    }

    std::vector<bary_dst> dst(bary.size());
    for (unsigned int i = 0; i < bary.size(); ++i){
        dst[i].dst = p.distance(bary[i]);
        dst[i].id = static_cast<int>(i);
    }

    // next sort them
    int N_sort = 10;
    if (bary.size() < 12){
        std::sort(dst.begin(), dst.end(), sort_distance);
        N_sort = dst.size();
    }
    else{
        std::partial_sort (dst.begin(), dst.begin() + N_sort, dst.end(), sort_distance);
    }
    for (int i = 0; i < N_sort; ++i){
        bool is_in = is_point_inside(p, dst[i].id);
        if (is_in){
            if (dim == 1){
                double x1 = P[MSH[dst[i].id][0]][0];
                double x2 = P[MSH[dst[i].id][1]][0];
                double t = (x1-p[0])/(x1-x2);
                values.clear();
                for (unsigned int j = 0; j < data_point[0].size(); ++j){
                    double v = 0;
                    v = (1 - t) * data_point[MSH[dst[i].id][0]][j] + t * data_point[MSH[dst[i].id][1]][j];
                    values.push_back(v);
                }
            }
            else if (dim == 2){
                // Calculate the barycentric coordinates
                std::vector<double> b_coords, xv, yv;
                for (unsigned int j = 0; j < MSH[dst[i].id].size(); ++j){
                    xv.push_back(P[MSH[dst[i].id][j]][0]);
                    yv.push_back(P[MSH[dst[i].id][j]][1]);
                }
                b_coords = barycentricCoords<dim>(xv, yv, p);

                values.clear();
                for (unsigned int j = 0; j < data_point[0].size(); ++j){
                    double v = 0;
                    for (unsigned int k = 0; k < b_coords.size(); ++k){
                        v += b_coords[k]*data_point[MSH[dst[i].id][k]][j];
                    }
                    values.push_back(v);
                }
            }
            return true;
        }
    }
    return false;
}

/*
template <int dim>
int mix_mesh<dim>::find_elem_id(Point<dim> p, Point<3> &t, int &quad){
    int el_id = -1;
    std::vector<bary_dst> dst(bary.size());
    for (unsigned int i = 0; i < bary.size(); ++i){
        dst[i].dst = p.distance(bary[i]);
        dst[i].id = i;
    }
    // next sort them
    int N_sort = 10;
    if (bary.size() < 12){
        std::sort(dst.begin(), dst.end(), sort_distance);
        N_sort = dst.size();
    }
    else{
        std::partial_sort (dst.begin(), dst.begin() + N_sort, dst.end(), sort_distance);
    }

    bool is_p_in;
    for (int i = 0; i < N_sort; ++i){
        is_p_in = is_point_in_elem(t, dst[i].id, p, quad);
        if (is_p_in){
            el_id = dst[i].id;
            break;
        }
    }
    return el_id;
}
*/

template <int dim>
void mix_mesh<dim>::add_point(Point<dim> p){
    P.push_back(p);
    for (unsigned int i = 0; i < dim; ++i){
        if (p[i] < MIN[i])
            MIN[i] = p[i];
        if (p[i] > MAX[i])
            MAX[i] = p[i];
    }
}

template <int dim>
bool mix_mesh<dim>::is_point_inside(Point<dim> p, int el_id){

    if (MSH[el_id].size() > 2 && dim == 3){
        b_polygon b_poly;
        std::vector<b_point> pnts;
        std::vector<double> xv, yv;
        for (unsigned int i = 0; i < MSH[el_id].size(); ++i){
            pnts.push_back(b_point(P[MSH[el_id][i]][0], P[MSH[el_id][i]][1]));
        }
        boost::geometry::assign_points(b_poly, pnts);
        boost::geometry::correct(b_poly);

        return boost::geometry::covered_by(b_point(p[0],p[1]),b_poly);
    }
    else if (MSH[el_id].size() == 2){
        double x1 = P[MSH[el_id][0]][0];
        double x2 = P[MSH[el_id][1]][0];
        double t = (x1-p[0])/(x1-x2);
        if (t>=0 && t<=1)
            return true;
        else{
            if (t > 1){
                if (std::abs(x2 - p[0]) < 0.001){
                    return true;
                }
            }
            else if (t < 0){
                if (std::abs(x1 - p[0]) < 0.001){
                    return true;
                }
            }
        }
    }
    return false;
}



#endif // MIX_MESH_H
