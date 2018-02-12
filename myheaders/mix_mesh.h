#ifndef MIX_MESH_H
#define MIX_MESH_H

#include <deal.II/grid/tria.h>

#include "helper_functions.h"

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

    /*! Return the id of the element that containts the point p.
     * If the returned value is negative then the point is outside of the mesh.
     * If the element is found the variable t has the parametric
     * coordinates of the triangle that contains the point. If the element is
     * quadrilateral the algorithm splits the element into 2 triangles and
     * the quad will indicate which subtriangle contains the point. quad takes either
     * 1 or 2. 1 means that the triangle with indices 0 1 2 containts the point. 2 means
     * that the triangle with indices 1 2 3 containts the point
     */
    int find_elem_id(Point<dim> p, Point<3> &t, int &quad);

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

    //!returns the id of the mesh node which is closer to point p.
    int find_nearest_node(Point<dim> p);

    bool interpolate_on_nodes(Point<dim> p, std::vector<double>& values);

    void reset();


private:
    /*! Checkes if the point is inside the element. When the element is quadrilateral, splits the element
     * into two triangles. If the point is found returns true and the variable t would contain the
     * parametric coordinates and the variable quad will indicate in which of the two triangles of
     * the quadrilateral the point is in.
     */
    bool is_point_in_elem(Point<3> &t, int el_id, Point<dim> p, int &quad);

    /*! Calculates the parametric coordinates of point p with respect to the triangle defined
     * from the coordinates xv and yv. The parametric coordinates are returned in variable
     * t. If any of the values of t are negative or greater to 1 then the point is actually outsilde of
     * the element.
     */
    void parametric_2D_triangle(Point<3> &t, std::vector<double> xv, std::vector<double> yv, Point<dim> p);

    bool in_triangle_exception(Point<dim-1> p, std::vector<double> xv, std::vector<double> yv);
};

template <int dim>
mix_mesh<dim>::mix_mesh(){

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

template <int dim>
bool mix_mesh<dim>::interpolate_on_nodes(Point<dim> p, std::vector<double>& values){
    bool output = false;
    Point<dim> t;
    int quad;
    values.clear();
    int el_id = find_elem_id(p, t, quad);
}

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

template <int dim>
bool mix_mesh<dim>::is_point_in_elem(Point<3> &t, int el_id, Point<dim> p, int &quad){
    bool is_p_in = false;
    if (MSH[el_id].size() == 3){
        std::vector<double> xv(3); std::vector<double> yv(3);
        xv[0] = P[MSH[el_id][0]][0]; yv[0] = P[MSH[el_id][0]][1];
        xv[1] = P[MSH[el_id][1]][0]; yv[1] = P[MSH[el_id][1]][1];
        xv[2] = P[MSH[el_id][2]][0]; yv[2] = P[MSH[el_id][2]][1];
        parametric_2D_triangle(t, xv, yv, p);
        if ((t[0]>=0 && t[0]<=1) && (t[1]>=0 && t[1]<=1) && (t[2]>=0 && t[2]<=1)){
            is_p_in = true;
            quad = 1;
        }
        else{

        }

    }
    else if (MSH[el_id].size() == 4){

    }
    else if (MSH[el_id].size() == 2){
        std::vector<double> xv(2);
        xv[0] = P[MSH[el_id][0]][0];
        xv[1] = P[MSH[el_id][1]][0];
        t[0] = (xv[0]-p[0])/(xv[0]-xv[1]);
        if (t[0]>=0 && t[0]<=1){
            is_p_in = true;
            quad = 1;
        }
        else{
            if (t[0] > 1){
                if (std::abs(xv[1] - p[0]) < 0.001){
                    t[0] = 1;
                    is_p_in = true;
                    quad = 1;
                }
            }
            else if (t[0] < 0){
                t[0] = 0;
                is_p_in = true;
                quad = 1;
            }
        }
    }
    return is_p_in;
}

template <int dim>
void mix_mesh<dim>::parametric_2D_triangle(Point<3> &t, std::vector<double> xv, std::vector<double> yv, Point<dim> p){
    double D = 1/((xv[1]*yv[2]-xv[2]*yv[1])+xv[0]*(yv[1]-yv[2])+yv[0]*(xv[2]-xv[1]));
    std::vector<double> CT(9);

    CT[0]=D*( xv[1]*yv[2] - xv[2]*yv[1] ); CT[1]=D*( yv[1] - yv[2] ); CT[2]=D*( xv[2] - xv[1] );
    CT[3]=D*( xv[2]*yv[0] - xv[0]*yv[2] ); CT[4]=D*( yv[2] - yv[0] ); CT[5]=D*( xv[0] - xv[2] );
    CT[6]=D*( xv[0]*yv[1] - xv[1]*yv[0] ); CT[7]=D*( yv[0] - yv[1] ); CT[8]=D*( xv[1] - xv[0] );
    t[0]=CT[0]*1 + CT[1]*p[0] + CT[2]*p[1];
    t[1]=CT[3]*1 + CT[4]*p[0] + CT[5]*p[1];
    t[2]=CT[6]*1 + CT[7]*p[0] + CT[8]*p[1];

}

template <int dim>
bool mix_mesh<dim>::in_triangle_exception(Point<dim-1> p, std::vector<double> xv, std::vector<double> yv){
    if (dim == 2){
        std::cerr << "it doesnt make sence to use 'in_triangle_exception' in 2D" << std::endl;
        return false;
    }else{

    }

}





#endif // MIX_MESH_H
