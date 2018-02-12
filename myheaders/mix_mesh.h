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

    void reset();


private:
    /*! Checkes if the point is inside the element. When the element is quadrilateral, splits the element
     * into two triangles. If the point is found returns true and the variable t would contain the
     * parametric coordinates and the variable quad will indicate in which of the two triangles of
     * the quadrilateral the point is in.
     */
    bool is_point_in_elem(Point<dim> &t, int el_id, Point<dim> p, int &quad)const;
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



#endif // MIX_MESH_H
