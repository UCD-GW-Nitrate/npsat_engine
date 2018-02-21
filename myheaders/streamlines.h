#ifndef STREAMLINES_H
#define STREAMLINES_H

#include <deal.II/base/point.h>

/*!
* \brief The Streamline class provides elemental functionality to store a streamline
*/
template <int dim>
class Streamline{
public:

    /*!
    * \brief Streamline. The constructor initialize the streamline by setting the Entity id and the
    * streamline id.
    * \param E is the entity id. For example the id of a well or stream
    * \param S is the streamline id
    * \param p is particle id
    */
    Streamline(int E, int S, dealii::Point<dim> p);

    /*!
     * \brief add_point_vel Adds a point in the streamline where the point consist of the position,
     * velocity and the rank of the locally onwed element that containts the position
     * \param p The position of the particle
     * \param v The velcity of the particle at that location
     * \param id_proc The rank of the processor that owns the element that contains the point
     */
    void add_point_vel(dealii::Point<dim> p, dealii::Point<dim> v, int id_proc);

    /*!
    * \brief add_point add the point without the velocity. This is used at the last point of the streamline which is
    * located outside of the domain and there is no velocity
    * \param p The location of the particle
    * \param id_proc The rank of the processor that owns the element that contains the point
    */
    void add_point(dealii::Point<dim> p, int id_proc);

    //! Entity identifier. For example well or stream id
    int E_id;

    //! Streamline identifier. Each entity has its own numbering for its streamlines
    int S_id;

    //! This is the id of the particles
    std::vector<int> p_id;

    //! A vector of particle positions
    std::vector<dealii::Point<dim> > P;

    //! The min XYZ point of the streamline bounding box
    dealii::Point<dim> BBl;

    //!The max XYZ point of the streamline bounding box
    dealii::Point<dim> BBu;

    //! a vector of particle velocities
    std::vector<dealii::Point<dim> > V;

    //! The rank of the current processor
    int proc_id;

    //! Counts the times that the streamline bounding box has not been expanded
    int times_not_expanded;

    bool del;
};

template <int dim>
Streamline<dim>::Streamline(int E, int S, dealii::Point<dim> p){
    E_id = E;
    S_id = S;
    P.push_back(p);
    BBl = p;
    BBu = p;
    times_not_expanded = 0;
    p_id.push_back(0);
    del = false;
}

template <int dim>
void Streamline<dim>::add_point(dealii::Point<dim> p, int id_pr){
    P.push_back(p);
    p_id.push_back(p_id[p_id.size()-1] + 1);
    proc_id = id_pr;
    bool expand = false;
    for (unsigned int i = 0; i < dim; ++i){
        if (p[i] < BBl[i]){
            BBl[i] = p[i];
            expand = true;
        }
        if (p[i] > BBu[i]){
            BBu[i] = p[i];
            expand = true;
        }
    }
    if (expand)
        times_not_expanded = 0;
    else
        times_not_expanded++;
}

template <int dim>
void Streamline<dim>::add_point_vel(dealii::Point<dim> p, dealii::Point<dim> v, int id_proc){
    add_point(p,id_proc);
    V.push_back(v);
}



#endif // STREAMLINES_H
