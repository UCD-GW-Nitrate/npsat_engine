#ifndef ZINFO_H
#define ZINFO_H

#include <iostream>
#include <ostream>
#include <cmath>

template<class T>
bool sort_Zlist(const T A, T B){ return (A.z < B.z); }


/*!
 * \brief The Zinfo class contains information regarding the z elevation of a
 * mesh node and how this node is  connected in the mesh.
 */
class Zinfo{
public:
    /*!
     * \brief Zinfo construct a new z vertex
     * \param z is the elevation
     * \param dof is the dof
     * \param level is the level of the node
     * \param constr is true if its a hanging node
     */
    Zinfo(double z, int dof, int level, bool constr);

    //! prints all the information of this vertex
    void print_me(std::ostream& stream);

    /*!
     * \brief is_same_z compares the elevation of this point with another elevation
     * \param z is the elevation to compare with
     * \param thres When the two elevations are smaller than the threshold are considered equal
     * \return returns true if the elevations are equal
     */
    bool compare(double z, double thres);

//    //! Copies the zinfo of the vertex to this vertex. The operation does that blindly
//    //! without checking if the input values make sense.
    //void copy(Zinfo zinfo);

    //! change all values to dummy ones (negative) except the elevation and the level
    void reset();

    //! This is the elevation
    double z;

    //! This is the relative position with respect to the nodes above and below
    double rel_pos;

    //! This is the index of the dof number
    int dof;

    //! This is the level of the vertex
    int level;

    //! This is set to 1 if the node is hanging
    int hanging;

    //! This is the dof of the node above this node
    int id_above;

    //! This is the dof of the node below this node
    int id_below;

    //! The dof of the node that serves as top for this node
    int id_top;

    //! The dof of the node that serves as bottom for this node
    int id_bot;

    //! This is a flag that specifies whether the vertex is used or not
    bool used;
    //! A flag that is set to true if this node is connected with the node above
    //! If this is a hanging node then one of the #connected_above or #connected_below
    //! must be false and the other true
    bool connected_above;

    //! A flag that is set to true if this node is connected with the node below
    //! If this is a hanging node then one of the #connected_above or #connected_below
    //! must be false and the other true
    bool connected_below;
};

Zinfo::Zinfo(double z_in, int dof_in, int level_in, bool constr){
    // To construct a new point we need to know the elevation,
    // the dof, the level and whether is a hanging node.
    if (dof_in <0)
        std::cerr << "The dof id cannot be negative" << std::endl;
    if (level_in <0)
        std::cerr << "The level id cannot be negative" << std::endl;

    z = z_in;
    dof = dof_in;
    level = level_in;
    if (constr) hanging = 1;
    else hanging = 0;

    id_above = -9;
    id_below = -9;
    used = true;

    id_top = -9;
    id_bot= -9;
    rel_pos = -9.0;
    connected_above = false;
    connected_below = false;
}

bool Zinfo::compare(double z_in, double thres){
    return (std::abs(z_in - z) < thres);
}

//void Zinfo::copy(Zinfo zinfo){
//    dof         =   zinfo.dof;
//    level       =   zinfo.level;
//    hanging     =   zinfo.hanging;
//    id_above    =   zinfo.id_above;
//    id_below    =   zinfo.id_below;

//    used        =   zinfo.used;

//    id_top      =   zinfo.id_top;
//    id_bot      =   zinfo.id_bot;
//    rel_pos     =   zinfo.rel_pos;
//    z           =   zinfo.z;
//}

void Zinfo::reset(){
    // when we reset a point we change all values to dummy ones except
    // the elevation and the level
    dof = -9;
    hanging = -9;
    id_above = -9;
    id_below = -9;

    used = false;

    id_top = -9;
    id_bot= -9;
    rel_pos = -9.0;
}


#endif // ZINFO_H
