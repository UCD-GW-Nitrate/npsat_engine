#ifndef ZINFO_H
#define ZINFO_H

#include <iostream>
#include <ostream>
#include <cmath>
#include <map>
#include <vector>
#include <algorithm>


template<class T>
bool sort_Zlist(const T A, T B){ return (A.z < B.z); }


/*!
 * \brief The DOFZ struct is a helper struct to hold some information about a given node.
 * It is used in the #Zinfo class to hold the information about the top and bottom nodes of
 * a given node
 */
struct DOFZ{
    //! The degree of freedom
    int dof;
    //! The Z coordinate
    double z;
    //! the id in the Zlist that this node can be found (NOT SURE IF I"LL USE THIS).
    int id;
    //! The processor id that this node is locally owned. If the id is negative then
    //! the node with #dof lives in another processor but we dont know in which
    //! processor it lives. Therefore if the #proc is negative #z, #id are also negative
    //! and the #isSet is false.
    int proc;
    //! isSet gets true during the #Mesh_struct<dim>::updateMeshElevation method.
    bool isSet;

    void dummy_values(){
        dof = -9;
        z = -9999;
        id = -9;
        proc = -9;
        isSet = false;
    }
};



/*!
 * \brief The Zinfo class contains information regarding the elevation of a
 * mesh node and information about how this node is connected in the mesh.
 * In 2D the elevation is the Y coordinate, while in 3D the elevation is the Z coordinate.
 */

class Zinfo{
public:
    /*!
     * \brief Zinfo construct a new z node.
     * The dof should never be negative, yet the contructor allows to create Zinfo points with negative dofs.
     * Therefore if the constructor should create non-negative dofs a check is needed outise of the constructor.
     * (During development there were cases where dummy points with negative dofs were needed.
     * However this maybe removed in the future).
     * \param z is the elevation. Y coord in 2D and Z coord in 3D
     * \param dof is the degree of freedom as it is generated from the dof_handler
     * \param cnstr_nodes is a vector with the dofs of the nodes that constrain this point.
     * If the point is not constraint the vector should be empty. The deal function
     * #ConstraintMatrix::resolve_indices returns a vector of the dofs that constraint a given point
     * in the input vector. The returned vector containts the dof of the point in question as well as the dofs
     * of the points that constraint this point. The constructor takes care of this by removing the dof of the node
     * if this is found in the #cnstr_nodes vector.
     * \param istop is a flag that indicates whether this node touches the top boundary of the domain.
     * If the # CellAccessor< dim, spacedim >::neighbor_index of the 3rd face in 2D or the 5th face int 3D returns negative
     * the vertices of that cell are flaged as top.
     * \param isbot is a flag that indicates whether this node touches the bottom boundary of the domain.
     * If the # CellAccessor< dim, spacedim >::neighbor_index of the 2nd face in 2D or the 4th face int 3D returns negative
     * the vertices of that cell are flaged as top.
     * \param dof_conn is a vector that containts the nodes that are connected with this node in the vertical direction only.
     */
    Zinfo(double z, int dof, std::vector<int> cnstr_nodes, int istop, int isbot, std::vector<int> dof_conn);

    //! This is a vector that holds the dofs of the triangulation points of the points that this point is connected with.
    std::vector<int> dof_conn;

    //! This is a vector that holds the dofs of the point that constrain this point
    std::vector<int> cnstr_nds;

    //! prints all the information of this vertex
    void print_me(std::ostream& stream);

    /*!
     * \brief (NOT USED)is_same_z compares the elevation of this point with another elevation
     * \param z is the elevation to compare with
     * \param thres When the two elevations are smaller than the threshold are considered equal
     * \return returns true if the elevations are equal
     */
    //bool compare(double z, double thres);

    //! Attempts to add connection to this point. If the connection already exists
    //! nothing is added
    void Add_connections(std::vector<int> conn);

    //! You should call this only after this point has at least z info assigned
    //! from a previous iteration.
    //! A typical case would be after resetting the mesh structure
    void update_main_info(Zinfo newZ);

    //! This method returns true if the point is connected to this one
    //! Essentially is considered connected if the point in question can be found
    //! in the #dof_conn map of connected nodes.
    //! In practice it appears that a particular node maybe connected with one node
    //! in one cell and not connected in an another cell if the cells that share the node
    //! have different level. However we really need this information only for the hanging
    //! nodes where that never happens,
    bool connected_with(int dof_in);

    //! Returns true if the dof in question belongs to the list of the dofs that constain this one
    bool is_constrainted_by(int dof_in);

    //! change all values to dummy ones (negative) except the elevation
    void reset();

    //! Attemps to set the dofs of the input vector as constraints for this point.
    //! If the dof exists nothing is added. If the input dof is the same as the #dof
    //! nothing is added.
    void add_constraint_nodes(std::vector<int> cnst);

    //! This is the elevation
    double z;

    //! This is the relative position with respect to the nodes above and below
    double rel_pos;

    //! This is the index of the dof number
    int dof;

    //! This is set to 1 if the node is hanging. This should never be set by the user
    //! but rather let the #add_constraint_nodes method to assign it appropriately
    int hanging;

    //! This is the dof of the node above this node. If its -9 then there is not node above.
    //! If the node is hanging and not connected with the node above, the #dof_above still can
    //! be set.
    int dof_above;

    //! This is the dof of the node below this node. If its -9 then there is not node below.
    //! If the node is hanging and not connected with the node below, the #dof_below still can
    //! be set.
    int dof_below;

    //! The dof of the node that serves as top for this node
    DOFZ Top;

    //! The dof of the node that serves as bottom for this node
    DOFZ Bot;

    //! A boolean flag that is true if the node lays on the top surface of the mesh
    int isTop;

    //! A boolean flag that is true if the node lays on the bottom of the mesh
    int isBot;

    //! A flag that is set to true if this node is connected with the node above
    //! If this is a hanging node then one of the #connected_above or #connected_below
    //! must be false and the other true
    bool connected_above;


    //! A flag that is set to true if this node is connected with the node below
    //! If this is a hanging node then one of the #connected_above or #connected_below
    //! must be false and the other true
    bool connected_below;

    //! This is a flag that is true if the elevetion of this node has been updated at a certain iteration
    //! If it is true you can use this node to calculate the elevation of another node that depends on this one.
    bool isZset;

    //! If the node is owned by the processor who holds it, it is local
    bool is_local;

};

Zinfo::Zinfo(double z_in, int dof_in, std::vector<int> cnstr_nodes, int istop, int isbot,  std::vector<int> conn){
    // To construct a new point we need to know the elevation,
    // the dof, the level and whether is a hanging node.
    // Although the ids should not be negative we allow to create Zinfo points with negative ids
    // However threr sould always be a check before calling this function
    //if (dof_in <0)
    //    std::cerr << "The dof id cannot be negative" << std::endl;
    //if (level_in <0)
    //    std::cerr << "The level id cannot be negative" << std::endl;

    z = z_in;
    dof = dof_in;
    add_constraint_nodes(cnstr_nodes);


    isTop = istop;
    isBot = isbot;

    dof_above = -9;
    dof_below = -9;

    Top.dummy_values();
    Bot.dummy_values();

    rel_pos = -9.0;
    connected_above = false;
    connected_below = false;
    isZset = false;
    is_local = false;

    Add_connections(conn);
}

void Zinfo::update_main_info(Zinfo newZ){
    if (newZ.dof <0)
        std::cerr << "The new dof id cannot be negative" << std::endl;
    if (dof >= 0){
        if (dof != newZ.dof){
            std::cerr << " You attempt to update on a point that has already dof\n"
                      <<  "However the updated dof is different from the current dof" << std::endl;
        }
    }
    dof = newZ.dof;
    hanging = newZ.hanging;
    Add_connections(newZ.dof_conn);
    add_constraint_nodes(newZ.cnstr_nds);
}

void Zinfo::Add_connections(std::vector<int> conn){
    std::vector<int>::iterator it;
    for (it = conn.begin(); it != conn.end(); ++it){
        if (std::find(dof_conn.begin(),dof_conn.end(), *it) == dof_conn.end()){
            dof_conn.push_back(*it);
        }
    }
}


//bool Zinfo::compare(double z_in, double thres){
//    return (std::abs(z_in - z) < thres);
//}

bool Zinfo::connected_with(int dof_in){
    return std::find(dof_conn.begin(), dof_conn.end(), dof_in) != dof_conn.end();
}

void Zinfo::reset(){
    // when we reset a point we change all values to dummy ones except
    // the elevation and the level
    dof = -9;
    hanging = -9;
    dof_above = -9;
    dof_below = -9;

    isZset = false;

    Top.dummy_values();
    Bot.dummy_values();
    rel_pos = -9.0;
    connected_above = false;
    connected_below = false;
    dof_conn.clear();
    cnstr_nds.clear();
}

void Zinfo::add_constraint_nodes(std::vector<int> cnstr_nodes){

    for (unsigned int i = 0; i < cnstr_nodes.size(); ++i){
        if (cnstr_nodes[i] == dof)
            continue;
        bool addthis = true;
        for (unsigned int j = 0; j < cnstr_nds.size(); ++j){
            if (cnstr_nodes[i] == cnstr_nds[j]){
                addthis = false;
                break;
            }
        }
        if (addthis){
            cnstr_nds.push_back(cnstr_nodes[i]);
        }
    }
    hanging = static_cast<int>(cnstr_nds.size() > 0);
}

bool Zinfo::is_constrainted_by(int dof_in){
    if (cnstr_nds.size() == 0)
        return false;
    else{
        for (unsigned int i = 0; i < cnstr_nds.size(); ++i){
            if (dof_in == cnstr_nds[i]){
                return true;
            }
        }
        return false;
    }
}

#endif // ZINFO_H
