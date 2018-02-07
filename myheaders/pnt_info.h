#ifndef PNT_INFO_H
#define PNT_INFO_H

#include <vector>

#include <deal.II/base/point.h>
#include <deal.II/lac/sparsity_tools.h>

#include "zinfo.h"

using namespace dealii;


template <int dim>
class PntsInfo{
public:
    //! The default constructor initialize the structure with dummy values
    //! *However it should never be used*.
    PntsInfo();

    /*!
    * \brief XYZpnts This constructor initializes the structure with actual coordinates.
    * This is the prefered way to initialize this as it avoids empty points in the structure
    *
    * \param p is a deal.ii #Point with dim - 1 space dimension
    * \param zinfo the zinfo is pushed into the z array.
    * Since this is initialization it first clears the Zlist before adding this point
    * This should not be used for update
    */
    PntsInfo(Point<dim-1> p, Zinfo zinfo);

    //! Adds as z node in the existing #PNT of dim-1 point. If the point exists we update the
    //! #Zinfo::dof, #Zinfo::constr, #Zinfo::hanging and #Zinfo::dof_conn.
    void add_Zcoord(Zinfo zinfo, double thres);

    //! This method checks if the input z elevation exists in the list of the z nodes in this point
    std::vector<Zinfo>::iterator check_if_z_exists(Zinfo zinfo, double thres);

    //! Resets all the informaion of the point except the point #PNT coordinates and the z coordinates of the #Zlist
    void reset();

    //! A point with dimensions dim-1 to hold the x and/or y coordinates
    Point<dim-1> PNT;

    //! an array with z coordinate with the same X and y
    std::vector<Zinfo > Zlist;

    //! The top elevation of the mesh at the x-y point
    double T;

    //! The bottom elevation of the mesh at the x-y point
    double B;

    /*! a flag that indicates that at least one of the z points that exist in the #Zlist
     * belong to a cell that has a ghost neighbor cell. If it is true then the entire column
     * of z points is transfered to all processors as it is likely that there would be z points
     * that are missing from the other processors (MAYBE THIS IS NOT USED)
    */
    int have_to_send;

    //! This is a list of processors id that share this node
    std::vector <int> shared_proc;

    //! For each processor in the #shared_proc list keep the key value that this processor can find this point
    //! THis is used to avoid communicate the coordinates back and forth (MAYBE THIS IS NOT USED)
    std::vector<int> key_val_shared_proc;

    //! This is the id that one can find this point in the #Mesh_struct::PointsMap map
    //! Essentially #Mesh_struct::PointsMap.find(find_id) should return an iterator to this point.
    //! (MAYBE THIS IS NOT USED)
    int find_id;

    /*!
     * \brief (MAYBE THIS IS NOT USED)
     * it is possible after a reset that not all the listed nodes have positive dof
     * Actually the vertices with negative id either no longer exist due to coarsening
     * or they now live on a different processor (which is the most common)
     * \return returns the number of z nodes with positive ids
     */
    int number_of_positive_dofs();

    /*! Assign the ids above and below of each node in the Zlist
     * Call this routine only if all the points have positive dofs.
     * It is also assumed that each processor has all the points it needs from the other processors
     * Two z nodes are connected if there is a cell in the triangulation that has a face with those two nodes
     * First we loop through the z nodes that exists under this xy point
     * The first point with index 0 is the lowest in this column. The lowest point is always connected with the node above.
     * As we loop through the points we set the dofs adove and below that point. In addition we set true/false the flag
     * that indicate the connections with the points above and below.
     * Next we loop two more times through the Z coordinates.
     * First we start from the top and set the #Zinfo::dof_top and #Zinfo::id_top,
     * and then we loop from the bottom to set the #Zinfo::dof_bot and #Zinfo::id_bot.
     * The index 0 node has always itself its #Zinfo::id_bot, while the last index has itsef as its Zinfo::id_top.
     * Finaly we loop a once more to calculate the relative position of each node with respect to its dof_top and dof_bot.
    */
    void set_ids_above_below();

    //! THis is another flag that MAYBE NOT NEEDED anymore.
    bool isEmpty;
};

template <int dim>
PntsInfo<dim>::PntsInfo(){
    PNT = Point<dim-1>();
    for (unsigned int d = 0; d < dim-1; ++d)
        PNT[d] = -9999.0;
    T = -9999.0;
    B = -9999.0;
    Zlist.clear();
    have_to_send = 0;
    shared_proc.clear();
    isEmpty = true;
}

template <int dim>
PntsInfo<dim>::PntsInfo(Point<dim-1> p, Zinfo zinfo){
    PNT = p;
    Zlist.clear();
    Zlist.push_back(zinfo);
    T = -9999.0;
    B = -9999.0;
    have_to_send = 0;
    shared_proc.clear();
    isEmpty = false;
}

template <int dim>
void PntsInfo<dim>::add_Zcoord(Zinfo zinfo, double thres){
    //if (zinfo.dof < 0){
    //    std::cerr << "You attempt to add a vertex with negative dof" << std::endl;
    //}
    std::vector<Zinfo >::iterator it = check_if_z_exists(zinfo, thres);
    if (it != Zlist.end()){
        // SHOULD WE UPDATE ALL THE INFO OR SOME OF IT OR NONE?????????
        it->update_main_info(zinfo);
    }
    else{
        Zlist.push_back(zinfo);
        std::sort(Zlist.begin(), Zlist.end(), sort_Zlist<Zinfo>);
    }
    isEmpty = false;
}

template<int dim>
std::vector<Zinfo>::iterator PntsInfo<dim>::check_if_z_exists(Zinfo zinfo, double thres){
    typename std::vector<Zinfo>::iterator it;
    for (it = Zlist.begin(); it != Zlist.end(); ++it){
        if (abs(it->z - zinfo.z) < thres){
            return it;
        }

        //std::cout << "Compare " << it->get_z() << " with " << zinfo.get_z() << std::endl;
        // I dont understand the logic for the following:
        if (it->z - zinfo.z > 2*thres)
            break;
    }
    return Zlist.end();
}

template <int dim>
void PntsInfo<dim>::reset(){
    have_to_send = 0;
    T = -9999.0;
    B = -9999.0;
    shared_proc.clear();
    typename std::vector<Zinfo>::iterator it = Zlist.begin();
    for (; it != Zlist.end(); ++it)
        it->reset();
}

template <int dim>
int PntsInfo<dim>::number_of_positive_dofs(){
    int N_dofs = 0;
    typename std::vector<Zinfo>::iterator it = Zlist.begin();
    for (; it != Zlist.end(); ++it){
        if (it->dof >= 0)
            N_dofs++;
    }
    return N_dofs;

}

template <int dim>
void PntsInfo<dim>::set_ids_above_below(){
    /*    a ---------       b   ---------
     *      |   |   |           |       |
     *      |-------|           |       |
     *      |   |   |           |       |
     *      ---------           ---------
     *      |  [0]  |           |   |   |
     *      |       |           ---------
     *      |       |           |   |   |
     *      ---------           ---------
     *                             [0]
     */

    for (unsigned int i = 0; i < Zlist.size(); ++i){
        if (i == 0){//================================================
            // If this is the first node from the bottom
            Zlist[i].dof_above = Zlist[i+1].dof;
            Zlist[i].connected_above = Zlist[i].connected_with(Zlist[i].dof_above);
            Zlist[i].dof_bot = Zlist[i].dof;
            Zlist[i].id_bot = i;

        }else if(i==Zlist.size()-1){//======================================
            //this is the top node on this list
            Zlist[i].dof_below = Zlist[i-1].dof;
            Zlist[i].connected_below = Zlist[i].connected_with(Zlist[i].dof_below);
            Zlist[i].dof_top = Zlist[i].dof;
            Zlist[i].id_top = i;

        }else{
            Zlist[i].dof_above = Zlist[i+1].dof;
            Zlist[i].dof_below = Zlist[i-1].dof;
            Zlist[i].connected_above = Zlist[i].connected_with(Zlist[i].dof_above);
            Zlist[i].connected_below = Zlist[i].connected_with(Zlist[i].dof_below);
        }
    }
    //=======================================
    int cur_dof_bot =Zlist[0].dof;
    int cur_id_bot = 0;
    for (unsigned int i = 1; i < Zlist.size(); ++i){
        if (Zlist[i].connected_below){
            Zlist[i].dof_bot = cur_dof_bot;
            Zlist[i].id_bot = cur_id_bot;
        }else{
            Zlist[i].dof_bot = Zlist[i].dof;
            Zlist[i].id_bot = i;
            cur_dof_bot = Zlist[i].dof;
            cur_id_bot = static_cast<int>(i);
        }
    }

    int cur_dof_top =Zlist[Zlist.size()-1].dof;
    int cur_id_top = Zlist.size()-1;
    for (int i = Zlist.size() - 2; i >=0; --i){// When we loop with --i unsigned int causes errors if i gets below 0
        //std::cout << i << std::endl;
        if (Zlist[i].connected_above){
            Zlist[i].dof_top = cur_dof_top;
            Zlist[i].id_top = cur_id_top;
        }
        else{
            Zlist[i].dof_top = Zlist[i].dof;
            Zlist[i].id_top = i;
            cur_dof_top = Zlist[i].dof;
            cur_id_top = i;
        }
    }
    // Set relative position
    for (unsigned int i = 0; i < Zlist.size(); ++i){
        Zlist[i].rel_pos = (Zlist[i].z - Zlist[Zlist[i].id_bot].z)/(Zlist[Zlist[i].id_top].z - Zlist[Zlist[i].id_bot].z);
    }
}




#endif // PNT_INFO_H
