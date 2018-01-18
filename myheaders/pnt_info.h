#ifndef PNT_INFO_H
#define PNT_INFO_H

#include <vector>

#include <deal.II/base/point.h>

#include "zinfo.h"

using namespace dealii;

template <int dim>
class PntsInfo{
public:
    //! The default constructor initialize the structure with dummy values
    //! This should never be used
    PntsInfo();

    /*!
    * \brief XYZpnts This constructor initializes the structure with actual coordinates.
    * This is the prefered way to initialize this as it avoids empty points in the structure
    *
    * \param p is a deal.ii point with dim - 1 space dimension
    * \param zinfo the zinfo is pushed on the z array
    * Since this is initialization it first clears the Zlist before adding this point
    * This should not used for update
    */
    PntsInfo(Point<dim-1> p, Zinfo zinfo);

    //! Adds as z node in the existing p<dim-1> point. If the point exists we update the
    //! #Zinfo::dof, #Zinfo::level and #Zinfo::constr
    void add_Zcoord(Zinfo zinfo, double thres);

    //! This method checks if the input z elevation exists in the list of the z nodes in this point
    std::vector<Zinfo>::iterator check_if_z_exists(Zinfo zinfo, double thres);

    //! Resets all the informaion of the point except the point #PNT coordinates and the z coordinates of the #Zlist
    void reset();

    //! A point with dimensions dim-1 to hold the x and/or y coordinates
    Point<dim-1> PNT;

    //! an array with z coordinate with the same X and y
    std::vector<Zinfo> Zlist;

    //! The top elevation of the aquifer at the x-y point
    double T;

    //! The bottom elevation of the aquifer at the x-y point
    double B;

    /*! a flag that indicates that at least one of the z points that exist in the #Zlist
     * belong to a cell that has a ghost neighbor cell. If it is true then the entire column
     * of z points is transfered to all processors as it is likely that there would be z points
     * that are missing from the other processors
    */
    int have_to_send;

    /*!
     * \brief it is possible after a reset that not all the listed nodes have positive dof
     * Actually the vertices with negative id either no longer exist due to coarsening
     * or they now live on a different processor (which is the most common)
     * \return returns the number of z nodes with positive ids
     */
    int number_of_positive_dofs();

    /*! Assign the ids above and below of each node in the Zlist
     * Call this routine only if all the points have positive dofs
     * Maybe you need to delete the unused before calling this
    */
    void set_ids_above_below();
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
}

template <int dim>
PntsInfo<dim>::PntsInfo(Point<dim-1> p, Zinfo zinfo){
    PNT = p;
    Zlist.clear();
    Zlist.push_back(zinfo);
    T = -9999.0;
    B = -9999.0;
    have_to_send = 0;
}

template <int dim>
void PntsInfo<dim>::add_Zcoord(Zinfo zinfo, double thres){
    if (zinfo.dof < 0){
        std::cerr << "You attempt to add a vertex with negative dof" << std::endl;
    }
    std::vector<Zinfo>::iterator it = check_if_z_exists(zinfo, thres);
    if (it != Zlist.end()){
        // SHOULD WE UPDATE ALL THE INFO OR SOME OF IT OR NONE?????????
        it->dof = zinfo.dof;
        it->level = zinfo.level;
        it->hanging = zinfo.hanging;
    }
    else{
        Zlist.push_back(zinfo);
        std::sort(Zlist.begin(), Zlist.end(), sort_Zlist<Zinfo>);
    }
}

template<int dim>
std::vector<Zinfo>::iterator PntsInfo<dim>::check_if_z_exists(Zinfo zinfo, double thres){
    std::vector<Zinfo>::iterator it;
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
    std::vector<Zinfo>::iterator it = Zlist.begin();
    for (; it != Zlist.end(); ++it)
        it->reset();
}

template <int dim>
int PntsInfo<dim>::number_of_positive_dofs(){
    int N_dofs = 0;
    std::vector<Zinfo>::iterator it = Zlist.begin();
    for (; it != Zlist.end(); ++it){
        if (it->dof >= 0)
            N_dofs++;
    }
    return N_dofs;

}

template <int dim>
void PntsInfo<dim>::set_ids_above_below(){
    bool hanging_pair = false;
    for (unsigned int i = 0; i < Zlist.size(); ++i){
        if (i == 0){//================================================
            // If the is the first node from the bottom
            Zlist[i].id_above = Zlist[i+1].dof;
            Zlist[i].connected_above = true;
            Zlist[i].id_bot = Zlist[i].dof;

            if (Zlist[i].hanging == 0){
                B = Zlist[i].z;// MAYBE THIS HAS TO BE SET LATER. IF YES CHANGE THE CONDITION
            }else{
                hanging_pair = !hanging_pair;
            }

        }else if(i==Zlist.size()-1){//======================================
            //this is the top node on this list
            Zlist[i].id_below = Zlist[i-1].dof;
            Zlist[i].connected_below = true;
            Zlist[i].id_top = Zlist[i].dof;

            if (Zlist[i].hanging == 0){
                T = Zlist[i].z;// MAYBE THIS HAS TO BE SET LATER. IF YES CHANGE THE CONDITION
            }else{
                hanging_pair = !hanging_pair;
            }
        }else{
            Zlist[i].id_above = Zlist[i+1].dof;
            Zlist[i].id_below = Zlist[i-1].dof;
            if (Zlist[i].hanging == 0){
                Zlist[i].connected_above = true;
                Zlist[i].connected_below = true;
            }else{
                hanging_pair = !hanging_pair;
                if (hanging_pair)
                    Zlist[i].connected_above = true;
                else
                    Zlist[i].connected_below = true;
            }
        }

    }

    for (unsigned int i = 0; i < Zlist.size(); ++i){

    }


}




#endif // PNT_INFO_H
