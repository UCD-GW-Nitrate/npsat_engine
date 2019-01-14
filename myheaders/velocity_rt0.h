#ifndef VELOCITY_RT0_H
#define VELOCITY_RT0_H

#include <deal.II/dofs/dof_handler.h>
#include <eigen3/Eigen/Dense>

#include "helper_functions.h"

using namespace dealii;

template <int dim>
class velocityCellRT{
public:
    CellId  cellid;
    int     icellid;
    std::vector<double> velocities;

    /*!
    * \brief calcVelocity calculates the velocity for this cell
    * \param p_unit is a point in the [0 1]
    * \return True is the velocity computation was successfull
    */
    bool calcVelocity(Point<dim> p_unit, Point<dim>& Vel);

};


template <int dim>
bool velocityCellRT<dim>::calcVelocity(Point<dim> p_unit, Point<dim> &Vel){
    if (dim == 2){
        double N0x = 1 - p_unit[0];
        double N0y = 0;
        double N1x = p_unit[0];
        double N1y = 0;
        double N2x = 0;
        double N2y = 1 - p_unit[1];
        double N3x = 0;
        double N3y = p_unit[1];

        Vel[0] = velocities[0]*N0x + velocities[1]*N1x + velocities[2]*N2x + velocities[3]*N3x;
        Vel[1] = velocities[0]*N0y + velocities[1]*N1y + velocities[2]*N2y + velocities[3]*N3y;
        return true;
    }
    else if (dim == 3){
        double N0x = 1 - p_unit[0];
        double N0y = 0;
        double N0z = 0;
        double N1x = p_unit[0];
        double N1y = 0;
        double N1z = 0;
        double N2x = 0;
        double N2y = 1 - p_unit[1];
        double N2z = 0;
        double N3x = 0;
        double N3y = p_unit[1];
        double N3z = 0;
        double N4x = 0;
        double N4y = 0;
        double N4z = 1 - p_unit[2];
        double N5x = 0;
        double N5y = 0;
        double N5z = p_unit[2];

        Vel[0] = velocities[0]*N0x + velocities[1]*N1x + velocities[2]*N2x + velocities[3]*N3x + velocities[4]*N4x + velocities[5]*N5x;
        Vel[1] = velocities[0]*N0y + velocities[1]*N1y + velocities[2]*N2y + velocities[3]*N3y + velocities[4]*N4y + velocities[5]*N5y;
        Vel[2] = velocities[0]*N0z + velocities[1]*N1z + velocities[2]*N2z + velocities[3]*N3z + velocities[4]*N4z + velocities[5]*N5z;
        return true;
    }
    else{
        std::cerr << "Wrong dimension in velocityCellRT<dim>::calcVelocity method" << std::endl;
        return false;
    }
}

template <int dim>
class ControlVolume{
public:
    ControlVolume(unsigned int dof_in);
    //! You can initialize a control volume with a dof and a
    ControlVolume(unsigned int dof_in, CellId cellid_in, int vertex_index);
    void addCell_vertex(CellId newcell, int vertex_index, int level);
    void addCell_face(CellId newcell, int face_index, int level);
    void addCell_edge(CellId newcell, int edge_index, int level);
    int Ncells();
    CellId getCellID(int ii);
    int getLevel(int ii);
    int maxLevel();
    void setQsource(double Q);

protected:
    //! The dof of the control volume
    unsigned int dof;

    double Qsource;

    //! A list of cell ids connected with the #dof
    std::vector<CellId> cells;

    std::vector<int> levels;

    // This is the vertex number of the dof as seen from each cell of the #cells vector.
    // if the vertex number is negative then this vertex is not part of the cell.
    // The dof should be hanging node, therefore we should look into the edge list to see
    std::vector<int> vertex_index_in_cell;

    // If the dof is hanging then there should be at least one cell that its edge touches the dof.
    // if the dof is part of the cell then it should get a negative id
    std::vector<int> face_index_in_cell;

    // A list of edges or faces that are connected with the id
     std::vector<int> edge_index_in_cell;

     bool checkIfcellexists(CellId cellid_in);
};

template <int dim>
ControlVolume<dim>::ControlVolume(unsigned int dof_in){
    dof = dof_in;
    Qsource = 0;
    if (dim == 2){
        cells.resize(4);
        levels.resize(4);
    }
    else if (dim ==3){
        cells.resize(8);
        levels.resize(8);
    }
    for (unsigned int i = 0; i < levels.size(); ++i)
        levels[i] = -9;
}

template <int dim>
void ControlVolume<dim>::setQsource(double Q){
    Qsource = Q;
}



template <int dim>
bool ControlVolume<dim>::checkIfcellexists(CellId cellid){
    for (unsigned i = 0; i < cells.size(); i++){
        if (cells[i] == cellid)
            return true;
    }
    return false;
}

template <int dim>
void ControlVolume<dim>::addCell_vertex(CellId newcell, int vertex_index, int level){
    int index = -9;
    if (!checkIfcellexists(newcell)){
        if (dim == 2){
            if (vertex_index == 0)
                index = 3;
            else if (vertex_index == 1)
                index = 2;
            else if (vertex_index == 2)
                index = 1;
            else if (vertex_index == 3)
                index = 0;
        }
        else if (dim == 3){
            if (vertex_index == 0)
                index = 7;
            else if (vertex_index = 1)
                index = 6;
            else if (vertex_index == 2)
                index = 5;
            else if (vertex_index == 3)
                index = 4;
            else if (vertex_index == 4)
                index = 3;
            else if (vertex_index == 5)
                index = 2;
            else if (vertex_index == 6)
                index = 1;
            else if (vertex_index == 7)
                index = 0;
        }
        cells[index] = newcell;
        levels[index] = level;
    }
}

template <int dim>
void ControlVolume<dim>::addCell_face(CellId newcell, int face_index, int level){
    if (!checkIfcellexists(newcell)){
        if (dim == 2){
            if (face_index == 0){
                cells[1] = newcell;
                cells[3] = newcell;
                levels[1] = level;
                levels[3] = level;
            }
            else if (face_index == 1){
                cells[0] = newcell;
                cells[2] = newcell;
                levels[0] = level;
                levels[2] = level;
            }
            else if (face_index == 2){
                cells[0] = newcell;
                cells[1] = newcell;
                levels[0] = level;
                levels[1] = level;
            }
            else if (face_index == 3){
                cells[0] = newcell;
                cells[1] = newcell;
                levels[0] = level;
                levels[1] = level;
            }
        }
        else if (dim == 3){
            if (face_index == 0){
                cells[1] = newcell;
                cells[3] = newcell;
                cells[5] = newcell;
                cells[7] = newcell;
                levels[1] = level;
                levels[3] = level;
                levels[5] = level;
                levels[7] = level;
            }
            else if (face_index == 1){
                cells[0] = newcell;
                cells[2] = newcell;
                cells[4] = newcell;
                cells[6] = newcell;
                levels[0] = level;
                levels[2] = level;
                levels[4] = level;
                levels[6] = level;
            }
            else if (face_index == 2){
                cells[2] = newcell;
                cells[3] = newcell;
                cells[6] = newcell;
                cells[7] = newcell;
                levels[2] = level;
                levels[3] = level;
                levels[6] = level;
                levels[7] = level;
            }
            else if (face_index == 3){
                cells[0] = newcell;
                cells[1] = newcell;
                cells[4] = newcell;
                cells[5] = newcell;
                levels[0] = level;
                levels[1] = level;
                levels[4] = level;
                levels[5] = level;
            }
            else if (face_index == 4){
                cells[4] = newcell;
                cells[5] = newcell;
                cells[6] = newcell;
                cells[7] = newcell;
                levels[4] = level;
                levels[5] = level;
                levels[6] = level;
                levels[7] = level;
            }
            else if (face_index == 5){
                cells[0] = newcell;
                cells[1] = newcell;
                cells[2] = newcell;
                cells[3] = newcell;
                levels[0] = level;
                levels[1] = level;
                levels[2] = level;
                levels[3] = level;
            }
        }
    }
}

template <int dim>
void ControlVolume<dim>::addCell_edge(CellId newcell, int edge_index, int level){
    if (!checkIfcellexists(newcell)){
        if (dim == 3){
            if (edge_index == 0){
                cells[5] = newcell;
                cells[7] = newcell;
                levels[5] = level;
                levels[7] = level;
            }
            else if (edge_index == 1){
                cells[4] = newcell;
                cells[6] = newcell;
                levels[4] = level;
                levels[6] = level;
            }
            else if (edge_index == 2){
                cells[6] = newcell;
                cells[7] = newcell;
                levels[6] = level;
                levels[7] = level;
            }
            else if (edge_index == 3){
                cells[4] = newcell;
                cells[5] = newcell;
                levels[4] = level;
                levels[5] = level;
            }
            else if (edge_index == 4){
                cells[1] = newcell;
                cells[3] = newcell;
                levels[1] = level;
                levels[3] = level;
            }
            else if (edge_index == 5){
                cells[0] = newcell;
                cells[2] = newcell;
                levels[0] = level;
                levels[2] = level;
            }
            else if (edge_index == 6){
                cells[2] = newcell;
                cells[3] = newcell;
                levels[2] = level;
                levels[3] = level;
            }
            else if (edge_index == 7){
                cells[0] = newcell;
                cells[1] = newcell;
                levels[0] = level;
                levels[1] = level;
            }
            else if (edge_index == 8){
                cells[3] = newcell;
                cells[7] = newcell;
                levels[3] = level;
                levels[7] = level;
            }
            else if (edge_index == 9){
                cells[2] = newcell;
                cells[6] = newcell;
                levels[2] = level;
                levels[6] = level;
            }
            else if (edge_index == 10){
                cells[1] = newcell;
                cells[5] = newcell;
                levels[1] = level;
                levels[5] = level;
            }
            else if (edge_index == 11){
                cells[0] = newcell;
                cells[4] = newcell;
                levels[0] = level;
                levels[4] = level;
            }
        }
    }
}

template <int dim>
int ControlVolume<dim>::Ncells(){
    return cells.size();
}

template <int dim>
CellId ControlVolume<dim>::getCellID(int ii){
    return cells[ii];
}

template <int dim>
int ControlVolume<dim>::getLevel(int ii){
    return levels[ii];
}

template <int dim>
int ControlVolume<dim>::maxLevel(){
    int maxlevel = 0;
    for (unsigned int i = 0; i < levels.size(); ++i){
        if (levels[i] > maxlevel)
            maxlevel = levels[i];
    }
    return maxlevel;
}

template <int dim>
class CellInfoRT0{
public:
    CellInfoRT0();

    bool calculateSubcellFlows(typename DoFHandler<dim>::active_cell_iterator& cell,
                               FE_Q<dim>&                          fe,
                               TrilinosWrappers::MPI::Vector& 	locally_relevant_solution,
                               MyTensorFunction<dim> &HK_function);

    double outflow_from_subcell_with_vertex(int ivertex);

    std::vector<int> getPairIDs(int iface);
    void setcellSideFlow(int iface, double value);
    void setcellSideFlow(int iface, int subface, double value);
    int getBCtype(int iface);
protected:
    //! constructPW returns the mid Points positions in the unit cell and the weights.
    //! The weights are only needed to construct the quadrature formula and they are all equal summing to 1
    void midCellQuadPoints(std::vector<Point<dim>>& p);

    void midFaceQuadPoints(std::vector<Point<dim>>& p);


    //! For the time being it is assume that the hydraulic conductivity is isotropic in X and Y and that the faces are oriented in allignement
    //! with the principal axis
    void calculateFlows(std::vector<double>& flowvolume_out,
                        std::vector<double> midHeads,
                        std::vector<Tensor<2, dim> > HK,
                        std::vector<double> facearea);

    //! For a given cell described by ite vertices, that follows the dea numbering convention
    //! Calculate the areas/lengths of the inner subfaces and stores the in the returned array.
    //! The ordering of the returned array is the same as described in #flowVolume
    std::vector<double> calculateInnerFaceAreas(std::vector<Point<dim>> vertices);
    void calc_helper_points(std::vector<Point<dim>> vertices,
                            std::vector<Point<dim>>& mid_edge_points,
                            std::vector<Point<dim>>& mid_face_points,
                            Point<dim> &CenterPoint);

    void calculateSubfacesArea(std::vector<Point<dim>> vertices);
    void calc_SubfacesAreaofSubvolumes(int subvolume, std::vector<Point<dim>> vertices, std::vector<Point<dim>>& subvolumeVertices);

    void calculate_heads(typename DoFHandler<dim>::active_cell_iterator& cell,
                         FE_Q<dim>& fe,
                         TrilinosWrappers::MPI::Vector& locally_relevant_solution,
                         std::vector<double>& heads,
                         std::vector<Point<dim>> unit_points);

    std::vector<Tensor<2,dim>> calculateHK(std::vector<Point<dim>> vertices,MyTensorFunction<dim> &HK_function);




    //! This contains the heads in the corner of this cell.
    //! Since we are using linear elements only this corresponds to the dofs head solution
    std::vector<double> cornerHead;

    //! This contains the heads on the middle point of the subcells
    std::vector<double> midCellHead;

    std::map<int, std::vector<double>> submidCellHead;

    //! This contains the heads on the middle point on the edge for each subcell
    std::vector<double> midFaceHead;

    std::map<int, std::vector<double>> submidFaceHead;

    // CellSide
    std::vector<std::vector<double>> cellSideFlow;

    /*!
     * Type of boundary condition
     * -1 Its no boundary
     * 0 No flow
     * 1 Constant head
     * 2 Constant flow
    */
    std::vector<int> BCtype;

    void SetBCType(int iface, int bctype);




    /*!
     * \brief flowVolume is a vector that stores the flow between the subcells of the cell
     *
     * The numbering is defined as follows:
     *
     *      2D and 3D faces that touch the bottom of the cell(face 4)
     *  *---------------*
     *  |       |     3 |
     *  |  ^    3-> ^   |
     *  |  |  2 |   |   |       The numbers inside the cells
     *  |--0----+---1---|       are the subcell numbers
     *  |       |       |
     *  |   0   2->  1  |
     *  |       |       |
     *  *---------------*
     *
     *     3D of faces that touch the top of the cell (face 5)
     *  *---------------*
     *  |       |       |
     *  |  6    7   7   |
     *  |       |       |
     *  |--4----+---5---|
     *  |       |       |
     *  |  4    6   5   |
     *  |       |       |
     *  *---------------*
     *
     *      3D of face between the top and bottom subsells
     *  *---------------*
     *  |       |       |   The flows are calculated from the
     *  |  10   |  11   |   lower to the upper cell. from 0 ->4 etc.
     *  |       |       |
     *  |-------+-------|
     *  |       |       |
     *  |   8   |   9   |
     *  |       |       |
     *  *---------------*
     *
     */
    std::vector<double> flowVolume;

    //! For the cells that have refined neighbors we compute the subvolumes as well
    //! The key of the map corresponds to the subcell ID
    std::map<int, std::vector<double>> subflowVolume;


    //! This will hold the area in 3D or the legth in 2D of the corresponding face.
    //!  This vector follows the same numbering as the #flowVolume vector
    std::vector<double> area;

    //! This holds the area for each
    std::map<int, std::vector<double>> subarea;

    unsigned int Nsubfaces;

    void construct_subPW(int iface, typename std::map<int, std::vector<Point<dim>>>& sub_quad_points);
};

template<int dim>
CellInfoRT0<dim>::CellInfoRT0(){

    cellSideFlow.resize(GeometryInfo<dim>::faces_per_cell);
    for (unsigned int ii = 0; ii < GeometryInfo<dim>::faces_per_cell; ++ii){
        cellSideFlow[ii].resize((dim-1)*2);
    }

    if (dim ==2){
        Nsubfaces = 4;
    }
    else if (dim == 3){
        Nsubfaces = 12;
    }
    BCtype.resize(GeometryInfo<dim>::faces_per_cell);
}

template <int dim>
bool CellInfoRT0<dim>::calculateSubcellFlows(typename DoFHandler<dim>::active_cell_iterator &cell,
                                             FE_Q<dim> &fe,
                                             TrilinosWrappers::MPI::Vector &locally_relevant_solution,
                                             MyTensorFunction<dim>& HK_function){
    //
    std::vector<Point<dim>> quad_points;
    midCellQuadPoints(quad_points);
    calculate_heads(cell, fe, locally_relevant_solution, midCellHead, quad_points);
    midFaceQuadPoints(quad_points);
    calculate_heads(cell, fe, locally_relevant_solution, midFaceHead, quad_points);

    std::vector<Point<dim>> vertices;
    for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell; i++){
        vertices.push_back(cell->vertex(i));
    }
    calculateSubfacesArea(vertices);
    std::vector<Tensor<2, dim> > HK = calculateHK(vertices,HK_function);
    calculateFlows(flowVolume, midCellHead, HK, area);

    for (unsigned int iface = 0; iface < GeometryInfo<dim>::faces_per_cell; ++iface){
        if (!cell->at_boundary(iface)){
            BCtype[iface] = -1;
            if (cell->neighbor(iface)->has_children()){
                // Then calculate the subflow volumes for the subcells that touch the iface

                std::map<int, std::vector<Point<dim>>> sub_quad_points;
                construct_subPW(iface, sub_quad_points);
                typename std::map<int, std::vector<Point<dim>>>::iterator itsub = sub_quad_points.begin();
                for (; itsub != sub_quad_points.end(); ++itsub){
                    std::vector<double> midsubcell_heads;
                    calculate_heads(cell, fe, locally_relevant_solution, midsubcell_heads, quad_points);
                    submidCellHead[itsub->first] = midsubcell_heads;
                    std::vector<Point<dim>> subvolumevertices;
                    calc_SubfacesAreaofSubvolumes(itsub->first, vertices, subvolumevertices);
                    std::vector<Tensor<2, dim> > subHK = calculateHK(subvolumevertices, HK_function);
                    std::vector<double> temp_subflowvolume;
                    calculateFlows(temp_subflowvolume, midsubcell_heads, subHK, subarea[itsub->first]);
                    subflowVolume[itsub->first] = temp_subflowvolume;
                }
            }
        }
        else{
            //std::cout << cell->face(iface)->user_index() << std::endl;
            if (cell->face(iface)->user_index() == 0)
                BCtype[iface] = 0;
            else if (cell->face(iface)->user_index() == 1)
                BCtype[iface] = 1;
            else{
                BCtype[iface] = 2;
            }
        }
    }
    return true;
}

template <int dim>
void CellInfoRT0<dim>::setcellSideFlow(int iface, int subface, double value){
    cellSideFlow[iface][subface] = value;
}

template <int dim>
void CellInfoRT0<dim>::setcellSideFlow(int iface, double value){
    for (unsigned int i = 0; i < cellSideFlow[iface].size(); ++i)
        cellSideFlow[iface][i] = value;
}

template <int dim>
int CellInfoRT0<dim>::getBCtype(int iface){
    return BCtype[iface];
}

template <int dim>
void CellInfoRT0<dim>::midCellQuadPoints(std::vector<Point<dim>>& p){
    p.clear();
    Point<dim> p_tmp;
    if (dim == 2){
        p_tmp[0] = 0.25; p_tmp[1] = 0.25;
        p.push_back(p_tmp);
        p_tmp[0] = 0.75; p_tmp[1] = 0.25;
        p.push_back(p_tmp);
        p_tmp[0] = 0.25; p_tmp[1] = 0.75;
        p.push_back(p_tmp);
        p_tmp[0] = 0.75; p_tmp[1] = 0.75;
        p.push_back(p_tmp);
    }
    else if (dim == 3){
        p_tmp[0] = 0.25; p_tmp[1] = 0.25; p_tmp[2] = 0.25;
        p.push_back(p_tmp);
        p_tmp[0] = 0.75; p_tmp[1] = 0.25; p_tmp[2] = 0.25;
        p.push_back(p_tmp);
        p_tmp[0] = 0.25; p_tmp[1] = 0.75; p_tmp[2] = 0.25;
        p.push_back(p_tmp);
        p_tmp[0] = 0.75; p_tmp[1] = 0.75; p_tmp[2] = 0.25;
        p.push_back(p_tmp);
        p_tmp[0] = 0.25; p_tmp[1] = 0.25; p_tmp[2] = 0.75;
        p.push_back(p_tmp);
        p_tmp[0] = 0.75; p_tmp[1] = 0.25; p_tmp[2] = 0.75;
        p.push_back(p_tmp);
        p_tmp[0] = 0.25; p_tmp[1] = 0.75; p_tmp[2] = 0.75;
        p.push_back(p_tmp);
        p_tmp[0] = 0.75; p_tmp[1] = 0.75; p_tmp[2] = 0.75;
        p.push_back(p_tmp);
    }
}

template <int dim>
void CellInfoRT0<dim>::midFaceQuadPoints(std::vector<Point<dim>>& p){
    p.clear();
    Point<dim> p_tmp;
    if (dim == 2){
        // face 0
        p_tmp[0] = 0.0; p_tmp[1] = 0.25; p.push_back(p_tmp);
        p_tmp[0] = 0.0; p_tmp[1] = 0.75; p.push_back(p_tmp);
        // face 1
        p_tmp[0] = 1.0; p_tmp[1] = 0.25; p.push_back(p_tmp);
        p_tmp[0] = 1.0; p_tmp[1] = 0.75; p.push_back(p_tmp);
        // face 2
        p_tmp[0] = 0.25; p_tmp[1] = 0.0; p.push_back(p_tmp);
        p_tmp[0] = 0.75; p_tmp[1] = 0.0; p.push_back(p_tmp);
        // face 3
        p_tmp[0] = 0.25; p_tmp[1] = 1.0; p.push_back(p_tmp);
        p_tmp[0] = 0.75; p_tmp[1] = 1.0; p.push_back(p_tmp);
    }
    else if (dim == 3){
        // face 0
        p_tmp[0] = 0.0; p_tmp[1] = 0.25; p_tmp[2] = 0.25; p.push_back(p_tmp);
        p_tmp[0] = 0.0; p_tmp[1] = 0.75; p_tmp[2] = 0.25; p.push_back(p_tmp);
        p_tmp[0] = 0.0; p_tmp[1] = 0.25; p_tmp[2] = 0.75; p.push_back(p_tmp);
        p_tmp[0] = 0.0; p_tmp[1] = 0.75; p_tmp[2] = 0.75; p.push_back(p_tmp);
        // face 1
        p_tmp[0] = 1.0; p_tmp[1] = 0.25; p_tmp[2] = 0.25; p.push_back(p_tmp);
        p_tmp[0] = 1.0; p_tmp[1] = 0.75; p_tmp[2] = 0.25; p.push_back(p_tmp);
        p_tmp[0] = 1.0; p_tmp[1] = 0.25; p_tmp[2] = 0.75; p.push_back(p_tmp);
        p_tmp[0] = 1.0; p_tmp[1] = 0.75; p_tmp[2] = 0.75; p.push_back(p_tmp);
        // face 2
        p_tmp[0] = 0.25; p_tmp[1] = 0.0; p_tmp[2] = 0.25; p.push_back(p_tmp);
        p_tmp[0] = 0.75; p_tmp[1] = 0.0; p_tmp[2] = 0.25; p.push_back(p_tmp);
        p_tmp[0] = 0.25; p_tmp[1] = 0.0; p_tmp[2] = 0.75; p.push_back(p_tmp);
        p_tmp[0] = 0.75; p_tmp[1] = 0.0; p_tmp[2] = 0.75; p.push_back(p_tmp);
        // face 3
        p_tmp[0] = 0.25; p_tmp[1] = 1.0; p_tmp[2] = 0.25; p.push_back(p_tmp);
        p_tmp[0] = 0.75; p_tmp[1] = 1.0; p_tmp[2] = 0.25; p.push_back(p_tmp);
        p_tmp[0] = 0.25; p_tmp[1] = 1.0; p_tmp[2] = 0.75; p.push_back(p_tmp);
        p_tmp[0] = 0.75; p_tmp[1] = 1.0; p_tmp[2] = 0.75; p.push_back(p_tmp);
        // face 4
        p_tmp[0] = 0.25; p_tmp[1] = 0.25; p_tmp[2] = 0.0; p.push_back(p_tmp);
        p_tmp[0] = 0.75; p_tmp[1] = 0.25; p_tmp[2] = 0.0; p.push_back(p_tmp);
        p_tmp[0] = 0.25; p_tmp[1] = 0.75; p_tmp[2] = 0.0; p.push_back(p_tmp);
        p_tmp[0] = 0.75; p_tmp[1] = 0.75; p_tmp[2] = 0.0; p.push_back(p_tmp);
        // face 5
        p_tmp[0] = 0.25; p_tmp[1] = 0.25; p_tmp[2] = 1.0; p.push_back(p_tmp);
        p_tmp[0] = 0.75; p_tmp[1] = 0.25; p_tmp[2] = 1.0; p.push_back(p_tmp);
        p_tmp[0] = 0.25; p_tmp[1] = 0.75; p_tmp[2] = 1.0; p.push_back(p_tmp);
        p_tmp[0] = 0.75; p_tmp[1] = 0.75; p_tmp[2] = 1.0; p.push_back(p_tmp);
    }
}

template <int dim>
void CellInfoRT0<dim>::calculateFlows(std::vector<double>& flowvolume_out,
                                      std::vector<double> midHeads,
                                      std::vector<Tensor<2, dim> > HK,
                                      std::vector<double> facearea){


    double DH, Kx, Kz, Q;
    std::vector<int> id;
    flowvolume_out.clear();
    if (dim == 2){
        for (unsigned int i = 0; i < Nsubfaces; i++){
            id = getPairIDs(i);
            DH = midHeads[id[0]] - midHeads[id[1]];
            Kx = 2.0/(1.0/HK[id[0]][0][0] + 1.0/HK[id[1]][0][0]);
            Kz = 2.0/(1.0/HK[id[0]][1][1] + 1.0/HK[id[1]][1][1]);
            if (i < 2)
                Q = facearea[i]*Kz*DH;
            else
                Q = facearea[i]*Kx*DH;
            flowvolume_out.push_back(Q);
        }
    }
    else if (dim == 3){
        for (unsigned int i = 0; i < Nsubfaces; i++){
            id = getPairIDs(i);
            DH = midHeads[id[0]] - midHeads[id[1]];
            Kx = 2.0/(1.0/HK[id[0]][0][0] + 1.0/HK[id[1]][0][0]);
            Kz = 2.0/(1.0/HK[id[0]][2][2] + 1.0/HK[id[1]][2][2]);

            if (i < 8){
                Q = facearea[i]*Kx*DH;
            }
            else
                Q = facearea[i]*Kz*DH;
            flowvolume_out.push_back(Q);
        }
    }
}

template <int dim>
std::vector<int> CellInfoRT0<dim>:: getPairIDs(int iface){
    std::vector<int> ids;
    if (dim == 2){
        switch (iface) {
        case 0:
            ids.push_back(0);
            ids.push_back(2);
            break;
        case 1:
            ids.push_back(1);
            ids.push_back(3);
            break;
        case 2:
            ids.push_back(0);
            ids.push_back(1);
            break;
        case 3:
            ids.push_back(2);
            ids.push_back(3);
            break;
        default:
            break;
        }
    }
    else if (dim == 3){
        switch (iface) {
        case 0:
            ids.push_back(0);
            ids.push_back(2);
            break;
        case 1:
            ids.push_back(1);
            ids.push_back(3);
            break;
        case 2:
            ids.push_back(0);
            ids.push_back(1);
            break;
        case 3:
            ids.push_back(2);
            ids.push_back(3);
            break;
        case 4:
            ids.push_back(4);
            ids.push_back(6);
            break;
        case 5:
            ids.push_back(5);
            ids.push_back(7);
            break;
        case 6:
            ids.push_back(4);
            ids.push_back(5);
            break;
        case 7:
            ids.push_back(6);
            ids.push_back(7);
            break;
        case 8:
            ids.push_back(0);
            ids.push_back(4);
            break;
        case 9:
            ids.push_back(1);
            ids.push_back(5);
            break;
        case 10:
            ids.push_back(2);
            ids.push_back(6);
            break;
        case 11:
            ids.push_back(3);
            ids.push_back(7);
            break;
        default:
            break;
        }
    }
    return ids;
}

template <int dim>
void CellInfoRT0<dim>::calculateSubfacesArea(std::vector<Point<dim> > vertices){
    area.clear();

    std::vector<double> temp_area = calculateInnerFaceAreas(vertices);
    for (unsigned int ii = 0; ii < temp_area.size(); ++ii)
        area.push_back(temp_area[ii]);
}

template <int dim>
void CellInfoRT0<dim>::calc_SubfacesAreaofSubvolumes(int subvolume, std::vector<Point<dim>> vertices, std::vector<Point<dim>>& subvolumeVertices){

    std::vector<Point<dim>> midEdgePoints;
    std::vector<Point<dim>> midFacePoints;
    Point<dim> CellCenter;
    subvolumeVertices.clear();
    calc_helper_points(vertices, midEdgePoints, midFacePoints, CellCenter);
    if (dim == 2){
        if (subvolume == 0){
            subvolumeVertices.push_back(vertices[0]); subvolumeVertices.push_back(midFacePoints[2]);
            subvolumeVertices.push_back(midFacePoints[0]); subvolumeVertices.push_back(CellCenter);

        }
        else if (subvolume == 1){
            subvolumeVertices.push_back(midFacePoints[2]); subvolumeVertices.push_back(vertices[1]);
            subvolumeVertices.push_back(CellCenter); subvolumeVertices.push_back(midFacePoints[1]);
        }
        else if (subvolume == 2){
            subvolumeVertices.push_back(midFacePoints[0]); subvolumeVertices.push_back(CellCenter);
            subvolumeVertices.push_back(vertices[2]); subvolumeVertices.push_back(midFacePoints[3]);
        }
        else if (subvolume == 3){
            subvolumeVertices.push_back(CellCenter); subvolumeVertices.push_back(midFacePoints[1]);
            subvolumeVertices.push_back(midFacePoints[3]); subvolumeVertices.push_back(vertices[3]);
        }
    }
    else if (dim == 3){
        if (subvolume == 0){
            subvolumeVertices.push_back(vertices[0]); subvolumeVertices.push_back(midEdgePoints[2]); subvolumeVertices.push_back(midEdgePoints[0]); subvolumeVertices.push_back(midFacePoints[4]);
            subvolumeVertices.push_back(midEdgePoints[8]); subvolumeVertices.push_back(midFacePoints[2]); subvolumeVertices.push_back(midFacePoints[0]); subvolumeVertices.push_back(CellCenter);
        }
        else if (subvolume == 1){
            subvolumeVertices.push_back(midEdgePoints[2]); subvolumeVertices.push_back(vertices[1]); subvolumeVertices.push_back(midFacePoints[4]); subvolumeVertices.push_back(midEdgePoints[1]);
            subvolumeVertices.push_back(midFacePoints[2]); subvolumeVertices.push_back(midEdgePoints[9]); subvolumeVertices.push_back(CellCenter); subvolumeVertices.push_back(midFacePoints[1]);
        }
        else if (subvolume == 2){
            subvolumeVertices.push_back(midEdgePoints[0]); subvolumeVertices.push_back(midFacePoints[4]); subvolumeVertices.push_back(vertices[2]); subvolumeVertices.push_back(midEdgePoints[3]);
            subvolumeVertices.push_back(midFacePoints[0]); subvolumeVertices.push_back(CellCenter); subvolumeVertices.push_back(midEdgePoints[10]); subvolumeVertices.push_back(midFacePoints[3]);
        }
        else if (subvolume == 3){
            subvolumeVertices.push_back(midFacePoints[4]); subvolumeVertices.push_back(midEdgePoints[1]); subvolumeVertices.push_back(midEdgePoints[3]); subvolumeVertices.push_back(vertices[3]);
            subvolumeVertices.push_back(CellCenter); subvolumeVertices.push_back(midFacePoints[1]); subvolumeVertices.push_back(midFacePoints[3]); subvolumeVertices.push_back(midEdgePoints[11]);
        }
        else if (subvolume == 4){
            subvolumeVertices.push_back(midEdgePoints[8]); subvolumeVertices.push_back(midFacePoints[2]); subvolumeVertices.push_back(midFacePoints[0]); subvolumeVertices.push_back(CellCenter);
            subvolumeVertices.push_back(vertices[4]); subvolumeVertices.push_back(midEdgePoints[6]); subvolumeVertices.push_back(midEdgePoints[4]); subvolumeVertices.push_back(midFacePoints[5]);
        }
        else if (subvolume == 5){
            subvolumeVertices.push_back(midFacePoints[2]); subvolumeVertices.push_back(midEdgePoints[9]); subvolumeVertices.push_back(CellCenter); subvolumeVertices.push_back(midFacePoints[1]);
            subvolumeVertices.push_back(midEdgePoints[6]); subvolumeVertices.push_back(vertices[5]); subvolumeVertices.push_back(midFacePoints[5]); subvolumeVertices.push_back(midEdgePoints[5]);
        }
        else if (subvolume == 6){
            subvolumeVertices.push_back(midFacePoints[0]); subvolumeVertices.push_back(CellCenter); subvolumeVertices.push_back(midEdgePoints[10]); subvolumeVertices.push_back(midFacePoints[3]);
            subvolumeVertices.push_back(midEdgePoints[4]); subvolumeVertices.push_back(midFacePoints[5]); subvolumeVertices.push_back(vertices[6]); subvolumeVertices.push_back(midEdgePoints[7]);
        }
        else if (subvolume == 7){
            subvolumeVertices.push_back(CellCenter); subvolumeVertices.push_back(midFacePoints[1]); subvolumeVertices.push_back(midFacePoints[3]); subvolumeVertices.push_back(midEdgePoints[11]);
            subvolumeVertices.push_back(midFacePoints[5]); subvolumeVertices.push_back(midEdgePoints[5]); subvolumeVertices.push_back(midEdgePoints[7]); subvolumeVertices.push_back(vertices[7]);
        }
    }
    std::vector<double> area_array = calculateInnerFaceAreas(subvolumeVertices);
    subarea[subvolume] = area_array;
}

template <int dim>
void CellInfoRT0<dim>::calc_helper_points(std::vector<Point<dim>> vertices,
                                          std::vector<Point<dim>>& mid_edge_points,
                                          std::vector<Point<dim>>& mid_face_points,
                                          Point<dim>& CenterPoint){
    mid_edge_points.clear();
    mid_face_points.clear();
    if (dim == 2){
        mid_face_points.push_back(midPoint<dim>(vertices[0], vertices[2])); // 0 edge
        mid_face_points.push_back(midPoint<dim>(vertices[1], vertices[3])); // 1 edge
        mid_face_points.push_back(midPoint<dim>(vertices[0], vertices[1])); // 2 edge
        mid_face_points.push_back(midPoint<dim>(vertices[2], vertices[3])); // 3 edge
        CenterPoint = midPoint<dim>(mid_face_points[0], mid_face_points[1]);
    }
    else if (dim == 3){
        mid_edge_points.push_back(midPoint<dim>(vertices[0], vertices[2])); // 0 edge
        mid_edge_points.push_back(midPoint<dim>(vertices[1], vertices[3])); // 1 edge
        mid_edge_points.push_back(midPoint<dim>(vertices[0], vertices[1])); // 2 edge
        mid_edge_points.push_back(midPoint<dim>(vertices[2], vertices[3])); // 3 edge
        mid_edge_points.push_back(midPoint<dim>(vertices[4], vertices[6])); // 4 edge
        mid_edge_points.push_back(midPoint<dim>(vertices[5], vertices[7])); // 5 edge
        mid_edge_points.push_back(midPoint<dim>(vertices[4], vertices[5])); // 6 edge
        mid_edge_points.push_back(midPoint<dim>(vertices[6], vertices[7])); // 7 edge
        mid_edge_points.push_back(midPoint<dim>(vertices[0], vertices[4])); // 8 edge
        mid_edge_points.push_back(midPoint<dim>(vertices[1], vertices[5])); // 9 edge
        mid_edge_points.push_back(midPoint<dim>(vertices[2], vertices[6])); // 10 edge
        mid_edge_points.push_back(midPoint<dim>(vertices[3], vertices[7])); // 11 edge

        mid_face_points.push_back(midPoint<dim>(mid_edge_points[0], mid_edge_points[4])); // 0 face
        mid_face_points.push_back(midPoint<dim>(mid_edge_points[1], mid_edge_points[5])); // 1 face
        mid_face_points.push_back(midPoint<dim>(mid_edge_points[2], mid_edge_points[6])); // 2 face
        mid_face_points.push_back(midPoint<dim>(mid_edge_points[3], mid_edge_points[7])); // 3 face
        mid_face_points.push_back(midPoint<dim>(mid_edge_points[0], mid_edge_points[1])); // 4 face
        mid_face_points.push_back(midPoint<dim>(mid_edge_points[4], mid_edge_points[5])); // 5 face

        CenterPoint = midPoint<dim>(mid_edge_points[0], mid_edge_points[1]);
    }

}

template <int dim>
std::vector<double> CellInfoRT0<dim>::calculateInnerFaceAreas(std::vector<Point<dim>> vertices){
    std::vector<double> area_array;
    std::vector<Point<dim>> midEdgePoints;
    std::vector<Point<dim>> midFacePoints;
    Point<dim> CellCenter;
    if (dim == 2){

        calc_helper_points(vertices, midEdgePoints, midFacePoints, CellCenter);

        // In 2D the area is actual the length of a unit width
        // Subface 0
        area_array.push_back(midFacePoints[0].distance(CellCenter));
        // Subface 1
        area_array.push_back(midFacePoints[1].distance(CellCenter));
        // Subface 2
        area_array.push_back(midFacePoints[2].distance(CellCenter));
        // Subface 3
        area_array.push_back(midFacePoints[2].distance(CellCenter));
    }
    else if (dim == 3){
        // The mid edge point numbering is identical with the deal edge numbering
        // and the mid face numbering is identical to deal face numbering
        // see the GeometryInfo documentation

        calc_helper_points(vertices, midEdgePoints, midFacePoints, CellCenter);

        double ar1, ar2;
        // Subface 0
        ar1 = triangle_area<dim>(midEdgePoints[0], midFacePoints[4], CellCenter,false);
        ar2 = triangle_area<dim>(midEdgePoints[0], CellCenter, midFacePoints[0],false);
        area_array.push_back(ar1 + ar2);
        // Subface 1
        ar1 = triangle_area<dim>(midEdgePoints[1], midFacePoints[4], CellCenter,false);
        ar2 = triangle_area<dim>(midEdgePoints[1], CellCenter, midFacePoints[1],false);
        area_array.push_back(ar1 + ar2);
        // Subface 2
        ar1 = triangle_area<dim>(midEdgePoints[2], midFacePoints[4], CellCenter,false);
        ar2 = triangle_area<dim>(midEdgePoints[2], CellCenter, midFacePoints[2],false);
        area_array.push_back(ar1 + ar2);
        // Subface 3
        ar1 = triangle_area<dim>(midEdgePoints[3], midFacePoints[4], CellCenter,false);
        ar2 = triangle_area<dim>(midEdgePoints[3], CellCenter, midFacePoints[3],false);
        area_array.push_back(ar1 + ar2);
        // Subface 4
        ar1 = triangle_area<dim>(midEdgePoints[4], midFacePoints[0], CellCenter,false);
        ar2 = triangle_area<dim>(midEdgePoints[4], CellCenter, midFacePoints[5],false);
        area_array.push_back(ar1 + ar2);
        // Subface 5
        ar1 = triangle_area<dim>(midEdgePoints[5], midFacePoints[1], CellCenter,false);
        ar2 = triangle_area<dim>(midEdgePoints[5], CellCenter, midFacePoints[5],false);
        area_array.push_back(ar1 + ar2);
        // Subface 6
        ar1 = triangle_area<dim>(midEdgePoints[6], midFacePoints[2], CellCenter,false);
        ar2 = triangle_area<dim>(midEdgePoints[6], CellCenter, midFacePoints[5],false);
        area_array.push_back(ar1 + ar2);
        // Subface 7
        ar1 = triangle_area<dim>(midEdgePoints[7], midFacePoints[3], CellCenter,false);
        ar2 = triangle_area<dim>(midEdgePoints[7], CellCenter, midFacePoints[5],false);
        area_array.push_back(ar1 + ar2);
        // Subface 8
        ar1 = triangle_area<dim>(midEdgePoints[8], midFacePoints[0], CellCenter,false);
        ar2 = triangle_area<dim>(midEdgePoints[8], CellCenter, midFacePoints[2],false);
        area_array.push_back(ar1 + ar2);
        // Subface 9
        ar1 = triangle_area<dim>(midEdgePoints[9], midFacePoints[1], CellCenter,false);
        ar2 = triangle_area<dim>(midEdgePoints[9], CellCenter, midFacePoints[2],false);
        area_array.push_back(ar1 + ar2);
        // Subface 10
        ar1 = triangle_area<dim>(midEdgePoints[10], midFacePoints[0], CellCenter,false);
        ar2 = triangle_area<dim>(midEdgePoints[10], CellCenter, midFacePoints[3],false);
        area_array.push_back(ar1 + ar2);
        // Subface 11
        ar1 = triangle_area<dim>(midEdgePoints[11], midFacePoints[1], CellCenter,false);
        ar2 = triangle_area<dim>(midEdgePoints[11], CellCenter, midFacePoints[3],false);
        area_array.push_back(ar1 + ar2);
    }
    return area_array;
}

template <int dim>
std::vector<Tensor<2, dim> > CellInfoRT0<dim>::calculateHK(std::vector<Point<dim> > vertices, MyTensorFunction<dim> &HK_function){
    Point<dim> tmp;
    std::vector<Tensor<2,dim>> HK;
    if (dim == 2){
        tmp[0] = vertices[0][0]*0.5625 + vertices[1][0]*0.1875 + vertices[2][0]*0.1875 + vertices[3][0]*0.0625;
        tmp[1] = vertices[0][1]*0.5625 + vertices[1][1]*0.1875 + vertices[2][1]*0.1875 + vertices[3][1]*0.0625;
        HK.push_back(HK_function.value(tmp));
        tmp[0] = vertices[0][0]*0.1875 + vertices[1][0]*0.5625 + vertices[2][0]*0.0625 + vertices[3][0]*0.1875;
        tmp[1] = vertices[0][1]*0.1875 + vertices[1][1]*0.5625 + vertices[2][1]*0.0625 + vertices[3][1]*0.1875;
        HK.push_back(HK_function.value(tmp));
        tmp[0] = vertices[0][0]*0.1875 + vertices[1][0]*0.0625 + vertices[2][0]*0.5625 + vertices[3][0]*0.1875;
        tmp[1] = vertices[0][1]*0.1875 + vertices[1][1]*0.0625 + vertices[2][1]*0.5625 + vertices[3][1]*0.1875;
        HK.push_back(HK_function.value(tmp));
        tmp[0] = vertices[0][0]*0.0625 + vertices[1][0]*0.1875 + vertices[2][0]*0.1875 + vertices[3][0]*0.5625;
        tmp[1] = vertices[0][1]*0.0625 + vertices[1][1]*0.1875 + vertices[2][1]*0.1875 + vertices[3][1]*0.5625;
        HK.push_back(HK_function.value(tmp));
    }
    else if (dim == 3){
        double w1 = 0.421875;
        double w2 = 0.140625;
        double w3 = 0.046875;
        double w4 = 0.015625;
        // vertex 0
        for (unsigned int idim = 0; idim < dim; idim++)
            tmp[idim] = vertices[0][idim]*w1 + vertices[1][idim]*w2 + vertices[2][idim]*w2 + vertices[3][idim]*w3 +
                        vertices[4][idim]*w2 + vertices[5][idim]*w3 + vertices[6][idim]*w3 + vertices[7][idim]*w4;
        HK.push_back(HK_function.value(tmp));

        // vertex 1
        for (unsigned int idim = 0; idim < dim; idim++)
            tmp[idim] = vertices[0][idim]*w2 + vertices[1][idim]*w1 + vertices[2][idim]*w3 + vertices[3][idim]*w2 +
                        vertices[4][idim]*w3 + vertices[5][idim]*w2 + vertices[6][idim]*w4 + vertices[7][idim]*w3;
        HK.push_back(HK_function.value(tmp));

        //vertex 2
        for (unsigned int idim = 0; idim < dim; idim++)
            tmp[idim] = vertices[0][idim]*w2 + vertices[1][idim]*w3 + vertices[2][idim]*w1 + vertices[3][idim]*w2 +
                        vertices[4][idim]*w3 + vertices[5][idim]*w4 + vertices[6][idim]*w2 + vertices[7][idim]*w3;
        HK.push_back(HK_function.value(tmp));

        //vertex3
        for (unsigned int idim = 0; idim < dim; idim++)
            tmp[idim] = vertices[0][idim]*w3 + vertices[1][idim]*w2 + vertices[2][idim]*w2 + vertices[3][idim]*w1 +
                        vertices[4][idim]*w4 + vertices[5][idim]*w3 + vertices[6][idim]*w3 + vertices[7][idim]*w2;
        HK.push_back(HK_function.value(tmp));

        //vertex4
        for (unsigned int idim = 0; idim < dim; idim++)
            tmp[idim] = vertices[0][idim]*w2 + vertices[1][idim]*w3 + vertices[2][idim]*w3 + vertices[3][idim]*w4 +
                        vertices[4][idim]*w1 + vertices[5][idim]*w2 + vertices[6][idim]*w2 + vertices[7][idim]*w3;
        HK.push_back(HK_function.value(tmp));

        //vertex5
        for (unsigned int idim = 0; idim < dim; idim++)
            tmp[idim] = vertices[0][idim]*w3 + vertices[1][idim]*w2 + vertices[2][idim]*w4 + vertices[3][idim]*w3 +
                        vertices[4][idim]*w2 + vertices[5][idim]*w1 + vertices[6][idim]*w3 + vertices[7][idim]*w2;
        HK.push_back(HK_function.value(tmp));

        //vertex6
        for (unsigned int idim = 0; idim < dim; idim++)
            tmp[idim] = vertices[0][idim]*w3 + vertices[1][idim]*w4 + vertices[2][idim]*w2 + vertices[3][idim]*w3 +
                        vertices[4][idim]*w2 + vertices[5][idim]*w3 + vertices[6][idim]*w1 + vertices[7][idim]*w2;
        HK.push_back(HK_function.value(tmp));

        //vertex7
        for (unsigned int idim = 0; idim < dim; idim++)
            tmp[idim] = vertices[0][idim]*w4 + vertices[1][idim]*w3 + vertices[2][idim]*w3 + vertices[3][idim]*w2 +
                        vertices[4][idim]*w3 + vertices[5][idim]*w2 + vertices[6][idim]*w2 + vertices[7][idim]*w1;
        HK.push_back(HK_function.value(tmp));
    }
    return HK;
}

template <int dim>
void CellInfoRT0<dim>::calculate_heads(typename DoFHandler<dim>::active_cell_iterator& cell,
                                       FE_Q<dim>& fe,
                                       TrilinosWrappers::MPI::Vector& locally_relevant_solution,
                                       std::vector<double>& heads,
                                       std::vector<Point<dim>> unit_points){

    heads.clear();
    std::vector<double> temp_weights(unit_points.size(), 1.0/static_cast<double>(unit_points.size()));
    Quadrature<dim> quadrature(unit_points,temp_weights);
    FEValues<dim> fe_values(fe, quadrature, update_values);
    fe_values.reinit(cell);
    heads.resize(unit_points.size());
    fe_values.get_function_values(locally_relevant_solution, heads);
}

template <int dim>
double CellInfoRT0<dim>::outflow_from_subcell_with_vertex(int ivertex){
    if (dim == 2){
        if (ivertex == 0)
            return flowVolume[0] + flowVolume[2];
        else if (ivertex == 1)
            return flowVolume[1] - flowVolume[2];
        else if (ivertex == 2)
            return -flowVolume[0] + flowVolume[3];
        else if (ivertex == 3)
            return -flowVolume[1] - flowVolume[3];

    }
    else if (dim == 3){
        if (ivertex == 0)
            return flowVolume[0] + flowVolume[2] + flowVolume[8];
        else if (ivertex == 1)
            return flowVolume[1] - flowVolume[2] + flowVolume[9];
        else if (ivertex == 2)
            return -flowVolume[0] + flowVolume[3] + flowVolume[10];
        else if (ivertex == 3)
            return -flowVolume[1] - flowVolume[3] + flowVolume[11];
        else if (ivertex == 4)
            return flowVolume[4] + flowVolume[6] - flowVolume[8];
        else if (ivertex == 5)
            return flowVolume[5] - flowVolume[6] - flowVolume[9];
        else if (ivertex == 6)
            return -flowVolume[4] + flowVolume[7] - flowVolume[10];
        else if (ivertex == 7)
            return -flowVolume[5] - flowVolume[7] - flowVolume[11];
    }
    return -999999.9;
}

template <int dim>
void CellInfoRT0<dim>::SetBCType(int iface, int bctype){
    if (dim == 2){
        if (iface == 0){

        }
    }
    else if (dim == 3){

    }
}

template <int dim>
void CellInfoRT0<dim>::construct_subPW(int iface, typename std::map<int, std::vector<Point<dim>>>& sub_quad_points){
    sub_quad_points.clear();
    std::map<int, std::vector<Point<dim>>> quad_per_subcell;
    std::vector<Point<dim>> pnts;
    Point<dim> p;
    if (dim == 2){
        p[0] =0.125;p[1] =0.125; pnts.push_back(p);
        p[0] =0.375;p[1] =0.125; pnts.push_back(p);
        p[0] =0.125;p[1] =0.375; pnts.push_back(p);
        p[0] =0.375;p[1] =0.375; pnts.push_back(p);
        quad_per_subcell[0] = pnts;
        pnts.clear();

        p[0] =0.625;p[1] =0.125; pnts.push_back(p);
        p[0] =0.875;p[1] =0.125; pnts.push_back(p);
        p[0] =0.625;p[1] =0.375; pnts.push_back(p);
        p[0] =0.875;p[1] =0.375; pnts.push_back(p);
        quad_per_subcell[1] = pnts;
        pnts.clear();

        p[0] =0.125;p[1] =0.625; pnts.push_back(p);
        p[0] =0.375;p[1] =0.625; pnts.push_back(p);
        p[0] =0.125;p[1] =0.875; pnts.push_back(p);
        p[0] =0.375;p[1] =0.875; pnts.push_back(p);
        quad_per_subcell[2] = pnts;
        pnts.clear();

        p[0] =0.625;p[1] =0.625; pnts.push_back(p);
        p[0] =0.875;p[1] =0.625; pnts.push_back(p);
        p[0] =0.625;p[1] =0.875; pnts.push_back(p);
        p[0] =0.875;p[1] =0.875; pnts.push_back(p);
        quad_per_subcell[3] = pnts;

        if (iface == 0){
            sub_quad_points[0] = quad_per_subcell[0];
            sub_quad_points[2] = quad_per_subcell[2];
        }
        else if (iface == 1){
            sub_quad_points[1] = quad_per_subcell[1];
            sub_quad_points[3] = quad_per_subcell[3];
        }
        else if (iface == 2){
            sub_quad_points[0] = quad_per_subcell[0];
            sub_quad_points[1] = quad_per_subcell[1];
        }
        else if (iface == 3){
            sub_quad_points[2] = quad_per_subcell[2];
            sub_quad_points[3] = quad_per_subcell[3];
        }
    }
    else if ( dim == 3){
        p[0] =0.125; p[1] =0.125; p[2] =0.125; pnts.push_back(p);
        p[0] =0.375; p[1] =0.125; p[2] =0.125; pnts.push_back(p);
        p[0] =0.125; p[1] =0.375; p[2] =0.125; pnts.push_back(p);
        p[0] =0.375; p[1] =0.375; p[2] =0.125; pnts.push_back(p);
        p[0] =0.125; p[1] =0.125; p[2] =0.375; pnts.push_back(p);
        p[0] =0.375; p[1] =0.125; p[2] =0.375; pnts.push_back(p);
        p[0] =0.125; p[1] =0.375; p[2] =0.375; pnts.push_back(p);
        p[0] =0.375; p[1] =0.375; p[2] =0.375; pnts.push_back(p);
        quad_per_subcell[0] = pnts;
        pnts.clear();

        p[0] =0.625; p[1] =0.125; p[2] =0.125; pnts.push_back(p);
        p[0] =0.875; p[1] =0.125; p[2] =0.125; pnts.push_back(p);
        p[0] =0.625; p[1] =0.375; p[2] =0.125; pnts.push_back(p);
        p[0] =0.875; p[1] =0.375; p[2] =0.125; pnts.push_back(p);
        p[0] =0.625; p[1] =0.125; p[2] =0.375; pnts.push_back(p);
        p[0] =0.875; p[1] =0.125; p[2] =0.375; pnts.push_back(p);
        p[0] =0.625; p[1] =0.375; p[2] =0.375; pnts.push_back(p);
        p[0] =0.875; p[1] =0.375; p[2] =0.375; pnts.push_back(p);
        quad_per_subcell[1] = pnts;
        pnts.clear();

        p[0] =0.125; p[1] =0.625; p[2] =0.125; pnts.push_back(p);
        p[0] =0.375; p[1] =0.625; p[2] =0.125; pnts.push_back(p);
        p[0] =0.125; p[1] =0.875; p[2] =0.125; pnts.push_back(p);
        p[0] =0.375; p[1] =0.875; p[2] =0.125; pnts.push_back(p);
        p[0] =0.125; p[1] =0.625; p[2] =0.375; pnts.push_back(p);
        p[0] =0.375; p[1] =0.625; p[2] =0.375; pnts.push_back(p);
        p[0] =0.125; p[1] =0.875; p[2] =0.375; pnts.push_back(p);
        p[0] =0.375; p[1] =0.875; p[2] =0.375; pnts.push_back(p);
        quad_per_subcell[2] = pnts;
        pnts.clear();

        p[0] =0.625; p[1] =0.625; p[2] =0.125; pnts.push_back(p);
        p[0] =0.875; p[1] =0.625; p[2] =0.125; pnts.push_back(p);
        p[0] =0.625; p[1] =0.875; p[2] =0.125; pnts.push_back(p);
        p[0] =0.875; p[1] =0.875; p[2] =0.125; pnts.push_back(p);
        p[0] =0.625; p[1] =0.625; p[2] =0.375; pnts.push_back(p);
        p[0] =0.875; p[1] =0.625; p[2] =0.375; pnts.push_back(p);
        p[0] =0.625; p[1] =0.875; p[2] =0.375; pnts.push_back(p);
        p[0] =0.875; p[1] =0.875; p[2] =0.375; pnts.push_back(p);
        quad_per_subcell[3] = pnts;
        pnts.clear();

        p[0] =0.125; p[1] =0.125; p[2] =0.625; pnts.push_back(p);
        p[0] =0.375; p[1] =0.125; p[2] =0.625; pnts.push_back(p);
        p[0] =0.125; p[1] =0.375; p[2] =0.625; pnts.push_back(p);
        p[0] =0.375; p[1] =0.375; p[2] =0.625; pnts.push_back(p);
        p[0] =0.125; p[1] =0.125; p[2] =0.875; pnts.push_back(p);
        p[0] =0.375; p[1] =0.125; p[2] =0.875; pnts.push_back(p);
        p[0] =0.125; p[1] =0.375; p[2] =0.875; pnts.push_back(p);
        p[0] =0.375; p[1] =0.375; p[2] =0.875; pnts.push_back(p);
        quad_per_subcell[4] = pnts;
        pnts.clear();

        p[0] =0.625; p[1] =0.125; p[2] =0.625; pnts.push_back(p);
        p[0] =0.875; p[1] =0.125; p[2] =0.625; pnts.push_back(p);
        p[0] =0.625; p[1] =0.375; p[2] =0.625; pnts.push_back(p);
        p[0] =0.875; p[1] =0.375; p[2] =0.625; pnts.push_back(p);
        p[0] =0.625; p[1] =0.125; p[2] =0.875; pnts.push_back(p);
        p[0] =0.875; p[1] =0.125; p[2] =0.875; pnts.push_back(p);
        p[0] =0.625; p[1] =0.375; p[2] =0.875; pnts.push_back(p);
        p[0] =0.875; p[1] =0.375; p[2] =0.875; pnts.push_back(p);
        quad_per_subcell[5] = pnts;
        pnts.clear();

        p[0] =0.125; p[1] =0.625; p[2] =0.625; pnts.push_back(p);
        p[0] =0.375; p[1] =0.625; p[2] =0.625; pnts.push_back(p);
        p[0] =0.125; p[1] =0.875; p[2] =0.625; pnts.push_back(p);
        p[0] =0.375; p[1] =0.875; p[2] =0.625; pnts.push_back(p);
        p[0] =0.125; p[1] =0.625; p[2] =0.875; pnts.push_back(p);
        p[0] =0.375; p[1] =0.625; p[2] =0.875; pnts.push_back(p);
        p[0] =0.125; p[1] =0.875; p[2] =0.875; pnts.push_back(p);
        p[0] =0.375; p[1] =0.875; p[2] =0.875; pnts.push_back(p);
        quad_per_subcell[6] = pnts;
        pnts.clear();

        p[0] =0.625; p[1] =0.625; p[2] =0.625; pnts.push_back(p);
        p[0] =0.875; p[1] =0.625; p[2] =0.625; pnts.push_back(p);
        p[0] =0.625; p[1] =0.875; p[2] =0.625; pnts.push_back(p);
        p[0] =0.875; p[1] =0.875; p[2] =0.625; pnts.push_back(p);
        p[0] =0.625; p[1] =0.625; p[2] =0.875; pnts.push_back(p);
        p[0] =0.875; p[1] =0.625; p[2] =0.875; pnts.push_back(p);
        p[0] =0.625; p[1] =0.875; p[2] =0.875; pnts.push_back(p);
        p[0] =0.875; p[1] =0.875; p[2] =0.875; pnts.push_back(p);
        quad_per_subcell[7] = pnts;
        pnts.clear();
        if (iface == 0){
            sub_quad_points[0] = quad_per_subcell[0];
            sub_quad_points[2] = quad_per_subcell[2];
            sub_quad_points[4] = quad_per_subcell[4];
            sub_quad_points[6] = quad_per_subcell[6];
        }
        else if (iface == 1){
            sub_quad_points[1] = quad_per_subcell[1];
            sub_quad_points[3] = quad_per_subcell[3];
            sub_quad_points[5] = quad_per_subcell[5];
            sub_quad_points[7] = quad_per_subcell[7];
        }
        else if (iface == 2){
            sub_quad_points[0] = quad_per_subcell[0];
            sub_quad_points[1] = quad_per_subcell[1];
            sub_quad_points[4] = quad_per_subcell[4];
            sub_quad_points[5] = quad_per_subcell[5];
        }
        else if (iface == 3){
            sub_quad_points[2] = quad_per_subcell[2];
            sub_quad_points[3] = quad_per_subcell[3];
            sub_quad_points[6] = quad_per_subcell[6];
            sub_quad_points[7] = quad_per_subcell[7];
        }
        else if (iface == 4){
            sub_quad_points[0] = quad_per_subcell[0];
            sub_quad_points[1] = quad_per_subcell[1];
            sub_quad_points[2] = quad_per_subcell[2];
            sub_quad_points[3] = quad_per_subcell[3];
        }
        else if (iface == 5){
            sub_quad_points[4] = quad_per_subcell[4];
            sub_quad_points[5] = quad_per_subcell[5];
            sub_quad_points[6] = quad_per_subcell[6];
            sub_quad_points[7] = quad_per_subcell[7];
        }
    }
}

template <int dim>
std::vector<int> FaceFlows2Consider(int subcellID){
    std::vector <int> out;
    if (dim == 2){

    }
    else if(dim == 3){

    }
}

template <int dim>
void subcellFlows2Consider(std::vector <int>& ids, std::vector <std::vector<int>>& dirs){
    ids.clear();
    dirs.clear();
    std::vector<int> temp, temp_dir;
    if (dim == 2){
//        //0
//        temp.push_back(1);temp.push_back(3);
//        temp_dir.push_back(-1);temp_dir.push_back(-1);
//        ids.push_back(temp); temp.clear();
//        dirs.push_back(temp_dir); temp_dir.clear();
//        //1
//        temp.push_back(0);temp.push_back(3);
//        temp_dir.push_back(-1);temp_dir.push_back(1);
//        ids.push_back(temp); temp.clear();
//        dirs.push_back(temp_dir); temp_dir.clear();
//        //2
//        temp.push_back(1);temp.push_back(2);
//        temp_dir.push_back(1);temp_dir.push_back(-1);
//        ids.push_back(temp); temp.clear();
//        dirs.push_back(temp_dir); temp_dir.clear();
//        //3
//        temp.push_back(0);temp.push_back(2);
//        temp_dir.push_back(1);temp_dir.push_back(1);
//        ids.push_back(temp); temp.clear();
//        dirs.push_back(temp_dir); temp_dir.clear();
        ids.clear();
        ids.push_back(3);
        ids.push_back(2);
        ids.push_back(1);
        ids.push_back(0);
    }
    else if(dim == 3){

    }
}

template <int dim>
void associate_unkowns_with_sub_volumes(std::vector<std::vector<int>>& unkown_dependence){
    unkown_dependence.clear();
    std::vector<int> temp;
    if (dim == 2){
        // 0 unknown describes the flow between 0 and 1 subcells
        temp.push_back(0);temp.push_back(1);
        unkown_dependence.push_back(temp); temp.clear();

        // 1 unknown describes the flow between 1 and 3 subcells
        temp.push_back(1);temp.push_back(3);
        unkown_dependence.push_back(temp); temp.clear();

        // 2 unknown describes the flow between 3 and 2 subcells
        temp.push_back(3);temp.push_back(2);
        unkown_dependence.push_back(temp); temp.clear();

        // 3 unknown describes the flow between 2 and 0 subcells
        temp.push_back(2);temp.push_back(0);
        unkown_dependence.push_back(temp); temp.clear();
    }
    else if (dim == 3){
        // 0 unknown describes the flow between 0 and 1 subcells
        temp.push_back(0);temp.push_back(1);
        unkown_dependence.push_back(temp); temp.clear();

        // 1 unknown describes the flow between 1 and 3 subcells
        temp.push_back(1);temp.push_back(3);
        unkown_dependence.push_back(temp); temp.clear();

        // 2 unknown describes the flow between 3 and 2 subcells
        temp.push_back(3);temp.push_back(2);
        unkown_dependence.push_back(temp); temp.clear();

        // 3 unknown describes the flow between 2 and 0 subcells
        temp.push_back(2);temp.push_back(0);
        unkown_dependence.push_back(temp); temp.clear();

        // 4 unknown describes the flow between 4 and 5 subcells
        temp.push_back(4);temp.push_back(5);
        unkown_dependence.push_back(temp); temp.clear();

        // 5 unknown describes the flow between 5 and 7 subcells
        temp.push_back(5);temp.push_back(7);
        unkown_dependence.push_back(temp); temp.clear();

        // 6 unknown describes the flow between 7 and 6 subcells
        temp.push_back(7);temp.push_back(6);
        unkown_dependence.push_back(temp); temp.clear();

        // 7 unknown describes the flow between 6 and 4 subcells
        temp.push_back(6);temp.push_back(4);
        unkown_dependence.push_back(temp); temp.clear();

        // 8 unknown describes the flow between 0 and 4 subcells
        temp.push_back(0);temp.push_back(4);
        unkown_dependence.push_back(temp); temp.clear();

        // 9 unknown describes the flow between 1 and 5 subcells
        temp.push_back(1);temp.push_back(5);
        unkown_dependence.push_back(temp); temp.clear();

        // 10 unknown describes the flow between 2 and 6 subcells
        temp.push_back(2);temp.push_back(6);
        unkown_dependence.push_back(temp); temp.clear();

        // 11 unknown describes the flow between 3 and 7 subcells
        temp.push_back(3);temp.push_back(7);
        unkown_dependence.push_back(temp); temp.clear();


    }
}

//! The order is a vector of size n equations. For 2D this is 4 and for 3D this is 12.
//! Each entry of the vector corresponds to a subcell from a set of subcells that have a common dof.
//! For example order[0] corresponds to 0 subvolume etc. order[i] is a vector of integers that show at
//! which column in the matrix A this unkonw should be. In other words the entry A(row,col) = order[row][col];
//! The sign of that entry is given b the signs vector.
template <int dim>
void unknowns_per_subcell(std::vector<std::vector<int>>& order,
                   std::vector<std::vector<int>>& signs){

    std::vector<int> t;
    std::vector<int> s;
    if (dim == 2){
        // subcell 0
        t.push_back(0); t.push_back(3);
        s.push_back(1); s.push_back(-1);
        order.push_back(t); t.clear();
        signs.push_back(s);s.clear();
        // subcell 1
        t.push_back(0); t.push_back(1);
        s.push_back(-1); s.push_back(1);
        order.push_back(t); t.clear();
        signs.push_back(s);s.clear();
        // subcell 2
        t.push_back(2); t.push_back(3);
        s.push_back(-1); s.push_back(1);
        order.push_back(t); t.clear();
        signs.push_back(s);s.clear();
        // subcell 3
        t.push_back(1); t.push_back(2);
        s.push_back(-1); s.push_back(1);
        order.push_back(t); t.clear();
        signs.push_back(s);s.clear();
    }
    else if (dim == 3){
        // subcell 0
        t.push_back(0); t.push_back(3);t.push_back(8);
        s.push_back(1); s.push_back(-1);s.push_back(1);
        order.push_back(t); t.clear();
        signs.push_back(s);s.clear();
        // subcell 1
        t.push_back(0); t.push_back(1);t.push_back(9);
        s.push_back(-1); s.push_back(1);s.push_back(1);
        order.push_back(t); t.clear();
        signs.push_back(s);s.clear();
        // subcell 2
        t.push_back(2); t.push_back(3);t.push_back(10);
        s.push_back(-1); s.push_back(1);s.push_back(1);
        order.push_back(t); t.clear();
        signs.push_back(s);s.clear();
        // subcell 3
        t.push_back(1); t.push_back(2);t.push_back(11);
        s.push_back(-1); s.push_back(1);s.push_back(1);
        order.push_back(t); t.clear();
        signs.push_back(s);s.clear();
        // subcell 4
        t.push_back(4); t.push_back(7);t.push_back(8);
        s.push_back(1); s.push_back(-1);s.push_back(-1);
        order.push_back(t); t.clear();
        signs.push_back(s);s.clear();
        // subcell 5
        t.push_back(4); t.push_back(5);t.push_back(9);
        s.push_back(-1); s.push_back(1);s.push_back(-1);
        order.push_back(t); t.clear();
        signs.push_back(s);s.clear();
        // subcell 6
        t.push_back(6); t.push_back(7);t.push_back(10);
        s.push_back(-1); s.push_back(1);s.push_back(-1);
        order.push_back(t); t.clear();
        signs.push_back(s);s.clear();
        // subcell 7
        t.push_back(5); t.push_back(6);t.push_back(11);
        s.push_back(-1); s.push_back(1);s.push_back(-1);
        order.push_back(t); t.clear();
        signs.push_back(s);s.clear();
    }
}

template <int dim>
int find_cell_relative_side(int iface, int mycellid){
    int out = -9;
    if (dim == 2){
        if (iface == 0){
            if (mycellid == 0)
                out = 3;
            if (mycellid == 2)
                out = 2;
        }
        if (iface == 1){
            if (mycellid == 1)
                out = 3;
            if (mycellid == 3)
                out = 2;
        }
        if (iface == 2){
            if (mycellid == 0)
                out = 1;
            if (mycellid == 1)
                out = 0;
        }
        if (iface == 3){
            if (mycellid == 2)
                out = 1;
            if (mycellid == 3)
                out = 0;
        }
    }
    else if (dim == 3){
        if (iface == 0){
            if (mycellid == 0)
                out = 3;
            if (mycellid == 2)
                out = 2;
        }
        if (iface == 1){
            if (mycellid == 1)
                out = 3;
            if (mycellid == 3)
                out = 2;
        }
        if (iface == 2){
            if (mycellid == 0)
                out = 1;
            if (mycellid == 1)
                out = 0;
        }
        if (iface == 3){
            if (mycellid == 2)
                out = 1;
            if (mycellid == 3)
                out = 0;
        }
        if (iface == 4){
            if (mycellid == 4)
                out = 3;
            if (mycellid == 6)
                out = 2;
        }
        if (iface == 5){
            if (mycellid == 5)
                out = 3;
            if (mycellid == 7)
                out = 2;
        }
        if (iface == 6){
            if (mycellid == 4)
                out = 1;
            if (mycellid == 5)
                out = 0;
        }
        if (iface == 7){
            if (mycellid == 6)
                out = 1;
            if (mycellid == 7)
                out = 0;
        }
        if (iface == 8){
            if (mycellid == 0)
                out = 5;
            if (mycellid == 4)
                out = 4;
        }
        if (iface == 9){
            if (mycellid == 1)
                out = 5;
            if (mycellid == 5)
                out = 4;
        }
        if (iface == 10){
            if (mycellid == 2)
                out = 5;
            if (mycellid == 6)
                out = 4;
        }
        if (iface == 11){
            if (mycellid == 3)
                out = 5;
            if (mycellid == 7)
                out = 4;
        }
    }
    return out;
}

template <int dim>
void CreateSolveUnknownFlows(unsigned dof,
                             typename std::map<unsigned int, ControlVolume<dim>> dofCV,
                             typename std::map<CellId, CellInfoRT0<dim>>& subcells){

    typename std::map<unsigned int, ControlVolume<dim>>::iterator itdof;
    typename std::map<CellId, CellInfoRT0<dim>>::iterator itsub;

    // Find the iterator for the dof
    itdof = dofCV.find(dof);

    std::vector<double> interCellflow;
    unsigned int Nsubfaces;
    if (dim == 2)
        Nsubfaces = 4;
    else if (dim == 3)
        Nsubfaces = 12;
    interCellflow.resize(Nsubfaces);

    int finer_level = itdof->second.maxLevel();

    for (unsigned int isf = 0; isf < Nsubfaces; ++isf){
        // Find the two cells that share this subface
        itsub = subcells.begin();
        std::vector<int> ids = itsub->second.getPairIDs(isf);
        // if both levels are negative then this subface doesnt exist
        if (itdof->second.getLevel(ids[0]) < 0 && itdof->second.getLevel(ids[1]) < 0)
            continue;

        if (itdof->second.getLevel(ids[0]) >= 0 && itdof->second.getLevel(ids[0]) >= 0){
            // if both levels are valid the keep on
            itsub = subcells.find(itdof->second.getCellID(ids[0]));
            CellInfoRT0<dim> subcell1 = itsub->second;
            itsub = subcells.find(itdof->second.getCellID(ids[1]));
            CellInfoRT0<dim> subcell2 = itsub->second;
            std::cout << "You are here" << std::endl;
        }
        else{
            // if one is valic find it
            int cell_side;
            if (itdof->second.getLevel(ids[0]) >= 0){
                itsub = subcells.find(itdof->second.getCellID(ids[0]));
                cell_side = find_cell_relative_side<dim>(isf, ids[0]);
            }
            else{
                itsub = subcells.find(itdof->second.getCellID(ids[1]));
                cell_side = find_cell_relative_side<dim>(isf, ids[1]);
            }
            if (itsub->second.getBCtype(isf) == 0){
                itsub->second.setcellSideFlow(isf, 0.0);
            }
            CellInfoRT0<dim> subcell = itsub->second;
            std::cout << "You are here" << std::endl;
        }

    }



    return;


    //std::vector<std::vector<double>> matrixA;
    Eigen::VectorXd matrixB;
    Eigen::MatrixXd matrixA;
    int Neq;
    if (dim == 2){
        Neq = 4;
        matrixA = Eigen::MatrixXd::Constant(5,4,0);
        matrixB = Eigen::VectorXd::Constant(5,0);
    }
    else if (dim == 3){
        matrixA = Eigen::MatrixXd::Constant(9,8,0);
        matrixB = Eigen::VectorXd::Constant(9,0);
    }

    std::vector <int> flowid;//
    std::vector<std::vector<int>> unknownspersubcell, signs, dirs;
    // The following function returns the positions and the signes of the matrix A
    unknowns_per_subcell<dim>(unknownspersubcell, signs);
    for (unsigned int ieq = 0; ieq < unknownspersubcell.size(); ++ieq){
        // if the level is negative then this subcell doesnt exist
        if (itdof->second.getLevel(ieq) >= 0){
            for (unsigned int iunkn = 0; iunkn < unknownspersubcell[ieq].size(); ++iunkn){
                matrixA(ieq, unknownspersubcell[ieq][iunkn]) = static_cast<double>(signs[ieq][iunkn]);
            }
        }
    }


    subcellFlows2Consider<dim>(flowid, dirs);

    std::vector<std::vector<int>> assoc_unkowns_subcells;
    associate_unkowns_with_sub_volumes<dim>(assoc_unkowns_subcells);

    // each subcell around the dof corresponds to an equation
    for (unsigned int icel = 0; icel < itdof->second.Ncells(); ++icel){
        //If there are cells with lower level than the maximum we should take half or 1/4 of their contributions
        // The lower level should not be lower than maxlevel - 1.
        int maxlevel = itdof->second.maxLevel();
        // If the level of the cell is negative then there is no cell at that position. Most likely the dof is on a boundary
        int ilvl = itdof->second.getLevel(icel);
        if (ilvl >= 0){
            // loop through the unknowns that are involved with the icel equation
            for (int j = 0; j < unknownspersubcell[icel].size(); ++j){
                matrixA(icel, unknownspersubcell[icel][j]) = static_cast<double>(signs[icel][j]);
            }
        }




        CellId cID = itdof->second.getCellID(icel);
        itsub = subcells.find(cID);

        double Qknown = itsub->second.outflow_from_subcell_with_vertex(flowid[icel]);
        //matrixB.push_back(Qknown);

    }




}





#endif // VELOCITY_RT0_H
