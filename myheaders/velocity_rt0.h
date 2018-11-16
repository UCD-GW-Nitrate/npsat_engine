#ifndef VELOCITY_RT0_H
#define VELOCITY_RT0_H

#include <deal.II/dofs/dof_handler.h>

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
    //! You can initialize a control volume with a dof and a
    ControlVolume(unsigned int dof_in, CellId cellid_in);
    void addCell(CellId newcell);
protected:
    //! The dof of the control volume
    unsigned int dof;

    //! A list of cell ids connected with the #dof
    std::vector<CellId> cells;

    //! A list of edges or faces that are connected with the id
};

template <int dim>
ControlVolume<dim>::ControlVolume(unsigned int dof_in, CellId cellid_in){
    dof = dof_in;
    addCell(cellid_in);
}

template <int dim>
void ControlVolume<dim>::addCell(CellId newcell){
    for (unsigned i = 0; i < cells.size(); i++){
        if (cells[i] == newcell)
            return;
    }
    cells.push_back(newcell);
}

template <int dim>
class CellInfoRT0{
public:
    CellInfoRT0();

    bool calculateSubcellFlows(typename DoFHandler<dim>::active_cell_iterator& cell,
                               FE_Q<dim>&                          fe,
                               TrilinosWrappers::MPI::Vector& 	locally_relevant_solution,
                               MyTensorFunction<dim> &HK_function);

protected:
    //! constructPW returns the mid Points positions in the unit cell and the weights.
    //! The weights are only needed to construct the quadrature formula and they are all equal summing to 1
    void constructPW(std::vector<Point<dim>>& p, std::vector<double>& w);


    //! For the time being it is assume that the hydraulic conductivity is isotropic in X and Y and that the faces are oriented in allignement
    //! with the principal axis
    void calculateFlows(std::vector<double> midHeads,
                        std::vector<Tensor<2, dim> > HK);
    void calculateSubfacesArea(std::vector<Point<dim>> vertices);

    std::vector<Tensor<2,dim>> calculateHK(std::vector<Point<dim>> vertices,MyTensorFunction<dim> &HK_function);


    std::vector<int> getPairIDs(int iface);

    //! This contains the heads in the corner of this cell.
    //! Since we are using linear elements only this corresponds to the dofs head solution
    std::vector<double> cornerHead;

    //! This contains the heads on the middle point of the subcells
    std::vector<double> midHead;


    /*!
     * \brief flowVolume is a vector that stores the flow between the subcells of the cell
     *
     * The numbering is defined as follows:
     *
     *      2D and 3D faces that touch the bottom of the cell(face 4)
     *  *---------------*
     *  |       |     3 |
     *  |  ^    3-> ^   |
     *  |  |  2 |   |   |
     *  |--0----+---1---|
     *  |       |       |
     *  |   0   2->  1  |
     *  |       |       |
     *  *---------------*
     *
     *     3D of faces that touch the top of the cell (face 5)
     *  *---------------*
     *  |       |       |
     *  |       7       |
     *  |       |       |
     *  |--4----+---5---|
     *  |       |       |
     *  |       6       |
     *  |       |       |
     *  *---------------*
     *
     *      3D of face between the top abd bottom subsells
     *  *---------------*
     *  |       |       |
     *  |  10   |  11   |
     *  |       |       |
     *  |-------+-------|
     *  |       |       |
     *  |   8   |   9   |
     *  |       |       |
     *  *---------------*
     *
     */
    std::vector<double> flowVolume;

    //! This will hold the area in 3D or the legth in 2D of the corresponding face.
    //!  This vector follows the same numbering as the #flowVolume vector
    std::vector<double> area;

    unsigned int Nsubfaces;
};

template<int dim>
CellInfoRT0<dim>::CellInfoRT0(){
    if (dim ==2)
        Nsubfaces = 4;
    else if (dim == 3)
        Nsubfaces = 12;
}

template <int dim>
bool CellInfoRT0<dim>::calculateSubcellFlows(typename DoFHandler<dim>::active_cell_iterator &cell,
                                             FE_Q<dim> &fe,
                                             TrilinosWrappers::MPI::Vector &locally_relevant_solution,
                                             MyTensorFunction<dim>& HK_function){

    std::vector<Point<dim>> quad_points;
    std::vector<double> quad_weights;
    constructPW(quad_points,quad_weights);
    Quadrature<dim> midSubcells_quadrature(quad_points,quad_weights);
    FEValues<dim> fe_values(fe, midSubcells_quadrature, update_values);
    fe_values.reinit(cell);
    std::vector<double> mid_subcell_heads(quad_points.size());
    fe_values.get_function_values(locally_relevant_solution, mid_subcell_heads);
    std::vector<Point<dim>> vertices;
    for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell; i++){
        vertices.push_back(cell->vertex(i));
    }
    calculateSubfacesArea(vertices);
    std::vector<Tensor<2, dim> > HK = calculateHK(vertices,HK_function);
    calculateFlows(mid_subcell_heads, HK);
    return true;
}

template <int dim>
void CellInfoRT0<dim>::constructPW(std::vector<Point<dim>>& p, std::vector<double>& w){
    p.clear();
    w.clear();
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
        for (unsigned int i = 0; i < 4; i++)
            w.push_back(0.25);
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
        for (unsigned int i = 0; i < 4; i++)
            w.push_back(0.125);
    }
}

template <int dim>
void CellInfoRT0<dim>::calculateFlows(std::vector<double> midHeads,
                                      std::vector<Tensor<2, dim> > HK){


    double DH, Kx, Kz, Q;
    std::vector<int> id;
    if (dim == 2){
        for (unsigned int i = 0; i < Nsubfaces; i++){
            id = getPairIDs(i);
            DH = midHeads[id[0]] - midHeads[id[1]];
            Kx = 2.0/(1.0/HK[id[0]][0][0] + 1.0/HK[id[1]][0][0]);
            Kz = 2.0/(1.0/HK[id[0]][1][1] + 1.0/HK[id[1]][1][1]);
            if (i < 2)
                Q = area[i]*Kz*DH;
            else
                Q = area[i]*Kx*DH;
            flowVolume.push_back(Q);
        }
    }
    else if (dim == 3){
        for (unsigned int i = 0; i < Nsubfaces; i++){
            id = getPairIDs(i);
            DH = midHeads[id[0]] - midHeads[id[1]];
            Kx = 2.0/(1.0/HK[id[0]][0][0] + 1.0/HK[id[1]][0][0]);
            Kz = 2.0/(1.0/HK[id[0]][2][2] + 1.0/HK[id[1]][2][2]);

            if (i < 8){
                Q = area[i]*Kx*DH;
            }
            else
                Q = area[i]*Kz*DH;
            flowVolume.push_back(Q);
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
    std::vector<Point<dim>> midEdgePoints;
    Point<dim> CellCenter;
    if (dim == 2){
        midEdgePoints.push_back(midPoint<dim>(vertices[0], vertices[2])); // 0 edge
        midEdgePoints.push_back(midPoint<dim>(vertices[1], vertices[3])); // 1 edge
        midEdgePoints.push_back(midPoint<dim>(vertices[0], vertices[1])); // 2 edge
        midEdgePoints.push_back(midPoint<dim>(vertices[2], vertices[3])); // 3 edge
        CellCenter = midPoint<dim>(midEdgePoints[0], midEdgePoints[1]);

        // In 2D the area is actual the length of a unit width
        // Subface 0
        area.push_back(midEdgePoints[0].distance(CellCenter));
        // Subface 1
        area.push_back(midEdgePoints[1].distance(CellCenter));
        // Subface 2
        area.push_back(midEdgePoints[2].distance(CellCenter));
        // Subface 3
        area.push_back(midEdgePoints[2].distance(CellCenter));
    }
    else if (dim == 3){
        // The mid edge point numbering is identical with the deal edge numbering
        // and the mid face numbering is identical to deal face numbering
        // see the GeometryInfo documentation

        std::vector<Point<dim>> midFacePoints;

        midEdgePoints.push_back(midPoint<dim>(vertices[0], vertices[2])); // 0 edge
        midEdgePoints.push_back(midPoint<dim>(vertices[1], vertices[3])); // 1 edge
        midEdgePoints.push_back(midPoint<dim>(vertices[0], vertices[1])); // 2 edge
        midEdgePoints.push_back(midPoint<dim>(vertices[2], vertices[3])); // 3 edge
        midEdgePoints.push_back(midPoint<dim>(vertices[4], vertices[6])); // 4 edge
        midEdgePoints.push_back(midPoint<dim>(vertices[5], vertices[7])); // 5 edge
        midEdgePoints.push_back(midPoint<dim>(vertices[4], vertices[5])); // 6 edge
        midEdgePoints.push_back(midPoint<dim>(vertices[6], vertices[7])); // 7 edge
        midEdgePoints.push_back(midPoint<dim>(vertices[0], vertices[4])); // 8 edge
        midEdgePoints.push_back(midPoint<dim>(vertices[1], vertices[5])); // 9 edge
        midEdgePoints.push_back(midPoint<dim>(vertices[2], vertices[6])); // 10 edge
        midEdgePoints.push_back(midPoint<dim>(vertices[3], vertices[7])); // 11 edge

        midFacePoints.push_back(midPoint<dim>(midEdgePoints[0], midEdgePoints[4])); // 0 face
        midFacePoints.push_back(midPoint<dim>(midEdgePoints[1], midEdgePoints[5])); // 1 face
        midFacePoints.push_back(midPoint<dim>(midEdgePoints[2], midEdgePoints[6])); // 2 face
        midFacePoints.push_back(midPoint<dim>(midEdgePoints[3], midEdgePoints[7])); // 3 face
        midFacePoints.push_back(midPoint<dim>(midEdgePoints[0], midEdgePoints[1])); // 4 face
        midFacePoints.push_back(midPoint<dim>(midEdgePoints[4], midEdgePoints[5])); // 5 face

        CellCenter = midPoint<dim>(midEdgePoints[0], midEdgePoints[1]);

        double ar1, ar2;
        // Subface 0
        ar1 = triangle_area<dim>(midEdgePoints[0], midFacePoints[4], CellCenter,false);
        ar2 = triangle_area<dim>(midEdgePoints[0], CellCenter, midFacePoints[0],false);
        area.push_back(ar1 + ar2);
        // Subface 1
        ar1 = triangle_area<dim>(midEdgePoints[1], midFacePoints[4], CellCenter,false);
        ar2 = triangle_area<dim>(midEdgePoints[1], CellCenter, midFacePoints[1],false);
        area.push_back(ar1 + ar2);
        // Subface 2
        ar1 = triangle_area<dim>(midEdgePoints[2], midFacePoints[4], CellCenter,false);
        ar2 = triangle_area<dim>(midEdgePoints[2], CellCenter, midFacePoints[2],false);
        area.push_back(ar1 + ar2);
        // Subface 3
        ar1 = triangle_area<dim>(midEdgePoints[3], midFacePoints[4], CellCenter,false);
        ar2 = triangle_area<dim>(midEdgePoints[3], CellCenter, midFacePoints[3],false);
        area.push_back(ar1 + ar2);
        // Subface 4
        ar1 = triangle_area<dim>(midEdgePoints[4], midFacePoints[0], CellCenter,false);
        ar2 = triangle_area<dim>(midEdgePoints[4], CellCenter, midFacePoints[5],false);
        area.push_back(ar1 + ar2);
        // Subface 5
        ar1 = triangle_area<dim>(midEdgePoints[5], midFacePoints[1], CellCenter,false);
        ar2 = triangle_area<dim>(midEdgePoints[5], CellCenter, midFacePoints[5],false);
        area.push_back(ar1 + ar2);
        // Subface 6
        ar1 = triangle_area<dim>(midEdgePoints[6], midFacePoints[2], CellCenter,false);
        ar2 = triangle_area<dim>(midEdgePoints[6], CellCenter, midFacePoints[5],false);
        area.push_back(ar1 + ar2);
        // Subface 7
        ar1 = triangle_area<dim>(midEdgePoints[7], midFacePoints[3], CellCenter,false);
        ar2 = triangle_area<dim>(midEdgePoints[7], CellCenter, midFacePoints[5],false);
        area.push_back(ar1 + ar2);
        // Subface 8
        ar1 = triangle_area<dim>(midEdgePoints[8], midFacePoints[0], CellCenter,false);
        ar2 = triangle_area<dim>(midEdgePoints[8], CellCenter, midFacePoints[2],false);
        area.push_back(ar1 + ar2);
        // Subface 9
        ar1 = triangle_area<dim>(midEdgePoints[9], midFacePoints[1], CellCenter,false);
        ar2 = triangle_area<dim>(midEdgePoints[9], CellCenter, midFacePoints[2],false);
        area.push_back(ar1 + ar2);
        // Subface 10
        ar1 = triangle_area<dim>(midEdgePoints[10], midFacePoints[0], CellCenter,false);
        ar2 = triangle_area<dim>(midEdgePoints[10], CellCenter, midFacePoints[3],false);
        area.push_back(ar1 + ar2);
        // Subface 11
        ar1 = triangle_area<dim>(midEdgePoints[11], midFacePoints[1], CellCenter,false);
        ar2 = triangle_area<dim>(midEdgePoints[11], CellCenter, midFacePoints[3],false);
        area.push_back(ar1 + ar2);
    }
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




#endif // VELOCITY_RT0_H
