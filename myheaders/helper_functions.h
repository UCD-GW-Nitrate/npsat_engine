#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

#include <vector>
#include <math.h>
#include <string>
#include <ctime>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

/*!
 * \brief linspace generate a linearly spaced vector between the two numbers min and max
 * \param min is the lower end
 * \param max is the upper end
 * \param n is the number of numbers to generate between the min and max
 * \return a vector of n numbers linearly spaced
 */
std::vector<double> linspace(double min, double max, int n){
    std::vector<double> result;
    int iterator = 0;
    for (int i = 0; i <= n-2; i++){
        double temp = min + i*(max-min)/(floor(static_cast<double>(n)) - 1);
        result.insert(result.begin() + iterator, temp);
        iterator += 1;
    }
    result.insert(result.begin() + iterator, max);
    return result;
}

template <int dim>
void print_cell_coords(typename dealii::DoFHandler<dim>::active_cell_iterator cell){
    std::vector<dealii::Point<dim>> verts;
    for (unsigned int vertex_no = 0; vertex_no < dealii::GeometryInfo<dim>::vertices_per_cell; ++vertex_no){
        verts.push_back(cell->vertex(vertex_no));
    }

    std::vector<unsigned int> ind;
    if (dim == 2){
        ind.push_back(0);
        ind.push_back(1);
        ind.push_back(3);
        ind.push_back(2);
        std::cout << "[" << std::setprecision(10);
        for (unsigned int ii = 0; ii < ind.size(); ++ii)
            std::cout << verts[ind[ii]][0] << " " << verts[ind[ii]][1] << "; ";
        std::cout << "]" << std::endl;
    }
    if (dim == 3){
        ind.push_back(0);
        ind.push_back(1);
        ind.push_back(3);
        ind.push_back(2);
        ind.push_back(4);
        ind.push_back(5);
        ind.push_back(7);
        ind.push_back(6);
        std::cout << "[" << std::setprecision(10);
        for (unsigned int ii = 0; ii < ind.size(); ++ii)
            std::cout << verts[ind[ii]][0] << " " << verts[ind[ii]][1] << " " << verts[ind[ii]][2] << "; ";
        std::cout << "]" << std::endl;
    }
}


template <int dim>
bool try_mapping(dealii::Point<dim>& p, dealii::Point<dim>& punit,
                 typename dealii::DoFHandler<dim>::active_cell_iterator cell, dealii::MappingQ1<dim> mapping){
    bool mapping_done = false;
    int count_try = 0;
    dealii::Point<dim> p_try = p;
    while (!mapping_done){
        try {
            punit = mapping.transform_real_to_unit_cell(cell, p_try);
            mapping_done = true;
        } catch (...) {
            for (unsigned int idim = 0; idim < dim; ++idim)
                p_try[idim] = p[idim] + 0.0001*(-1.0 + 2.0*(static_cast<double>(rand())/static_cast<double>(RAND_MAX)));
            ++count_try;
            if (count_try > 20){
                std::cerr << "Transformation Failed for DofHandler cell " << cell->is_locally_owned() << std::endl;
                print_cell_coords<dim>(cell);
                std::cout << p << std::endl;
                break;
            }
        }
    }
    return mapping_done;
}

template <int dim>
bool try_mapping(dealii::Point<dim> p, dealii::Point<dim> &p_unit,
                 typename dealii::Triangulation<dim>::active_cell_iterator cell, dealii::MappingQ1<dim> mapping){
    bool mapping_done = false;
    int count_try = 0;
    dealii::Point<dim> p_try = p;
    while (!mapping_done){
        try{
            p_unit = mapping.transform_real_to_unit_cell(cell, p_try);
            mapping_done = true;
        }
        catch(...){
            for (unsigned int idim = 0; idim < dim; ++idim)
                p_try[idim] = p[idim] + 0.0001*(-1.0 + 2.0*(double(rand())/double(RAND_MAX)));
            ++count_try;
            if (count_try > 20){
                break;
                std::cerr << "transformation Failed for Triangulation cell" << std::endl;
            }

        }
    }
    return mapping_done;
}

double distance_2_points(double px1, double py1, double px2, double py2){
    return std::sqrt((px1-px2)*(px1-px2)+(py1-py2)*(py1-py2));
}

/*!
 * \brief distance_on_2D_line projects the point (px, py) onto line and returns the distance from the first point
 * \param l1x
 * \param l1y
 * \param l2x
 * \param l2y
 * \param px
 * \param py
 * \return
 */
double distance_on_2D_line(double l1x, double l1y, double l2x, double l2y, double px, double py){
    double e1x = l2x - l1x;
    double e1y = l2y - l1y;
    double len2 = e1x*e1x + e1y*e1y;
    double e2x = px - l1x;
    double e2y = py - l1y;
    double dot = e1x*e2x + e1y*e2y;
    double ppx = l1x + dot*e1x/len2;
    double ppy = l1y + dot*e1y/len2;
    double dst = distance_2_points(ppx, ppy, l1x, l1y);
    //double dst = sqrt((ppx-l1x)*(ppx-l1x)+(ppy-l1y)*(ppy-l1y));
    double t;
    if (std::abs(e1x) >= std::abs(e1y)){
        t = (ppx - l1x)/(l2x-l1x);
    }
    else{
        t = (ppy - l1y)/(l2y-l1y);
    }
    if (t<0)
        return -dst;
    else
        return dst;
}

//! distance_point_line calculates the distance between a point and a line segment.
//! If the projection of the point to the line falls outside the line segment then
//! returns the minimum distance from the the two points
double distance_point_line(double px, double py, // Point coordinates
                           double l1x, double l1y, // first point of line
                           double l2x, double l2y){ // second point of line

    double e1x = l2x - l1x;
    double e1y = l2y - l1y;
    double len2 = e1x*e1x + e1y*e1y;
    double e2x = px - l1x;
    double e2y = py - l1y;
    double dot = e1x*e2x + e1y*e2y;
    double ppx = l1x + dot*e1x/len2;
    double ppy = l1y + dot*e1y/len2;
    double t;
    if (std::abs(e1x) > std::abs(e1y))
        t = (ppx - l1x)/(l2x-l1x);
    else
        t = (ppy - l1y)/(l2y-l1y);

    double dst;
    if (t <0 || t > 1){
        dst = std::min(distance_2_points(px,py,l1x,l1y), distance_2_points(px,py,l1x,l1y));
    }
    else{
        dst = distance_2_points(ppx, ppy, px, py);
    }
    return dst;
}

/*!
 * \brief is_input_a_scalar check if the string can be converted into a scalar value
 * \param input is the string to test
 * \return true if the input can be a scalars
 */
bool is_input_a_scalar(std::string input){
    // try to convert the input to scalar
    bool outcome;
    try{
        double value = stod(input);
        value++; // something to surpress the warning
        outcome = true;
    }
    catch(...){
        outcome = false;
    }
    return outcome;
}

/*!
 * \brief line_line_intersection
 * \param b1 intercept of 1st line
 * \param m1 slope of 1st line
 * \param b2 intercept of 2nd line
 * \param m2 slope of 2nd line
 * \param x coordinate of intersection point
 * \param y coordinate of intersection point
 * \return
 */
bool line_line_intersection(double b1, double m1, double b2, double m2,
                            double &x, double &y){
    if (abs(m1 - m2) < 0.0001){
        return false;
    }
    else{
        x = (b2 - b1)/(m1 - m2);
        y = m1*x + b1;
        return true;
    }
}


/*!
 * \brief triangle_area Calculates the area of a triangle defined by the three vertices
 * \param A 1st Point of triangle
 * \param B 2nd Point of triangle
 * \param C 3rd Point of triangle
 * \param project if true this will calculate the area of the triangle when projected in the XY plane
 * \return the area of the triangle
 */
template <int dim>
double triangle_area(dealii::Point<dim> A, dealii::Point<dim> B, dealii::Point<dim> C, bool project){
    if (dim == 2)
        std::cerr << "You can use triangle_area in other than dim == 3" << std::endl;

    double area = 0;
    double x1 = A[0]; double y1 = A[1]; double z1 = A[2];
    double x2 = B[0]; double y2 = B[1]; double z2 = B[2];
    double x3 = C[0]; double y3 = C[1]; double z3 = C[2];
    if (project){
        area = pow(x1*y2 - x2*y1 - x1*y3 + x3*y1 + x2*y3 - x3*y2, 2);
        // projected area?
        //area = std::abs( 0.5*(A[0]*(B[1] - C[1]) + B[0]*(C[1] - A[1]) + C[0]*(A[1] - B[1])));
    }
    else{
        //http://mathworld.wolfram.com/TriangleArea.html
        area = pow(x1*y2 - x2*y1 - x1*y3 + x3*y1 + x2*y3 - x3*y2, 2) +
               pow(x1*z2 - x2*z1 - x1*z3 + x3*z1 + x2*z3 - x3*z2, 2) +
               pow(y1*z2 - y2*z1 - y1*z3 + y3*z1 + y2*z3 - y3*z2, 2);
    }
    area = 0.5*sqrt(area);

    return area;
}


/*!
 * \brief recharge_weight calculates a weight factor which is used during recharge calculations.
 * To calculate the correct amount of groundwater recharge (e.g. from precipitation), the recharge rate
 * should be multiplied by the projected area on the xy plane of the top face of the element.
 * \param cell is the element where the recharge is applied
 * \param face is the face id
 * \return the weight
 */
template<int dim>
double recharge_weight(typename dealii::DoFHandler<dim>::active_cell_iterator cell, unsigned int face){
    double weight = 1;
    if (dim == 2){
        dealii::Point<dim> v1 = cell->face(face)->vertex(0);
        dealii::Point<dim> v2 = cell->face(face)->vertex(1);
        double actual_length = v1.distance(v2);
        double projected_length = std::abs(v2[0] - v1[0]);
        weight = projected_length/actual_length;
    }else if (dim == 3){
        dealii::Point<dim> v1 = cell->face(face)->vertex(0);
        dealii::Point<dim> v2 = cell->face(face)->vertex(1);
        dealii::Point<dim> v3 = cell->face(face)->vertex(2);
        dealii::Point<dim> v4 = cell->face(face)->vertex(3);
        double A_real = triangle_area(v1,v2,v4,false) + triangle_area(v1,v4,v3,false);
        double A_proj = triangle_area(v1,v2,v4,true)  + triangle_area(v1,v4,v3,true);
        weight = A_proj/A_real;
    }
    if (weight > 1)
        std::cerr << "Projected area is higher than real??" << std::endl;
    return weight;
}


template<int dim>
int is_point_in_list(dealii::Point<dim>& temp_point, std::vector<dealii::Point<dim> >& list, double tol){
    int out = -1;
    if (list.size() > 0){
        for (unsigned int i = 0; i < list.size(); i++){
            if (temp_point.distance(list[i]) < tol){
                out = static_cast<int>(i);
                break;
            }
        }
    }
    return out;
}

/*!
 * \brief This function returns a list of indices that are connected to the #ii node in a cell
 * \param #ii is the index of the node we seek its connected nodes
 * \return a vector of connected indices. Note that this is simply returns the ids of the nodes
 * as they are defined by the geometry info class. As we are interested only in the connections
 * along the vertical direction we return only those. However inside the function there is boolean flag
 * that can be hardcoded to return all connections.
 */
template <int dim>
std::vector<int> get_connected_indices(int ii){
    bool return_all = false;
    std::vector<int> out;
    if (dim == 2){
        if (return_all){
            if (ii == 0){
                out.push_back(1);
                out.push_back(2);
            }else if (ii == 1){
                out.push_back(0);
                out.push_back(3);
            }else if (ii == 2){
                out.push_back(0);
                out.push_back(3);
            }else if (ii == 3){
                out.push_back(1);
                out.push_back(2);
            }
        }
        else{
            if (ii == 0){
                out.push_back(2);
            }else if (ii == 1){
                out.push_back(3);
            }else if (ii == 2){
                out.push_back(0);
            }else if (ii == 3){
                out.push_back(1);
            }

        }
    }else if (dim == 3){
        if (return_all){
            if (ii == 0){
                out.push_back(1);
                out.push_back(2);
                out.push_back(4);
            }else if (ii == 1){
                out.push_back(0);
                out.push_back(3);
                out.push_back(5);
            }else if (ii == 2){
                out.push_back(0);
                out.push_back(3);
                out.push_back(6);
            }else if (ii == 3){
                out.push_back(1);
                out.push_back(2);
                out.push_back(7);
            }else if (ii == 4){
                out.push_back(0);
                out.push_back(5);
                out.push_back(6);
            }else if (ii == 5){
                out.push_back(1);
                out.push_back(4);
                out.push_back(7);
            }else if (ii == 6){
                out.push_back(2);
                out.push_back(4);
                out.push_back(7);
            }else if (ii == 7){
                out.push_back(3);
                out.push_back(5);
                out.push_back(6);
            }
        }
        else{
            if (ii == 0){
                out.push_back(4);
            }else if (ii == 1){
                out.push_back(5);
            }else if (ii == 2){
                out.push_back(6);
            }else if (ii == 3){
                out.push_back(7);
            }else if (ii == 4){
                out.push_back(0);
            }else if (ii == 5){
                out.push_back(1);
            }else if (ii == 6){
                out.push_back(2);
            }else if (ii == 7){
                out.push_back(3);
            }
        }
    }
    return out;
}

template <int dim>
std::vector<int> facesIdonVertex(int ivertex){
    std::vector<int> out;
    if (dim == 2){
        if (ivertex == 0){
            out.push_back(0);
            out.push_back(2);
        }
        else if (ivertex == 1){
            out.push_back(1);
            out.push_back(2);
        }
        else if (ivertex == 2){
            out.push_back(0);
            out.push_back(3);
        }
        else if (ivertex == 3){
            out.push_back(1);
            out.push_back(3);
        }
    }
    else if (dim == 3){
        if (ivertex == 0){
            out.push_back(0);
            out.push_back(2);
            out.push_back(4);
        }
        else if (ivertex == 1){
            out.push_back(1);
            out.push_back(2);
            out.push_back(4);
        }
        else if (ivertex == 2){
            out.push_back(0);
            out.push_back(3);
            out.push_back(4);
        }
        else if (ivertex == 3){
            out.push_back(1);
            out.push_back(3);
            out.push_back(4);
        }
        else if (ivertex == 4){
            out.push_back(0);
            out.push_back(2);
            out.push_back(5);
        }
        else if (ivertex == 5){
            out.push_back(1);
            out.push_back(2);
            out.push_back(5);
        }
        else if (ivertex == 6){
            out.push_back(0);
            out.push_back(3);
            out.push_back(5);
        }
        else if (ivertex == 7){
            out.push_back(1);
            out.push_back(3);
            out.push_back(5);
        }
    }
    return out;
}

//! Returns the edge index of a cell that has a neibor of higher level whose ivertex is constraint and the iface touch the cell.
template <int dim>
int edge_index_of_level_1_cell(int ivertex, int iface){
    if (dim == 2){
        if (ivertex == 0){
            if (iface == 0)
                return 1;
            else if (iface == 2)
                return 3;
        }
        else if (ivertex == 0){
            if (iface == 1)
                return 0;
            else if (iface == 2)
                return 3;
        }
        else if (ivertex == 2){
            if (iface == 3)
                return 2;
            else if (iface == 0)
                return 1;
        }
        else if (ivertex == 3){
            if (iface == 3)
                return 2;
            else if (iface == 1)
                return 0;
        }
    }
    else if (dim == 3){
        if (ivertex == 0){
            if (iface == 4)
                return -9;
            else if (iface == 1)
                return 0;
            else if (iface == 0)
                return 0;
        }
    }

}


template <int dim>
dealii::Point<dim> midPoint(dealii::Point<dim> A, dealii::Point<dim> B){
    dealii::Point<dim> AB;
    for (unsigned int i = 0; i < dim; i++)
        AB[i] = (A[i] + B[i])/2.0;
    return AB;
}

template <int dim>
void initTria(dealii::Triangulation<dim-1> &tria){
    std::vector< dealii::Point<dim-1> > vertices(dealii::GeometryInfo<dim-1>::vertices_per_cell);
    std::vector< dealii::CellData<dim-1> > cells(1);

    if (dim == 2){
        vertices[0] = dealii::Point<dim-1>(0);
        vertices[1] = dealii::Point<dim-1>(1);
        cells[0].vertices[0] = 0;
        cells[0].vertices[1] = 1;
    }else if (dim == 3){
        vertices[0] = dealii::Point<dim-1>(0,0);
        vertices[1] = dealii::Point<dim-1>(1,0);
        vertices[2] = dealii::Point<dim-1>(0,1);
        vertices[3] = dealii::Point<dim-1>(1,1);
        cells[0].vertices[0] = 0;
        cells[0].vertices[1] = 1;
        cells[0].vertices[2] = 2;
        cells[0].vertices[3] = 3;
    }
    tria.create_triangulation(vertices, cells, dealii::SubCellData());


}

void print_poly_matlab(std::vector<double> xp, std::vector<double> yp){
    std::cout << "plot([";
    for (unsigned int kk = 0; kk < xp.size(); ++kk)
        std::cout << xp[kk] << " ";
    std::cout << xp[0] << "],[";
    for (unsigned int kk = 0; kk < yp.size(); ++kk)
        std::cout << yp[kk] << " ";
    std::cout << yp[0] << "])" << std::endl;
}

template <int dim>
void print_cell_face_matlab(typename dealii::Triangulation<dim>::active_cell_iterator cell, int iface){
    if (dim == 3){
        std::vector<int> id;
        id.push_back(0);
        id.push_back(1);
        id.push_back(3);
        id.push_back(2);
        std::vector<double> xp;
        std::vector<double> yp;
        std::vector<double> zp;
        for (unsigned int i = 0; i < 4; ++i){
            xp.push_back(cell->face(iface)->vertex(i)[0]);
            yp.push_back(cell->face(iface)->vertex(i)[1]);
            zp.push_back(cell->face(iface)->vertex(i)[2]);
        }

        std::cout << "plot3([";
        for (unsigned int i = 0; i < 4; ++i)
            std::cout << xp[id[i]] << " ";
        std::cout << xp[0] << "]";

        std::cout << ",[";
        for (unsigned int i = 0; i < 4; ++i)
            std::cout << yp[id[i]] << ",";
        std::cout << yp[0] << "]";

        std::cout << ",[";
        for (unsigned int i = 0; i < 4; ++i)
            std::cout << zp[id[i]] << ",";
        std::cout << zp[0] << "])" << std::endl;;

    }
}

void dummy_function(bool in, int &ii){
    ii = 1;
    if (in){
        std::cout << "Input is true" << std::endl;
        ii = ii + 1;
    }
    else{
        std::cout << "Input is false" << std::endl;
        ii = ii - 1;
    }
}

std::string print_current_time(){
    std::time_t t = std::time(nullptr);
    std::stringstream buffer;
    buffer << std::put_time(std::localtime(&t), "%c %Z");
    //std::string out =
    //std::tm* now = std::localtime(&t);
    //std::string out = "Simulation Started:";
    //std::cout << (now->tm_year + 1900) << "/" << (now->tm_mon+1) << "/" << now->tm_mday << std::endl;
    return buffer.str();;

}
void append_slash(std::string &s){
    char lc = s.back();
    if (lc != '/' )
        s = s+ "/";
}

//! If there is a finer cell that its vertexth vertex touches the faceth face of a coarser cell at the subfaceth subface
//! then this method returns the face or edge id of the coarser cell that the vertexth vertex touches the coarser.
//! If the verteth vertex is also a vertex of this cell then this method returns nothing.
//! REQUIRES HEAVY DEBUGING
template <int dim>
int is_the_vertex_on_faceEdge(int vertex,int face, int subface, bool& onface){
    int out = -9;
    if (dim == 2){
        if (vertex == 0){
            if (face == 1 && subface == 1)
                out = 1;
            if (face == 3 && subface == 1)
                out = 3;
        }
       else if (vertex == 1){
            if (face == 0 && subface == 1)
                out = 0;
            if (face == 3 && subface == 0)
                out = 3;
        }
        else if (vertex == 2){
             if (face == 1 && subface == 0)
                 out = 1;
             if (face == 2 && subface == 1)
                 out = 2;
         }
        else if (vertex == 3){
             if (face == 0 && subface == 0)
                 out = 0;
             if (face == 2 && subface == 0)
                 out = 2;
         }
        if (out >=0)
            onface = true;
    }
    else if (dim == 3){
        if (vertex == 0){
            if(face == 1){
                if (subface == 3){
                    out = 1;
                    onface = true;
                }
                else if (subface == 1){
                    out = 1;
                    onface = false;
                }
                else if (subface == 2){
                    out = 9;
                    onface = false;
                }
            }
            else if (face == 3){
                if (subface == 1){
                       out = 3;
                       onface = false;
                }
                else if (subface == 2){
                    out = 10;
                    onface = false;
                }
                else if (subface == 3){
                    out = 3;
                    onface = true;
                }
            }
            else if (face == 5){
                if (subface == 1){
                       out = 6;
                       onface = false;
                }
                else if (subface == 2){
                    out = 4;
                    onface = false;
                }
                else if (subface == 3){
                    out = 5;
                    onface = true;
                }
            }
        }
        else if (vertex == 1){
            if(face == 0){
                if (subface == 1){
                    out = 0;
                    onface = false;
                }
                else if (subface == 2){
                    out = 8;
                    onface = false;
                }
                else if (subface == 3){
                    out = 0;
                    onface = true;
                }
            }
            else if (face == 3){
                if (subface == 0){
                       out = 3;
                       onface = false;
                }
                else if (subface == 2){
                    out = 3;
                    onface = true;
                }
                else if (subface == 3){
                    out = 11;
                    onface = false;
                }
            }
            else if (face == 5){
                if (subface == 0){
                       out = 6;
                       onface = false;
                }
                else if (subface == 2){
                    out = 5;
                    onface = true;
                }
                else if (subface == 3){
                    out = 5;
                    onface = false;
                }
            }
        }
        else if (vertex == 2){
            if(face == 1){
                if (subface == 0){
                    out = 1;
                    onface = false;
                }
                else if (subface == 2){
                    out = 1;
                    onface = true;
                }
                else if (subface == 3){
                    out = 11;
                    onface = false;
                }
            }
            else if (face == 2){
                if (subface == 1){
                       out = 2;
                       onface = false;
                }
                else if (subface == 2){
                    out = 8;
                    onface = false;
                }
                else if (subface == 3){
                    out = 2;
                    onface = true;
                }
            }
            else if (face == 5){
                if (subface == 0){
                       out = 4;
                       onface = false;
                }
                else if (subface == 1){
                    out = 5;
                    onface = true;
                }
                else if (subface == 3){
                    out = 7;
                    onface = false;
                }
            }
        }
        else if (vertex == 3){
            if(face == 0){
                if (subface == 0){
                    out = 0;
                    onface = false;
                }
                else if (subface == 2){
                    out = 0;
                    onface = true;
                }
                else if (subface == 3){
                    out = 10;
                    onface = false;
                }
            }
            else if (face == 2){
                if (subface == 0){
                       out = 2;
                       onface = false;
                }
                else if (subface == 2){
                    out = 2;
                    onface = true;
                }
                else if (subface == 3){
                    out = 9;
                    onface = false;
                }
            }
            else if (face == 5){
                if (subface == 0){
                       out = 3;
                       onface = true;
                }
                else if (subface == 1){
                    out = 5;
                    onface = true;
                }
                else if (subface == 2){
                    out = 7;
                    onface = false;
                }
            }
        }
        else if (vertex == 4){
            if(face == 1){
                if (subface == 0){
                    out = 9;
                    onface = false;
                }
                else if (subface == 1){
                    out = 1;
                    onface = true;
                }
                else if (subface == 3){
                    out = 5;
                    onface = false;
                }
            }
            else if (face == 3){
                if (subface == 0){
                       out = 10;
                       onface = false;
                }
                else if (subface == 1){
                    out = 3;
                    onface = true;
                }
                else if (subface == 3){
                    out = 7;
                    onface = false;
                }
            }
            else if (face == 4){
                if (subface == 1){
                       out = 2;
                       onface = false;
                }
                else if (subface == 2){
                    out = 0;
                    onface = false;
                }
                else if (subface == 3){
                    out = 4;
                    onface = true;
                }
            }
        }
        else if (vertex == 5){
            if(face == 0){
                if (subface == 0){
                    out = 8;
                    onface = false;
                }
                else if (subface == 1){
                    out = 0;
                    onface = true;
                }
                else if (subface == 3){
                    out = 4;
                    onface = false;
                }
            }
            else if (face == 3){
                if (subface == 0){
                       out = 3;
                       onface = true;
                }
                else if (subface == 1){
                    out = 11;
                    onface = false;
                }
                else if (subface == 2){
                    out = 7;
                    onface = false;
                }
            }
            else if (face == 4){
                if (subface == 0){
                       out = 2;
                       onface = false;
                }
                else if (subface == 2){
                    out = 4;
                    onface = true;
                }
                else if (subface == 3){
                    out = 1;
                    onface = false;
                }
            }
        }
        else if (vertex == 6){
            if(face == 1){
                if (subface == 0){
                    out = 1;
                    onface = true;
                }
                else if (subface == 1){
                    out = 11;
                    onface = false;
                }
                else if (subface == 2){
                    out = 5;
                    onface = false;
                }
            }
            else if (face == 2){
                if (subface == 0){
                       out = 8;
                       onface = false;
                }
                else if (subface == 1){
                    out = 2;
                    onface = true;
                }
                else if (subface == 3){
                    out = 6;
                    onface = false;
                }
            }
            else if (face == 4){
                if (subface == 0){
                       out = 0;
                       onface = false;
                }
                else if (subface == 2){
                    out = 4;
                    onface = true;
                }
                else if (subface == 3){
                    out = 3;
                    onface = false;
                }
            }
        }
        else if (vertex == 7){
            if(face == 0){
                if (subface == 0){
                    out = 0;
                    onface = true;
                }
                else if (subface == 1){
                    out = 10;
                    onface = false;
                }
                else if (subface == 2){
                    out = 4;
                    onface = false;
                }
            }
            else if (face == 2){
                if (subface == 0){
                       out = 2;
                       onface = true;
                }
                else if (subface == 1){
                    out = 9;
                    onface = false;
                }
                else if (subface == 2){
                    out = 6;
                    onface = false;
                }
            }
            else if (face == 4){
                if (subface == 0){
                       out = 4;
                       onface = true;
                }
                else if (subface == 1){
                    out = 1;
                    onface = false;
                }
                else if (subface == 2){
                    out = 3;
                    onface = false;
                }
            }
        }
    }
    return out;
}

/*
template <int dim>
void Print_Mesh_DofHandler(std::string filename,
                           unsigned int my_rank,
                           dealii::DoFHandler<dim>& mesh_dof_handler,
                           dealii::FESystem<dim>& mesh_fe){

    std::map<int,std::pair<int,dealii::Point<dim>>> Points; // <dof> - <counter - Point>
    std::map<int,std::vector<int> > Mesh;

    typename std::map<int,std::pair<int,dealii::Point<dim>>>::iterator itp;

    int p_cnt = 0; // Point counter
    int m_cnt = 0; // Mesh counter

    const dealii::MappingQ1<dim> mapping;
    const std::vector<dealii::Point<dim> > mesh_support_points
                                  = mesh_fe.base_element(0).get_unit_support_points();
    dealii::FEValues<dim> fe_mesh_points (mapping,
                                  mesh_fe,
                                  mesh_support_points,
                                  update_quadrature_points);

    std::vector<unsigned int> cell_dof_indices (mesh_fe.dofs_per_cell);
    typename dealii::DoFHandler<dim>::active_cell_iterator
    cell = mesh_dof_handler.begin_active(),
    endc = mesh_dof_handler.end();
    for (; cell != endc; ++cell){
        if (cell->is_locally_owned()){
            fe_mesh_points.reinit(cell);
            cell->get_dof_indices (cell_dof_indices);
            for (unsigned int idof = 0; idof < mesh_fe.base_element(0).dofs_per_cell; ++idof){
                dealii::Point <dim> current_node;
                std::vector<int> current_dofs(dim);
                for (unsigned int dir = 0; dir < dim; ++dir){
                    unsigned int support_point_index = mesh_fe.component_to_system_index(dir, idof );
                    current_dofs[dir] = static_cast<int>(cell_dof_indices[support_point_index]);
                    current_node[dir] = fe_mesh_points.quadrature_point(idof)[dir];
                }
                itp = Points.find(current_dofs[dim-1]);
                if (itp == Points.end()){
                    Points.insert(std::pair<int, std::pair<int,dealii::Point<dim>>>(current_dofs[dim-1], std::pair<int,Point<dim>>(p_cnt,current_node)));
                    p_cnt++;
                }
            }
        }
    }

}
*/
#endif // HELPER_FUNCTIONS_H
