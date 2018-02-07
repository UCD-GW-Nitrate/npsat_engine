#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

#include <vector>
#include <math.h>
#include <string>

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
void Print_Mesh_DofHandler(std::string filename,
                           unsigned int my_rank,
                           DoFHandler<dim>& mesh_dof_handler,
                           FESystem<dim>& mesh_fe){

    std::map<int,std::pair<int,Point<dim>>> Points; // <dof> - <counter - Point>
    std::map<int,std::vector<int> > Mesh;

    typename std::map<int,std::pair<int,Point<dim>>>::iterator itp;

    int p_cnt = 0; // Point counter
    int m_cnt = 0; // Mesh counter

    const MappingQ1<dim> mapping;
    const std::vector<Point<dim> > mesh_support_points
                                  = mesh_fe.base_element(0).get_unit_support_points();
    FEValues<dim> fe_mesh_points (mapping,
                                  mesh_fe,
                                  mesh_support_points,
                                  update_quadrature_points);

    std::vector<unsigned int> cell_dof_indices (mesh_fe.dofs_per_cell);
    typename DoFHandler<dim>::active_cell_iterator
    cell = mesh_dof_handler.begin_active(),
    endc = mesh_dof_handler.end();
    for (; cell != endc; ++cell){
        if (cell->is_locally_owned()){
            fe_mesh_points.reinit(cell);
            cell->get_dof_indices (cell_dof_indices);
            for (unsigned int idof = 0; idof < mesh_fe.base_element(0).dofs_per_cell; ++idof){
                Point <dim> current_node;
                std::vector<int> current_dofs(dim);
                for (unsigned int dir = 0; dir < dim; ++dir){
                    unsigned int support_point_index = mesh_fe.component_to_system_index(dir, idof );
                    current_dofs[dir] = static_cast<int>(cell_dof_indices[support_point_index]);
                    current_node[dir] = fe_mesh_points.quadrature_point(idof)[dir];
                }
                itp = Points.find(current_dofs[dim-1]);
                if (itp == Points.end()){
                    Points.insert(std::pair<int, std::pair<int,Point<dim>>>(current_dofs[dim-1], std::pair<int,Point<dim>>(p_cnt,current_node)));
                    p_cnt++;
                }
            }
        }
    }

}

#endif // HELPER_FUNCTIONS_H
