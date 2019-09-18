#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <iostream>
#include <fstream>

#include "../myheaders/interpinterface.h"

using namespace dealii;

/*!
 * \brief distribute_numbers distributes equally spaced N numbers between a and b
 * and returns them int a vector.
 * \param a
 * \param b
 * \param N
 * \return
 */
std::vector<double> distribute_numbers(double a, double b, unsigned int N){
    if (a > b){
        double bb = b;
        b = a;
        a = bb;
    }

    double len = b - a;
    double dst = len/static_cast<double>(N+1);
    std::vector<double> p;
    for (unsigned int i = 0; i < N; ++i){
        p.push_back(a+(static_cast<double>(i)+1)*dst);
    }
    return p;
}

void create_logo(){
    Triangulation<2> triangulation;
    std::vector<unsigned int> repetitions;
    repetitions.push_back(20);
    repetitions.push_back(8);
    Point<2> p1(1,1);
    Point<2> p2(400,161);
    GridGenerator::subdivided_hyper_rectangle(triangulation,repetitions,p1,p2,false);

    InterpInterface<2> img;
    img.get_data("logo_data.npsat");

    // Define how many points will be sampled for each cell
    const unsigned int N = 4;

    for (unsigned int step=0; step<6; ++step){
        std::cout << "Step: " << step << std::endl;
        Triangulation<2>::active_cell_iterator cell = triangulation.begin_active();
        Triangulation<2>::active_cell_iterator endc = triangulation.end();
        for (; cell!=endc; ++cell){
            double xa, xb, ya, yb;
            xa = cell->vertex(0)[0];
            ya = cell->vertex(0)[1];
            xb = cell->vertex(3)[0];
            yb = cell->vertex(3)[1];
            std::vector<double> xx = distribute_numbers(xa,xb,N);
            std::vector<double> yy = distribute_numbers(ya,yb,N);
            bool set_this_cell = false;
            unsigned int n = 0;
            for (unsigned int i = 0; i < yy.size(); ++i){
                for (unsigned int j = 0; j < xx.size(); ++j){
                    double v = img.interpolate(Point<2>(xx[j], yy[i]));
                    if (v > 0.1 && v < 254){
                        n++;
                        if (n > 1){
                            cell->set_refine_flag ();
                            set_this_cell = true;
                            break;
                        }
                    }
                }
                if (set_this_cell)
                    break;
            }


        }
        triangulation.execute_coarsening_and_refinement ();
    }


    std::ofstream out ("test1.vtk");
    GridOut grid_out;
    grid_out.write_ucd (triangulation, out);
    std::cout << "Grid written to file" << std::endl;
}

int main() 
{
    std::cout << "Creating logo ..." << std::endl;
    create_logo();
    return 0;
}
