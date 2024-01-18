#ifndef MAKE_GRID_H
#define MAKE_GRID_H

#include <deal.II/grid/tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_reordering.h>

#include "my_macros.h"

#include "dsimstructs.h"

namespace AquiferGrid{
    using namespace dealii;

    //===================================================
    // DECLARATIONS
    //===================================================

    /*!
    * \brief The GridGenerator class provides the required functionality to generate a grid based on the
    * user input
    */
    template <int dim>
    class GridGenerator{
    public:
        /*!
        * \brief GridGenerator The constructor simply sets the input
        * \param g class with the data for generating the geometry domain
        */
        GridGenerator(AquiferProperties<dim>& g);

        /*!
        * \brief make_grid This is the main entry of this class.
        * \param triangulation will hold the output mesh
        */
        void make_grid(parallel::distributed::Triangulation<dim>& triangulation);

        //! Print the initial grid to file
        void print_Initial_grid(std::string filename);

    private:
        //! Parameters that control the shape of the domain and all properties
        AquiferProperties<dim> geom_param;

        //! This method is called when the aquifer is a box. Creates a box domain with dimensions
        //! indicated by #AquiferProperties::left_lower_point and #AquiferProperties::Length.
        //! However the #dim-1 dimension is set to 0 with lenght 1. The actual elevation is set
        //! afterwards using the inputs from the #AquiferProperties::top_elevation and
        //! #AquiferProperties::bottom_elevation functions.
        void make_box(parallel::distributed::Triangulation<dim>& triangulation);

        /*!
        * \brief Reads the mesh file and creates a 2D triangulation.
        * \param triangulation will hold the 2D output mesh.
        */
        bool read_2D_grid(Triangulation<dim-1>& triangulation);

        //! Converts the triangulation to a parallel version
        void convert_to_parallel(Triangulation<dim>& tria3D, parallel::distributed::Triangulation<dim>& triangulation);

        //! Assigns boundary ids
        void assign_default_boundary_id(parallel::distributed::Triangulation<dim>& triangulation);

        int readLayerRelativeElevation();

        std::vector<std::vector<double>> laydata;
        PointIdCloud interpCloud;
        std::shared_ptr<pointid_kd_tree> interpIndex;


    };

    //===================================================
    // IMPLEMENTATION
    //===================================================
    template<int dim>
    GridGenerator<dim>::GridGenerator(AquiferProperties<dim>& g)
        :
        geom_param(g)
    {}

    template <int dim>
    void GridGenerator<dim>::make_box(parallel::distributed::Triangulation<dim>& triangulation){
        Point<dim> left_bottom, right_top;
        std::vector<unsigned int>	n_cells;
        for (unsigned int i = 0; i < dim; ++i){
            left_bottom[i] = geom_param.left_lower_point[i];
            right_top[i] = left_bottom[i] + geom_param.Length[i];
            n_cells.push_back(geom_param.Nxyz[i]);
        }
        if (geom_param.vert_discr.size() > 1)
            n_cells[dim-1] = geom_param.vert_discr.size()-1;

        left_bottom[dim-1] = 0;
        right_top[dim-1] = 100;

        dealii::GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                          n_cells,
                                                          left_bottom,
                                                          right_top,
                                                          true);
    }

    template <int dim>
    void GridGenerator<dim>::make_grid(parallel::distributed::Triangulation<dim>& triangulation){
        if (geom_param.geomtype == "BOX"){
            make_box((triangulation));
            //triangulation.refine_global(geom_param.N_init_refinement);
        }
        else if (geom_param.geomtype == "FILE"){
            if (dim !=3 ){
                std::cerr << "You cannot use the FILE geometry with problem dimension other that 3D" << std::endl;
            }else{
                Triangulation<dim-1> tria2D;
                Triangulation<dim> tria3D;
                bool done = read_2D_grid(tria2D);

                if (done){
#if _DIM>2
                    if (!geom_param.use_rel_elev_file) {
                        dealii::GridGenerator::extrude_triangulation(tria2D,geom_param.vert_discr.size(), 100, tria3D);
                    }
                    else{
                        int nlay = readLayerRelativeElevation();
                        dealii::GridGenerator::extrude_triangulation(tria2D, nlay+2, 100, tria3D);

                        nanoflann::SearchParams params;
                        params.sorted = true;
                        std::vector<std::pair<size_t,double> > ret_matches;
                        std::vector<size_t>   ret_index(1);
                        std::vector<double> out_dist_sqr(1);
                        size_t num_results;
                        std::map<int,double> processed_vertices;
                        std::map<int,double>::iterator vert_it;
                        typename Triangulation<dim>::active_cell_iterator
                        cell = tria3D.begin_active(),
                        endc = tria3D.end();
                        for (; cell!=endc; ++cell){
                            for (unsigned int vertex_no = 0;  vertex_no < GeometryInfo<dim>::vertices_per_cell; ++vertex_no){
                                int vert_idx = cell->vertex_index(vertex_no);
                                vert_it = processed_vertices.find(vert_idx);
                                if (vert_it != processed_vertices.end())
                                    continue;

                                Point<dim> &v = cell->vertex(vertex_no);

                                if (std::abs(v[2]) < 0.001 || std::abs(v[2] - 100) < 0.001)
                                    continue;
                                double d_idx = std::round(v[2]*static_cast<double>(nlay+1)/100);
                                int idx = static_cast<int>(d_idx)-1;

                                double query_pt[2];
                                query_pt[0] = v[0];
                                query_pt[1] = v[1];
                                num_results = interpIndex->knnSearch(&query_pt[0],
                                                                     1,
                                                                     &ret_index[0],
                                                                     &out_dist_sqr[0]);

                                //std::cout << ret_index[0] << std::endl;
                                //std::cout << laydata[ret_index[0]][idx] << std::endl;
                                v[2] = 100.0*laydata[ret_index[0]][idx];

                                //std::cout << vert_idx << ": " << v << " | " << ret_index[0] << " | " << out_dist_sqr[0] << std::endl;
                                processed_vertices.insert(std::pair<int,double>(vert_idx, v[2]));
                            }
                        }
                    }

                    // This is going to work for deal version 9 and higher
                    /*std::vector<double> slices;
                    for (unsigned int i = 0; i < geom_param.vert_discr.size(); ++i)
                        slices.push_back(100.0 * static_cast<double>(geom_param.vert_discr[i]));

                    dealii::GridGenerator::extrude_triangulation(tria2D, slices, tria3D); */

                    convert_to_parallel(tria3D, triangulation);
                    //triangulation.refine_global(geom_param.N_init_refinement);
                    assign_default_boundary_id(triangulation);
#endif

                }
                else{
                    std::cerr<< "Unable to read triangulation" << std::endl;
                }
            }
        }

        //std::ofstream out ("test_tria.vtk");
        //GridOut grid_out;
        //grid_out.write_ucd(triangulation, out);
    }

    template <int dim>
    bool GridGenerator<dim>::read_2D_grid(Triangulation<dim-1>& triangulation){
        bool outcome = false;
        std::ifstream tria_file(geom_param.input_mesh_file.c_str());
        if (tria_file.good()){
            std::vector< Point<dim-1>> vertices;
            std::vector< CellData<dim-1>> cells;
            SubCellData subcelldata;
            char buffer[512];
            unsigned int Nvert, Nelem;
            tria_file.getline(buffer,512);
            std::istringstream inp(buffer);
            inp >> Nvert; inp >> Nelem;
            vertices.resize(Nvert);
            cells.resize(Nelem);
            {// Read vertices
                Point<dim-1> temp;
                for (unsigned int i = 0; i < Nvert; ++i){
                    tria_file.getline(buffer,512);
                    std::istringstream inp(buffer);
                    inp >> temp(0);
                    inp >> temp(1);
                    vertices[i] = temp;
                }
            }

            {//Read elements
                std::vector<int> temp_int(4);
                for (unsigned int i = 0; i < Nelem; ++i){
                    tria_file.getline(buffer,512);
                    std::istringstream inp(buffer);
                    for (unsigned int j = 0; j < 4; ++j){
                        inp >> temp_int[j];
                    }
                    cells[i].vertices[0] = temp_int[0];
                    cells[i].vertices[1] = temp_int[1];
                    cells[i].vertices[2] = temp_int[2];
                    cells[i].vertices[3] = temp_int[3];
                }
            }
            GridTools::delete_unused_vertices(vertices, cells, subcelldata);
            GridReordering<dim-1>::invert_all_cells_of_negative_grid(vertices,cells);
            GridReordering<dim-1>::reorder_cells(cells);
            triangulation.create_triangulation_compatibility(vertices, cells, subcelldata);
            outcome = true;
        }
        return outcome;
    }

    template <int dim>
    void GridGenerator<dim>::convert_to_parallel(Triangulation<dim>& tria3D, parallel::distributed::Triangulation<dim>& triangulation){
        std::vector<Point<dim> > tria3Dvert = tria3D.get_vertices();
        std::vector<Point<dim> > vert(tria3Dvert.size());
        for (unsigned int i = 0; i < tria3Dvert.size(); ++i){
            Point<dim> temp;
            for (unsigned int j = 0; j < dim; ++j){
                temp[j] = tria3Dvert[i][j];
            }
            vert[i]=temp;
        }

        std::vector< CellData<dim> > cells(tria3D.n_cells());
        std::cout << tria3D.n_vertices() << " " << tria3D.n_used_vertices() << std::endl;

        typename Triangulation<dim>::active_cell_iterator
        cell = tria3D.begin_active(),
        endc = tria3D.end();
        int cell_index = 0;
        for (; cell!=endc; ++cell){
            for (unsigned int j = 0; j < GeometryInfo<dim>::vertices_per_cell; ++j){
                cells[cell_index].vertices[j] = cell->vertex_index(j);
            }
            ++cell_index;
        }

        SubCellData subcelldata;
        triangulation.create_triangulation(vert, cells, subcelldata);
    }

    template <int dim>
    void GridGenerator<dim>::assign_default_boundary_id(parallel::distributed::Triangulation<dim>& triangulation){
        typename parallel::distributed::Triangulation<3>::active_cell_iterator
        cell = triangulation.begin_active(),
        endc = triangulation.end();
        for (; cell!=endc; ++cell){
            if (cell->is_locally_owned()){
                for (unsigned int i_face = 0; i_face < GeometryInfo<dim>::faces_per_cell; ++i_face){
                    if (cell->face(i_face)->at_boundary()){
                        cell->face(i_face)->set_all_boundary_ids(static_cast<types::boundary_id>(i_face));
                    }
                }
            }
        }
    }

    template <int dim>
    int GridGenerator<dim>::readLayerRelativeElevation() {
        int nlay;
        std::ifstream rel_elev_file(geom_param.rel_elev_file.c_str());
        if (rel_elev_file.good()){
            char buffer[512];
            int Npnts;
            rel_elev_file.getline(buffer, 512);
            {
                std::istringstream inp(buffer);
                inp >> Npnts;
                inp >> nlay;
            }
            {//Read info
                std::vector<double> tmp_data;
                for (int i = 0; i < Npnts; ++i){
                    tmp_data.clear();
                    rel_elev_file.getline(buffer, 512);
                    std::istringstream inp(buffer);
                    PointId pid;
                    double x, y, v;
                    inp >> x;
                    inp >> y;
                    pid.x = x;
                    pid.y = y;
                    pid.id = i;
                    for (int j = 0; j < nlay; ++j){
                        inp >> v;
                        tmp_data.push_back(v);
                    }
                    laydata.push_back(tmp_data);
                    interpCloud.pts.push_back(pid);
                }
                int Nleafs = 10;
                interpIndex = std::shared_ptr<pointid_kd_tree>(
                        new pointid_kd_tree (2, interpCloud,
                                             nanoflann::KDTreeSingleIndexAdaptorParams(Nleafs)));

                interpIndex->buildIndex();
            }
        }

        return nlay;
    }
}





#endif // MAKE_GRID_H
