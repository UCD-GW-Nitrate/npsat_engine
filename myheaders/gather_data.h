#ifndef GATHER_DATA_H
#define GATHER_DATA_H

#include <deal.II/base/point.h>

#include "dsimstructs.h"
#include "boost_functions.h"

namespace Gather_Data{

using namespace dealii;

//! This is a class container  to hold information for a single particle
template <int dim>
class particle{
public:
    //! An empty constructor which should not be used
    particle();
    //! A constractor that initializes a particle with the required information
    particle(Point<dim> P, Point<dim> V, int proc);
    //! The position of the particle
    Point<dim> P;
    //! The velocity of the particle at position #P
    Point<dim> V;
    //! The age of the particle at position #P
    double AGE;
    //! The processor id that this particle was at position #P
    int proc_id;
};

template <int dim>
particle<dim>::particle()
{}

template <int dim>
particle<dim>::particle(Point<dim> P_in, Point<dim> V_in, int proc){
    P = P_in;
    V = V_in;
    proc_id = proc;
}

//! A class container to hold the streamline. This is slightly different than the streamline class used in
//! particle tracking
template<int dim>
class Streamline{
public:
    //! A constructor that initializes the #Length to 0
    Streamline();
    //! A constructor that initializes the streamline with the starting particle
    Streamline(int p_id, Gather_Data::particle<dim> part, int out);
    //! This is a flag that indicates the reason to terminate this streamline
    int Out_code;
    //! A map to hold the particles. the key for the map is a particle identifier produced by the particle
    //! tracking algorithm
    std::map<int, Gather_Data::particle<dim> > particles;
    //! The length of the streamline
    double Length;
    //! a method to add a particle in the streamline.
    void add_new_particle(int p_id, Gather_Data::particle<dim> part, int out);
};

template <int dim>
Streamline<dim>::Streamline(){
    Length = 0;
}

template <int dim>
Streamline<dim>::Streamline(int p_id, Gather_Data::particle<dim> part, int out){
    //if (p_id != 0)
        //std::cout << "The first point if the streamline has id > 0" << std::endl;
    add_new_particle(p_id, part, out);
    Length = 0;
}

template<int dim>
void Streamline<dim>::add_new_particle(int p_id, Gather_Data::particle<dim> part, int out){
    typename std::map<int, Gather_Data::particle<dim> >::iterator it = particles.find(p_id);
    if (it == particles.end()){
        particles[p_id] = part;
        Out_code = out;
    }
}

template <int dim>
class gather_particles{
public:
    gather_particles();
    void print_vtk(std::string filename, ParticleParameters param);
    void print_osg(std::string filename, ParticleParameters param);
    void print_streamline_length_age(std::string filename);
    void gather_streamlines(std::string basename, int n_proc, int n_chunks, std::vector<int> entity_ids);
    void print_stats();
    void calculate_age(bool backward, double unit_convertor);
    void simplify_XYZ_streamlines(double thres);
    void print_streamlines4URF(std::string basename, ParticleParameters param);
private:
    MPI_Comm    mpi_communicator;
    std::map<int, std::map<int,  Gather_Data::Streamline<dim> > > Entities;
    int Npos; // number of particle positions in the map
    void add_new_particle(int E_id, int S_id, int p_id, Point<dim> p, Point<dim> v, int proc, int out);
    std::map<int,int> get_Enitity_ids_for_my_rank(std::vector<int> entity_ids);
    unsigned int g_n_proc;
    unsigned int g_my_rank;
};

template <int dim>
gather_particles<dim>::gather_particles()
    :
    mpi_communicator (MPI_COMM_WORLD)
{
    Npos = 0;
    g_my_rank = Utilities::MPI::this_mpi_process(mpi_communicator);
    g_n_proc = Utilities::MPI::n_mpi_processes(mpi_communicator);
}

template<int dim>
void gather_particles<dim>::add_new_particle(int E_id, int S_id, int p_id, Point<dim> p, Point<dim> v, int proc, int out){
    typename std::map<int, std::map<int,  Gather_Data::Streamline<dim> > >::iterator entity_it;
    typename std::map<int,  Gather_Data::Streamline<dim> >::iterator strm_it;
    entity_it = Entities.find(E_id);
    if (entity_it == Entities.end()){
        Gather_Data::particle<dim> new_particle(p, v, proc);
        Gather_Data::Streamline<dim> new_streamline(p_id, new_particle, out);
        std::map<int,  Gather_Data::Streamline<dim> > temp_map;
        temp_map[S_id] = new_streamline;
        Entities[E_id] = temp_map;
        Npos++;
    }
    else{
        strm_it = entity_it->second.find(S_id);
        if (strm_it == entity_it->second.end()){
            Gather_Data::particle<dim> new_particle(p, v, proc);
            Gather_Data::Streamline<dim> new_streamline(p_id, new_particle, out);
            entity_it->second[S_id] = new_streamline;
            Npos++;
        }
        else{
            Gather_Data::particle<dim> new_particle(p, v, proc);
            strm_it->second.add_new_particle(p_id, new_particle, out);
            Npos++;
        }
    }
}

template <int dim>
std::map<int,int> gather_particles<dim>::get_Enitity_ids_for_my_rank(std::vector<int> entity_ids){
    int NEntities_per_proc = entity_ids.size() / g_n_proc;
    int start_Eid = 0;
    int end_Eid = NEntities_per_proc;
    std::map<int,int> Entity_Map_ids;
    for (unsigned int i_proc = 0; i_proc < g_n_proc; ++i_proc){
        if ( i_proc == g_my_rank){
            if (i_proc == g_n_proc - 1){
                end_Eid = entity_ids.size();
            }
            std::cout << "I'm processor " << g_my_rank << " and I'll gather from " << start_Eid << " to " << end_Eid << std::endl;
            for (int i = start_Eid; i < end_Eid; ++i){
                Entity_Map_ids.insert(std::pair<int,int>(entity_ids[i], entity_ids[i]));
            }
            break;
        }
        start_Eid = end_Eid;
        end_Eid += NEntities_per_proc;
    }

    return Entity_Map_ids;
}

template<int dim>
void gather_particles<dim>::gather_streamlines(std::string basename, int n_proc, int n_chunks, std::vector<int> entity_ids){

    std::map<int,int> Entity_map_id = get_Enitity_ids_for_my_rank(entity_ids);
    std::map<int,int>::iterator Eit;

    for (int i_chnk = 0; i_chnk < n_chunks; ++i_chnk){
        if (g_my_rank == 0)
            std::cout << "chunk " << i_chnk+1 << " out of " << n_chunks << std::endl;
        //std::cout << "\t";
        for (int i_proc = 0; i_proc < n_proc; ++i_proc){
            const std::string filename = (basename +
                                          Utilities::int_to_string(i_chnk, 4) +
                                          "_particles_"	+
                                          Utilities::int_to_string(i_proc, 4) +
                                          ".traj");

            std::ifstream  datafile(filename.c_str());
            if (!datafile.good()){
                std::cout << "Can't load the file " << filename << std::endl;
            }
            else{
                //std::cout << i_proc << " " << std::flush;
                //std::cout << "Reading particles from processor " << i_proc << std::endl;
                //typename std::map<int, std::map<int,  Gather_Data::Streamline<dim> > >::iterator well_it;
                //typename std::map<int,  Gather_Data::Streamline<dim> >::iterator strm_it;
                char buffer[512];
                while (datafile.good()){
                    datafile.getline(buffer,512);
                    std::istringstream inp(buffer);
                    int E_id, S_id, p_id, out;
                    double val;
                    Point<dim> p, v;
                    inp >> E_id;
                    Eit = Entity_map_id.find(E_id);
                    if (Eit != Entity_map_id.end()){
                        inp >> S_id;
                        inp >> out;
                        inp >> p_id;
                        for (unsigned int idim = 0; idim < dim; ++idim){
                            inp >> val;
                            p[idim] = val;
                        }
                        for (unsigned int idim = 0; idim < dim; ++idim){
                            inp >> val;
                            v[idim] = val;
                        }
                        add_new_particle(E_id, S_id, p_id, p, v, i_proc, out);
                    }

                    if( datafile.eof() )
                        break;
                }
            }
        }
        //std::cout << std::endl;
    }
}

template <int dim>
void gather_particles<dim>::calculate_age(bool backward, double unit_convertor){
    if (g_my_rank == 0)
        std::cout << "Calculating particles Age..." << std::endl;
    typename std::map<int, std::map<int,  Gather_Data::Streamline<dim> > >::iterator well_it = Entities.begin();
    for (; well_it != Entities.end(); ++well_it){
        typename std::map<int,  Gather_Data::Streamline<dim> >::iterator strm_it = well_it->second.begin();
        for (; strm_it != well_it->second.end(); ++strm_it){
            std::vector<double> DAGE;
            typename std::map<int, Gather_Data::particle<dim> >::iterator part_it = strm_it->second.particles.begin();
            typename std::map<int, Gather_Data::particle<dim> >::iterator part_it_1 = strm_it->second.particles.begin();
            for (; part_it != strm_it->second.particles.end(); ++part_it){
                if (part_it == strm_it->second.particles.begin())
                    continue;
                double dst = part_it->second.P.distance(part_it_1->second.P);
                double vel = (part_it->second.V.norm() + part_it_1->second.V.norm())/2;
                DAGE.push_back((dst/vel)/unit_convertor);
                part_it_1++;
                strm_it->second.Length += dst;
            }

            int iter = 0;
            if (backward){
                std::reverse(DAGE.begin(), DAGE.end());
                typename std::map<int, Gather_Data::particle<dim> >::reverse_iterator rpart_it = strm_it->second.particles.rbegin();
                typename std::map<int, Gather_Data::particle<dim> >::reverse_iterator rpart_it_1 = strm_it->second.particles.rbegin();
                for (; rpart_it != strm_it->second.particles.rend(); ++rpart_it){
                    if (rpart_it == strm_it->second.particles.rbegin()){
                        rpart_it->second.AGE = 0;
                    }
                    else{
                        rpart_it->second.AGE = rpart_it_1->second.AGE + DAGE[iter];
                        iter++;
                        rpart_it_1++;
                    }
                }
            }
            else{
                part_it = strm_it->second.particles.begin();
                part_it_1 = strm_it->second.particles.begin();
                for (; part_it != strm_it->second.particles.end(); ++part_it){
                    if (part_it == strm_it->second.particles.begin()){
                        part_it->second.AGE = 0;
                    }
                    else{
                        part_it->second.AGE = part_it_1->second.AGE + DAGE[iter];
                        iter++;
                        part_it_1++;
                    }
                }
            }
        }
    }
}

template<int dim>
void gather_particles<dim>::print_vtk(std::string basename, ParticleParameters param){

    std::cout << "Proc " << g_my_rank << " has " << Npos << "  particle positions" << std::endl;
    const std::string filename = (basename +
                                  Utilities::int_to_string(g_my_rank, 4) + "_Streamlines.vtk");
    std::ofstream file_strml;
    file_strml.open(filename.c_str());

    std::vector<Point<3> > vertices;
    std::vector<int> id_start;
    std::vector<int> id_end;
    std::vector<double> age;
    std::vector<double> velocity;
    std::vector<int> proc;
    int n_cell_id = 0;

    typename std::map<int, std::map<int,  Gather_Data::Streamline<dim> > >::iterator Entity_it = Entities.begin();
    int cnt_entities = 0; int cnt_strm = 0;
    for (; Entity_it != Entities.end(); ++Entity_it){
        if (cnt_entities == param.Entity_freq - 1){
            typename std::map<int,  Gather_Data::Streamline<dim> >::iterator strm_it = Entity_it->second.begin();
            for (; strm_it != Entity_it->second.end(); ++strm_it){
                if (cnt_strm == param.Streaml_freq - 1){
                    typename std::map<int, Gather_Data::particle<dim> >::iterator part_it = strm_it->second.particles.begin();
                    for (; part_it != strm_it->second.particles.end(); ++part_it){
                        if (dim == 2){
                            vertices.push_back(Point<3>(part_it->second.P[0], part_it->second.P[1], 0.0));
                        }
                        else if (dim == 3){
                            vertices.push_back(Point<3>(part_it->second.P[0], part_it->second.P[1], part_it->second.P[2]));
                        }
                        velocity.push_back(part_it->second.V.norm());
                        age.push_back(part_it->second.AGE);
                        proc.push_back(part_it->second.proc_id);
                        if (part_it == strm_it->second.particles.begin())
                            id_start.push_back(static_cast<int>(vertices.size()) - 1);
                    }
                    id_end.push_back(static_cast<int>(vertices.size()) - 1);
                    n_cell_id += 1 + (id_end[id_end.size()-1] - id_start[id_start.size()-1] + 1);
                    cnt_strm = 0;
                }
                else
                    cnt_strm++;
            }
            cnt_entities = 0;
        }
        else
            cnt_entities++;
    }

    // write headers
    file_strml << "# vtk DataFile Version 3.0" << std::endl;
    file_strml << "Streamlines for " << basename << std::endl;
    file_strml << "ASCII" << std::endl;
    file_strml << "DATASET UNSTRUCTURED_GRID" << std::endl;
    file_strml << "POINTS " << vertices.size() << " double" << std::endl;

    for (unsigned int i = 0; i < vertices.size(); ++i){
        file_strml << std::setprecision(10)
                   << vertices[i][0] << " "
                   << vertices[i][1] << " "
                   << vertices[i][2] << std::endl;
    }

    file_strml << "CELLS " << id_start.size() << " " << n_cell_id << std::endl;
    for (unsigned int i = 0; i < id_start.size(); ++i){
        file_strml << (id_end[i] - id_start[i]) + 1;
        for (int j = id_start[i]; j <=id_end[i]; ++j)
            file_strml << " " << j;
        file_strml << std::endl;
    }

    file_strml << "CELL_TYPES " << id_start.size() << std::endl;
    for (unsigned int i = 0; i < id_start.size(); ++i)
        file_strml << "4" << std::endl;

    file_strml << "POINT_DATA " << vertices.size() << std::endl;
    file_strml << "SCALARS Age double 1" << std::endl;
    file_strml << "LOOKUP_TABLE default" << std::endl;
    for (unsigned int i = 0; i < age.size(); ++i){
        file_strml << std::setprecision(8) << age[i] << std::endl;
    }

    file_strml << "SCALARS Velocity double 1" << std::endl;
    file_strml << "LOOKUP_TABLE default" << std::endl;
    for (unsigned int i = 0; i < velocity.size(); ++i){
        file_strml << std::setprecision(8) << velocity[i] << std::endl;
    }

    file_strml << "SCALARS Proc double 1" << std::endl;
    file_strml << "LOOKUP_TABLE default" << std::endl;
    for (unsigned int i = 0; i < velocity.size(); ++i){
        file_strml << proc[i] << std::endl;
    }
    file_strml.close();

    std::cout << "Vtk data written in: " << filename << std::endl;
}

template <int dim>
void gather_particles<dim>::simplify_XYZ_streamlines(double thres){
    if (g_my_rank == 0)
        std::cout << "Simplifying streamlines... " << std::endl;
    typename std::map<int, std::map<int,  Gather_Data::Streamline<dim> > >::iterator it = Entities.begin();
    for (; it != Entities.end(); ++it){
        typename std::map<int,  Gather_Data::Streamline<dim> >::iterator itt = it->second.begin();
        for (; itt != it->second.end(); ++itt){
            typename std::map<int, Gather_Data::particle<dim> >::iterator ittt = itt->second.particles.begin();
            std::vector<double> x, y, z;
            for (; ittt != itt->second.particles.end(); ++ittt){
                x.push_back(ittt->second.P[0]);
                y.push_back(ittt->second.P[1]);
                if (dim == 2){
                    y.push_back(0.0);
                }
                else if (dim == 3){
                    z.push_back(ittt->second.P[2]);
                }
            }
            simplify_polyline(thres, x, y, z);
            int loc = 0; double dst;
            ittt = itt->second.particles.begin();
            for (; ittt != itt->second.particles.end();){
                if (dim == 2)
                    dst = ittt->second.P.distance(Point<dim>(x[loc], y[loc]));
                else if (dim == 3)
                    dst = ittt->second.P.distance(Point<dim>(x[loc], y[loc], z[loc]));

                if (dst < 0.001){
                    ittt++;
                    loc++;
                }
                else{
                    // if the point is not found delete. The erase will move the iterator to the next element of the vector
                    ittt = itt->second.particles.erase(ittt);
                }
            }
        }
    }
}

template <int dim>
void gather_particles<dim>::print_streamlines4URF(std::string basename, ParticleParameters param){

    int file_id = 0;
    int count_strmln = 0;

    std::string filename = (basename +
                            Utilities::int_to_string(file_id, 4) + "_" +
                            Utilities::int_to_string(g_my_rank, 4) + "_" +
                            "streamlines.urfs");
    if (g_my_rank == 0)
        std::cout << "Printing streamline file: " << filename << std::endl;
    std::ofstream file_strml;
    file_strml.open(filename.c_str());

    typename std::map<int, std::map<int,  Gather_Data::Streamline<dim> > >::iterator well_it = Entities.begin();
    for (; well_it != Entities.end(); ++well_it){
        typename std::map<int,  Gather_Data::Streamline<dim> >::iterator strm_it = well_it->second.begin();
        for (; strm_it != well_it->second.end(); ++strm_it){
            file_strml << well_it->first << " "
                       << strm_it->first << " "
                       << strm_it->second.particles.size() << std::endl;
            typename std::map<int, Gather_Data::particle<dim> >::iterator part_it = strm_it->second.particles.begin();
            for (; part_it != strm_it->second.particles.end(); ++part_it){
                file_strml << std::setprecision(10)
                           << part_it->second.P[0] << " "
                           << part_it->second.P[1] << " "
                           << part_it->second.P[2] << " "
                           << part_it->second.V.norm() << std::endl;
            }
            count_strmln++;
        }

        if (count_strmln > param.Nparallel_particles){
            count_strmln = 0;
            file_strml.close();
            file_id++;
            filename = (basename +
                        Utilities::int_to_string(file_id, 4) + "_" +
                        Utilities::int_to_string(g_my_rank, 4) + "_" +
                        "streamlines.urfs");
            if (g_my_rank == 0)
                std::cout << "Printing streamline file: " << filename << std::endl;
            file_strml.open(filename.c_str());
        }
    }
    file_strml.close();

}

template<int dim>
void gather_particles<dim>::print_osg(std::string base_name, ParticleParameters param){
    const std::string filename = (base_name + "osg_Streamlines.nps");
    std::ofstream file_strml;
    file_strml.open(filename.c_str());

    typename std::map<int, std::map<int,  Gather_Data::Streamline<dim> > >::iterator well_it = Entities.begin();
    int cnt_wells = 0; int cnt_strm = 0;
    for (; well_it != Entities.end(); ++well_it){
        if (cnt_wells == param.Entity_freq - 1){
            typename std::map<int,  Gather_Data::Streamline<dim> >::iterator strm_it = well_it->second.begin();
            for (; strm_it != well_it->second.end(); ++strm_it){
                if (cnt_strm == param.Streaml_freq - 1){
                    file_strml << strm_it->second.particles.size() << " "
                               << well_it->first << " "
                               << strm_it->first << std::endl;
                    typename std::map<int, Gather_Data::particle<dim> >::iterator part_it = strm_it->second.particles.begin();
                    for (; part_it != strm_it->second.particles.end(); ++part_it){
                        file_strml << std::setprecision(10)
                                   << part_it->second.P[0] << " "
                                   << part_it->second.P[1] << " "
                                   << part_it->second.P[2] << " "
                                   << part_it->second.V.norm() << " "
                                   << part_it->second.AGE << " "
                                   << part_it->second.proc_id << std::endl;
                    }
                    cnt_strm = 0;
                }
                else
                    cnt_strm++;
            }
            cnt_wells = 0;
        }
        else
            cnt_wells++;
    }
    file_strml.close();
}



}


#endif // GATHER_DATA_H
