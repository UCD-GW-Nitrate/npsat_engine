#ifndef MPI_HELP_H
#define MPI_HELP_H

#include <vector>
#include <mpi.h>

#include "streamlines.h"

//! A method that prints the size of the input vector that my_rank processor has
//! This is used for debuging only as it will produce alot of output
template <typename T>
void print_size_msg(std::vector<std::vector<T>> v, int my_rank){
    for (unsigned int i_proc = 0; i_proc < v.size(); ++i_proc){
        std::cout << "I'm Rank: " << my_rank << " and I have from/for proc: " << i_proc << ": " << v[i_proc].size() << " data." << std::endl;
    }
}

/*!
 * \brief This function reads the #i, #j element of a 2D vector after checking
 * whether the indices are in the range of the vector. It is supposed to be a
 * safe way to do v[i][j]
 * \param v the 2D vector.
 * \param i is the index of first element.
 * \param j is the index of the second element.
 */
template <typename T>
T get_v(std::vector<std::vector<T> > v, unsigned int i, unsigned int j){
    if (i < v.size()){
        if (j < v[i].size()){
            return v[i][j];
        }else
            std::cerr << "vector index j:" << j << " out of size: " << v[i].size() << std::endl;
    }else
        std::cerr << "vector index i:" << i << " out of size: " << v.size() << std::endl;
    return 0;
}

/*!
 * \brief Send_receive_size: Each processor sends and receives an integer
 * \param N This is the integer to be sent from this processor
 * \param n_proc The total number of processors
 * \param output A vector of size n_proc which containts the integer that all processors have been sent
 * \param comm The typical MPI communicator
 */
void Send_receive_size(unsigned int N, unsigned int n_proc, std::vector<int> &output, MPI_Comm comm){

        output.clear();
        output.resize(n_proc);
        std::vector<int> temp(n_proc,1);
        std::vector<int> displs(n_proc);
        for (unsigned int i=1; i<n_proc; i++)
                displs[i] = displs[i-1] + 1;

        MPI_Allgatherv(&N, // This is what this processor will send to every other
                       1, //This is the size of the message from this processor
                       MPI_INT, // The data type will be sent
                       &output[0], // This is where the data will be send on each processor
                       &temp[0], // an array with the number of points to be sent/receive
                       &displs[0],
                       MPI_INT, comm);
}

/*!
 * \brief Sent_receive_data: This function sends a vector to all processors and receives all the vectors that the other processor
 * have sent
 * \param data Is a vector of vectors of type T1 with size equal to n_proc.
 * \param N_data_per_proc This is the amount of data that each processor will send.
 * Typically prior to this function the Send_receive_size function should be called to send the sizes
 * \param my_rank the rank of the current processor
 * \param comm The MPI communicator
 * \param MPI_TYPE The mpi type which should match with the templated parameter T1
 */
template <typename T1>
void Sent_receive_data(std::vector<std::vector<T1> > &data,
                       std::vector <int> N_data_per_proc,
                       unsigned int my_rank,
                       MPI_Comm comm,
                       MPI_Datatype MPI_TYPE){

    // data is a vector of vectors of type T1 with size equal to n_proc.
    // This function transfer to all processors the content of data[my_rank]
    // if there are any data in data[i], where i=[1,n_proc; i!=myrank] this will be deleted
    // The size of data[my_rank].size() = N_data_per_proc[my_rank]. This is the responsibility of user

    int N = data[my_rank].size();
    unsigned int n_proc = data.size();
    std::vector<int> displs(n_proc);
    displs[0] = 0;
    for (unsigned int i=1; i<n_proc; i++)
            displs[i] = displs[i-1] + N_data_per_proc[i-1];

    int totdata = displs[n_proc-1] + N_data_per_proc[n_proc-1];
    std::vector<T1> temp_receive(totdata);

    MPI_Allgatherv(&data[my_rank][0], // This is what this processor will send to every other
                   N, //This is the size of the message from this processor
                   MPI_TYPE, // The data type will be sent
                   &temp_receive[0], // This is where the data will be send on each processor
                   &N_data_per_proc[0], // an array with the number of points to be sent/receive
                   &displs[0],
                   MPI_TYPE, comm);

    // Now put the data in the data vector
    for (unsigned int i = 0; i < n_proc; ++i){
            data[i].clear();
            data[i].resize(N_data_per_proc[i]);
            for (int j = 0; j < N_data_per_proc[i]; ++j)
                    data[i][j] = temp_receive[displs[i] +j];
    }

}

/*!
 * \brief Sent_receive_streamlines_all_to_all As the name suggests each processor will send the streamline it owns
 * and receives streamlines from evey other processor
 * \param streamlines The Vector of streamlines of size n_proc. Each processor will send only the data containted in
 * streamlines[my_rank]
 * \param my_rank The rank of the current processor
 * \param n_proc The total number of processors
 * \param mpi_communicator The typical MPI communicator
 */
template <int dim>
void Sent_receive_streamlines_all_to_all(std::vector<std::vector<Streamline<dim>>> &streamlines,
                                         unsigned int my_rank, unsigned int n_proc, MPI_Comm mpi_communicator){
    std::vector<std::vector<double> > px(n_proc);
    std::vector<std::vector<double> > py(n_proc);
    std::vector<std::vector<double> > pz(n_proc);
    std::vector<std::vector<int> > E_id(n_proc);
    std::vector<std::vector<int> > S_id(n_proc);
    std::vector<std::vector<int> > proc_id(n_proc);
    std::vector<std::vector<int> > p_id(n_proc);
    std::vector<std::vector<double> > BBlx(n_proc);
    std::vector<std::vector<double> > BBly(n_proc);
    std::vector<std::vector<double> > BBlz(n_proc);
    std::vector<std::vector<double> > BBux(n_proc);
    std::vector<std::vector<double> > BBuy(n_proc);
    std::vector<std::vector<double> > BBuz(n_proc);

    // copy the data
    for (unsigned int i = 0; i < streamlines[my_rank].size(); ++i){
        px[my_rank].push_back(streamlines[my_rank][i].P[0][0]);
        py[my_rank].push_back(streamlines[my_rank][i].P[0][1]);
        if (dim == 3)
            pz[my_rank].push_back(streamlines[my_rank][i].P[0][2]);
        E_id[my_rank].push_back(streamlines[my_rank][i].E_id);
        S_id[my_rank].push_back(streamlines[my_rank][i].S_id);
        proc_id[my_rank].push_back(streamlines[my_rank][i].proc_id);
        p_id[my_rank].push_back(streamlines[my_rank][i].p_id[0]);
        BBlx[my_rank].push_back(streamlines[my_rank][i].BBl[0]);
        BBly[my_rank].push_back(streamlines[my_rank][i].BBl[1]);
        if (dim == 3)
            BBlz[my_rank].push_back(streamlines[my_rank][i].BBl[2]);
        BBux[my_rank].push_back(streamlines[my_rank][i].BBu[0]);
        BBuy[my_rank].push_back(streamlines[my_rank][i].BBu[1]);
        if (dim == 3)
            BBuz[my_rank].push_back(streamlines[my_rank][i].BBu[2]);
    }
    MPI_Barrier(mpi_communicator);

    // Send everything to every processor
    std::vector<int> data_per_proc;
    Send_receive_size(static_cast<unsigned int>(px[my_rank].size()), n_proc, data_per_proc, mpi_communicator);
    Sent_receive_data<double>(px, data_per_proc, my_rank, mpi_communicator, MPI_DOUBLE);
    Sent_receive_data<double>(py, data_per_proc, my_rank, mpi_communicator, MPI_DOUBLE);
    if (dim == 3)
        Sent_receive_data<double>(pz, data_per_proc, my_rank, mpi_communicator, MPI_DOUBLE);
    Sent_receive_data<int>(E_id, data_per_proc, my_rank, mpi_communicator, MPI_INT);
    Sent_receive_data<int>(S_id, data_per_proc, my_rank, mpi_communicator, MPI_INT);
    Sent_receive_data<int>(proc_id, data_per_proc, my_rank, mpi_communicator, MPI_INT);
    Sent_receive_data<int>(p_id, data_per_proc, my_rank, mpi_communicator, MPI_INT);
    Sent_receive_data<double>(BBlx, data_per_proc, my_rank, mpi_communicator, MPI_DOUBLE);
    Sent_receive_data<double>(BBly, data_per_proc, my_rank, mpi_communicator, MPI_DOUBLE);
    if (dim == 3)
        Sent_receive_data<double>(BBlz, data_per_proc, my_rank, mpi_communicator, MPI_DOUBLE);
    Sent_receive_data<double>(BBux, data_per_proc, my_rank, mpi_communicator, MPI_DOUBLE);
    Sent_receive_data<double>(BBuy, data_per_proc, my_rank, mpi_communicator, MPI_DOUBLE);
    if (dim == 3)
        Sent_receive_data<double>(BBuz, data_per_proc, my_rank, mpi_communicator, MPI_DOUBLE);

    // now we loop through the data and get all the data from the other processors
    for (unsigned int i = 0; i < n_proc; ++i){
        if (i == my_rank)
            continue;
        for (unsigned int j = 0; j < px[i].size(); ++j){
            dealii::Point<dim> p;
            p[0] = px[i][j];
            p[1] = py[i][j];
            if (dim == 3)
                p[2] = pz[i][j];
            Streamline<dim> temp(E_id[i][j], S_id[i][j], p);
            p[0] = BBlx[i][j];
            p[1] = BBly[i][j];
            if (dim == 3)
                p[2] = BBlz[i][j];
            temp.BBl = p;
            p[0] = BBux[i][j];
            p[1] = BBuy[i][j];
            if (dim == 3)
                p[2] = BBuz[i][j];
            temp.BBu = p;
            temp.p_id[0] = p_id[i][j];
            streamlines[my_rank].push_back(temp);
        }
    }

}

#endif // MPI_HELP_H
