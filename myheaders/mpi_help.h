#ifndef MPI_HELP_H
#define MPI_HELP_H

#include <vector>
#include <mpi.h>

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

#endif // MPI_HELP_H