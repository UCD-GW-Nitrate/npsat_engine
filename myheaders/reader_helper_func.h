//
// Created by giorgk on 6/23/26.
//

#ifndef READER_HELPER_FUNC_H
#define READER_HELPER_FUNC_H

//#include <mpi.h>
#include <deal.II/base/mpi.h>

#include <limits>
#include <stdexcept>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <cctype>
#include <algorithm>

#include "flow_structures.h"

namespace npsat_flow{

    static inline bool is_comment_or_empty_line(const std::string &line)
    {
        for (char c : line)
        {
            if (std::isspace(static_cast<unsigned char>(c))) continue;
            return (c == '#'); // treat lines starting with '#' as comments
        }
        return true; // all whitespace => empty
    }

    static inline void normalize_separators(std::string &line)
    {
        // Convert common separators to spaces so operator>> works.
        // Comma-separated and/or space-separated input is supported.
        for (char &c : line)
        {
            if (c == ',' || c == ';' || c == '\t')
                c = ' ';
        }
    }


    template <typename T>
        static std::vector<std::vector<T>>
        read_ascii_2d_on_rank0_typed(const std::string &filename,
                                  std::int64_t &nrows_out,
                                  std::int64_t &ncols_out)
    {
        static_assert(std::is_arithmetic<T>::value, "T must be arithmetic.");

        std::ifstream in(filename);
        if (!in)
            throw std::runtime_error("Cannot open file: " + filename);

        std::vector<std::vector<T>> rows;
        rows.reserve(1024);

        std::string line;
        std::int64_t ncols = -1;

        std::int64_t line_no = 0;
        while (std::getline(in, line))
        {
            ++line_no;
            if (is_comment_or_empty_line(line))
                continue;

            normalize_separators(line);

            std::istringstream iss(line);
            std::vector<T> vals;
            vals.reserve(256);

            // Parse through a wide type to avoid surprising overflow behavior during extraction.
            // Then cast to T (you may clamp/validate if you want stricter behavior).
            long double tmp;
            while (iss >> tmp)
                vals.push_back(static_cast<T>(tmp));

            if (vals.empty())
                continue;

            if (ncols < 0)
            {
                ncols = static_cast<std::int64_t>(vals.size());
            }
            else if (static_cast<std::int64_t>(vals.size()) != ncols)
            {
                throw std::runtime_error("Inconsistent column count in '" + filename +
                                 "' at line " + std::to_string(line_no) +
                                 ": expected " + std::to_string(ncols) +
                                 ", got " + std::to_string(vals.size()));
            }

            rows.emplace_back(std::move(vals));
        }

        nrows_out = static_cast<std::int64_t>(rows.size());
        ncols_out = (ncols < 0) ? 0 : ncols;
        return rows;
    }


    // =============================
    // MPI datatype mapping
    // =============================
    template <typename T>
    struct MpiType;

    template <>
    struct MpiType<double>
    {
        static MPI_Datatype type() { return MPI_DOUBLE; }
    };

    template <>
    struct MpiType<float>
    {
        static MPI_Datatype type() { return MPI_FLOAT; }
    };

    template <>
    struct MpiType<int>
    {
        static MPI_Datatype type() { return MPI_INT; }
    };

    template <>
    struct MpiType<std::int64_t>
    {
        static MPI_Datatype type() { return MPI_INT64_T; }
    };

    /// =============================
    // read_2d_array_mpi (typed, flat broadcast)
    // - All ranks call
    // - rank 0 reads ASCII and broadcasts dims + flat data
    // - Returns flat storage and dims
    // =============================
template <typename T>
    static void read_2d_array_mpi_flat(const std::string &filename,
                                  MPI_Comm comm,
                                  std::vector<T> &flat_out,
                                  std::int64_t &nrows_out,
                                  std::int64_t &ncols_out)
    {
        int rank = 0;
        MPI_Comm_rank(comm, &rank);

        std::int64_t nrows = 0, ncols = 0;
        std::vector<T> flat; // contiguous [r*ncols + c]

        if (rank == 0) {
            auto rows = read_ascii_2d_on_rank0_typed<T>(filename, nrows, ncols);

            const std::int64_t total64 = nrows * ncols;
            if (total64 < 0)
                throw std::runtime_error("Invalid matrix size (overflow).");

            flat.resize(static_cast<std::size_t>(total64));

            for (std::int64_t r = 0; r < nrows; ++r)
            {
                const auto &row = rows[static_cast<std::size_t>(r)];
                std::copy(row.begin(), row.end(),
                          flat.begin() + static_cast<std::size_t>(r) * static_cast<std::size_t>(ncols));
            }
        }

        // Broadcast dimensions first
        MPI_Bcast(&nrows, 1, MPI_INT64_T, 0, comm);
        MPI_Bcast(&ncols, 1, MPI_INT64_T, 0, comm);

        // Broadcast data
        const std::int64_t total64 = nrows * ncols;
        if (total64 < 0)
            throw std::runtime_error("Invalid matrix size (overflow).");

        const std::size_t total = static_cast<std::size_t>(total64);

        if (rank != 0)
            flat.resize(total);

        MPI_Datatype mpi_t = MpiType<T>::type();

        // MPI_Bcast takes an int count; for very large arrays you may need chunking.
        // 200MB doubles ~ 25 million => count fits in 32-bit int (~25e6).
        if (total > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        {
            // Chunked broadcast for extremely large matrices (> ~2B doubles).
            const std::size_t chunk = static_cast<std::size_t>(std::numeric_limits<int>::max());
            std::size_t offset = 0;
            while (offset < total)
            {
                const int this_count = static_cast<int>(std::min(chunk, total - offset));
                MPI_Bcast(flat.data() + offset, this_count, mpi_t, 0, comm);
                offset += static_cast<std::size_t>(this_count);
            }
        }
        else
        {
            MPI_Bcast(flat.data(), static_cast<int>(total), mpi_t, 0, comm);
        }

        // Output
        flat_out.swap(flat);
        nrows_out = nrows;
        ncols_out = ncols;
    }


    /// =============================
    // Wrapper: returns MatrixView<T> + ensures rectangularity (on rank 0 parse)
    // - Uses flat broadcast => no vector<vector<T>> on non-root ranks
    // - Rectangularity is enforced during root parse (inconsistent cols throws)
    // =============================
    template <typename T>
    MatrixView<T> read_2d_array_mpi_as_view(const std::string &filename,
                                            MPI_Comm comm,
                                            std::vector<T> &storage)
    {
        std::int64_t nrows = 0, ncols = 0;
        read_2d_array_mpi_flat<T>(filename, comm, storage, nrows, ncols);

        MatrixView<T> view;
        view.data  = storage.empty() ? nullptr : storage.data();
        view.nrows = nrows;
        view.ncols = ncols;
        return view;

    }

    static inline void broadcast_string_mpi(std::string &text,
                                            MPI_Comm comm,
                                            const int root = 0)
    {
        int rank = 0;
        MPI_Comm_rank(comm, &rank);

        std::uint64_t text_size = static_cast<std::uint64_t>(text.size());
        MPI_Bcast(&text_size, 1, MPI_UINT64_T, root, comm);

        if (rank != root)
            text.resize(static_cast<std::size_t>(text_size));

        std::uint64_t offset = 0;
        while (offset < text_size)
        {
            const std::uint64_t remaining = text_size - offset;
            const int this_count =
                static_cast<int>(std::min<std::uint64_t>(
                    remaining,
                    static_cast<std::uint64_t>(std::numeric_limits<int>::max())));

            MPI_Bcast(&text[static_cast<std::size_t>(offset)],
                      this_count,
                      MPI_CHAR,
                      root,
                      comm);

            offset += static_cast<std::uint64_t>(this_count);
        }
    }

    static inline void read_text_file_mpi(const std::string &filename,
                                          MPI_Comm comm,
                                          std::string &text_out)
    {
        int rank = 0;
        MPI_Comm_rank(comm, &rank);

        std::string text;
        std::string error_message;
        int ok = 1;

        if (rank == 0)
        {
            try
            {
                std::ifstream in(filename.c_str(), std::ios::in | std::ios::binary);
                if (!in)
                    throw std::runtime_error("Cannot open file: " + filename);

                std::ostringstream buffer;
                buffer << in.rdbuf();
                text = buffer.str();
            }
            catch (const std::exception &exc)
            {
                ok = 0;
                error_message = exc.what();
            }
            catch (...)
            {
                ok = 0;
                error_message = "Unknown error while reading file: " + filename;
            }
        }

        MPI_Bcast(&ok, 1, MPI_INT, 0, comm);
        if (!ok)
        {
            broadcast_string_mpi(error_message, comm);
            throw std::runtime_error(error_message);
        }

        broadcast_string_mpi(text, comm);

        text_out.swap(text);
    }

}

#endif //READER_HELPER_FUNC_H
