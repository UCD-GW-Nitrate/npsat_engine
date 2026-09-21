//
// Created by giorgk on 6/23/26.
//

#ifndef TIMESERIES_H
#define TIMESERIES_H

#include <vector>
#include <tuple>

#include "flow_structures.h"
#include "reader_helper_func.h"

namespace npsat_flow{

    using namespace dealii;

    enum class TimeExtrapolation
    {
        CONST,      // Use first/last value for times outside range
        CYCLE      // Cycle through time data periodically
    };

    enum class TimeInterpolation
    {
        NEAREST,
        LINEAR
    };

    template <typename T = double>
    class TimeSeriesData{
        public:
        TimeSeriesData() = default;

        void read_data(const std::string &filename, MPI_Comm comm = MPI_COMM_WORLD) {
            // Fill owned storage_ and get a view into it
            const MatrixView<T> view = read_2d_array_mpi_as_view<T>(filename, comm, data_storage_);

            // Store as const-view internally (TimeSeriesData does not modify data values)
            data_ = MatrixView<const T>{view.data, view.nrows, view.ncols};

            // Data changed => revalidate on next query
            is_validated_ = false;
        }

        void set_data_owned(std::vector<T> &&flat, std::int64_t nrows, std::int64_t ncols)
        {
            data_storage_ = std::move(flat);
            data_ = MatrixView<const T>{data_storage_.data(), nrows, ncols};
            is_validated_ = false;
        }

        // Convenience: accept vector-of-vectors and flatten internally (still owned)
        void set_data_owned(std::vector<std::vector<T>> &&rows)
        {
            const std::int64_t nrows = static_cast<std::int64_t>(rows.size());
            const std::int64_t ncols = (nrows == 0) ? 0 : static_cast<std::int64_t>(rows[0].size());

            // Check rectangularity
            for (std::int64_t r = 0; r < nrows; ++r)
            {
                if (static_cast<std::int64_t>(rows[static_cast<std::size_t>(r)].size()) != ncols)
                {
                    throw std::invalid_argument("set_data_owned(rows): non-rectangular data.");
                }
            }

            std::vector<T> flat;
            flat.resize(static_cast<std::size_t>(nrows * ncols));
            for (std::int64_t r = 0; r < nrows; ++r)
            {
                for (std::int64_t c = 0; c < ncols; ++c)
                    flat[static_cast<std::size_t>(r * ncols + c)] =
                        rows[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)];
            }

            set_data_owned(std::move(flat), nrows, ncols);
        }

        // Advanced: external view (NO ownership). Caller must keep backing storage alive.
        void set_data_view(MatrixView<const T> view)
        {
            data_storage_.clear();
            data_storage_.shrink_to_fit(); // optional; remove if you dislike shrink
            data_ = view;
            is_validated_ = false;
        }

        void set_time(std::vector<double> time_points)
        {
            time_points_ = std::move(time_points);
            is_validated_ = false;
        }

        void set_extrapolation(TimeExtrapolation method)
        {
            extrapolation_method_ = method;
            is_validated_ = false;
        }

        void set_interpolation(TimeInterpolation method)
        {
            interpolation_method_ = method;
            is_validated_ = false;
        }

        void set_cycle_period(double cycle_period)
        {
            cycle_period_ = cycle_period;
            is_validated_ = false;
        }

        T get_value_at_step(int point_index, const int time_step_number) const
        {
            // --- Basic sanity checks
            AssertThrow(point_index >= 0, ExcMessage("point_index must be non-negative."));
            AssertThrow(time_step_number >= 0, ExcMessage("time_step_number must be non-negative."));
            const int n_points = data_.nrows;   // adjust if needed
            const int n_steps  = data_.ncols;
            AssertThrow(n_steps > 0, ExcMessage("TimeSeriesData has zero time steps (no columns)."));
            AssertThrow(point_index < n_points,
                        ExcMessage("point_index out of bounds: " + std::to_string(point_index) +
                                   " (n_points=" + std::to_string(n_points) + ")"));

            // --- Convert arbitrary requested time_step_number -> valid column index
            int t = time_step_number;
            if (t >= n_steps)
            {
                switch (extrapolation_method_)
                {
                    case TimeExtrapolation::CYCLE:
                    {
                        // Wrap: e.g. n_steps=8, t=54 -> 54 % 8 = 6
                        t = t % n_steps;
                        break;
                    }
                    case TimeExtrapolation::CONST:
                        default:
                    {
                        t = n_steps - 1;
                        break;
                    }
                }
            }
            return data_(point_index, t);
        }

        // Query API
        T get_value(unsigned int point_index, double time)
        {
            ensure_validated_();

            const std::int64_t row = static_cast<std::int64_t>(point_index);
            if (row < 0 || row >= data_.nrows)
                throw std::out_of_range("get_value(): point_index out of range.");

            if (time_points_.size() == 1)
                return data_(row, 0);

            const double adjusted_time = adjust_time_for_extrapolation_(time);

            unsigned int lower_idx;
            unsigned int upper_idx;
            double weight;
            bool exact_match;
            std::tie(lower_idx, upper_idx, weight, exact_match) = find_time_indices_(adjusted_time);

            return interpolate_at_indices_(row, lower_idx, upper_idx, weight, exact_match);
        }

        void get_values(std::vector<T> &values,
                    const std::vector<unsigned int> &row_indices,
                    double time)
        {
            ensure_validated_();

            values.clear();
            values.reserve(row_indices.size());

            if (time_points_.size() == 1)
            {
                for (unsigned int rui : row_indices)
                {
                    const std::int64_t r = static_cast<std::int64_t>(rui);
                    if (r < 0 || r >= data_.nrows)
                        throw std::out_of_range("get_values(): row index out of range.");
                    values.push_back(data_(r, 0));
                }
                return; // IMPORTANT: do not continue to interpolation logic
            }

            const double adjusted_time = adjust_time_for_extrapolation_(time);

            unsigned int lower_idx;
            unsigned int upper_idx;
            double weight;
            bool exact_match;
            std::tie(lower_idx, upper_idx, weight, exact_match) = find_time_indices_(adjusted_time);

            for (unsigned int rui : row_indices)
            {
                const std::int64_t r = static_cast<std::int64_t>(rui);
                if (r < 0 || r >= data_.nrows)
                    throw std::out_of_range("get_values(): row index out of range.");

                values.push_back(interpolate_at_indices_(r, lower_idx, upper_idx, weight, exact_match));
            }
        }

        void multiply_by_factor(const T factor)
        {
            if (data_.empty())
                return;

            // If we own the storage, modify directly
            if (!data_storage_.empty())
            {
                for (auto &v : data_storage_)
                    v *= factor;

                // Refresh view in case storage moved/reallocated previously
                data_ = MatrixView<const T>{
                    data_storage_.data(),
                    data_.nrows,
                    data_.ncols
                };
            }
            else
            {
                // Non-owned view cannot be modified safely
                throw std::runtime_error(
                    "TimeSeriesData::multiply_by_factor(): "
                    "cannot modify external const MatrixView data.");
            }
        }

        std::int64_t n_points() const { return data_.nrows; }
        std::int64_t n_times()  const { return data_.ncols; }


        private:
        std::vector<T>      data_storage_;
        MatrixView<const T> data_{};

        std::vector<double> time_points_;

        TimeInterpolation interpolation_method_ = TimeInterpolation::LINEAR;
        TimeExtrapolation extrapolation_method_ = TimeExtrapolation::CONST;

        double cycle_period_ = 0.0;

        static constexpr double tolerance_ = 1e-6;

        bool is_validated_ = false;

        void ensure_validated_()
        {
            if (!is_validated_)
            {
                validate_data_();
                is_validated_ = true;
            }
        }

        double adjust_time_for_extrapolation_(double time) const
        {
            if (time_points_.size() < 2)
                return time;

            const double t_min = time_points_.front();
            const double t_max = time_points_.back();

            if (time >= t_min && time <= t_max)
                return time;

            switch (extrapolation_method_)
            {
                case TimeExtrapolation::CONST:
                    return (time < t_min) ? t_min : t_max;

                case TimeExtrapolation::CYCLE:
                    return map_time_to_cycle_(time, t_min);

                default:
                    return time;
            }
        }

        double map_time_to_cycle_(double time, double t_min) const
        {
            // Preconditions validated in validate_data_(): cycle_period_ > 0
            const double time_from_start = time - t_min;
            const double cycles_passed = std::floor(time_from_start / cycle_period_);
            const double time_in_cycle = time_from_start - cycles_passed * cycle_period_;
            return t_min + time_in_cycle;
        }

        std::tuple<unsigned int, unsigned int, double, bool> find_time_indices_(double time) const
        {
            if (time <= time_points_.front())
                return std::make_tuple(0, 0, 0.0, true);

            if (time >= time_points_.back())
            {
                const unsigned int last = static_cast<unsigned int>(time_points_.size() - 1);
                return std::make_tuple(last, last, 0.0, true);
            }

            auto it = std::lower_bound(time_points_.begin(), time_points_.end(), time);

            if (it != time_points_.end() && std::abs(*it - time) < tolerance_)
            {
                const unsigned int idx = static_cast<unsigned int>(std::distance(time_points_.begin(), it));
                return std::make_tuple(idx, idx, 0.0, true);
            }

            if (it == time_points_.begin())
                return std::make_tuple(0, 0, 0.0, true);

            if (it == time_points_.end())
            {
                const unsigned int last = static_cast<unsigned int>(time_points_.size() - 1);
                return std::make_tuple(last, last, 0.0, true);
            }

            const unsigned int upper_idx = static_cast<unsigned int>(std::distance(time_points_.begin(), it));
            const unsigned int lower_idx = upper_idx - 1;

            const double denom = (time_points_[upper_idx] - time_points_[lower_idx]);
            const double weight = (denom == 0.0) ? 0.0 : (time - time_points_[lower_idx]) / denom;

            return std::make_tuple(lower_idx, upper_idx, weight, false);
        }

        T interpolate_at_indices_(std::int64_t row_idx,
                              unsigned int lower_idx,
                              unsigned int upper_idx,
                              double weight,
                              bool exact_match) const
        {
            if (exact_match || interpolation_method_ == TimeInterpolation::NEAREST)
            {
                if (exact_match || weight < 0.5)
                    return data_(row_idx, static_cast<std::int64_t>(lower_idx));
                else
                    return data_(row_idx, static_cast<std::int64_t>(upper_idx));
            }

            const T lower_val = data_(row_idx, static_cast<std::int64_t>(lower_idx));
            const T upper_val = data_(row_idx, static_cast<std::int64_t>(upper_idx));
            return static_cast<T>(lower_val * (1.0 - weight) + upper_val * weight);
        }

        void validate_data_() const
        {
            if (data_.empty())
                throw std::runtime_error("TimeSeriesData: data is empty. Call set_data_*() first.");

            if (time_points_.empty())
                throw std::runtime_error("TimeSeriesData: time_points is empty. Call set_time() first.");

            if (data_.ncols != static_cast<std::int64_t>(time_points_.size()))
            {
                throw std::invalid_argument(
                    "TimeSeriesData: #time_points (" + std::to_string(time_points_.size()) +
                    ") does not match #columns in data (" + std::to_string(data_.ncols) + ").");
            }

            if (!std::is_sorted(time_points_.begin(), time_points_.end()))
                throw std::invalid_argument("TimeSeriesData: time_points must be sorted ascending.");

            for (std::size_t i = 1; i < time_points_.size(); ++i)
            {
                if (time_points_[i] - time_points_[i - 1] < tolerance_)
                {
                    throw std::invalid_argument(
                        "TimeSeriesData: time_points must be distinct (within tolerance).");
                }
            }

            if (extrapolation_method_ == TimeExtrapolation::CYCLE)
            {
                if (cycle_period_ <= 0.0)
                {
                    throw std::invalid_argument(
                        "TimeSeriesData: CYCLE extrapolation requires cycle_period > 0.");
                }
            }
        }



    };


}

#endif //TIMESERIES_H
