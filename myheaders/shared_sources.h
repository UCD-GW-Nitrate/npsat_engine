#ifndef NPSAT_SHARED_SOURCES_H
#define NPSAT_SHARED_SOURCES_H

#include "reader_helper_func.h"
#include "TimeSeries.h"
#include <cmath>
#include <iomanip>
#include <set>

namespace npsat_input {
inline void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}
inline void finish_record(std::istream &in, const std::string &label) {
    require(!in.fail(), "Malformed " + label);
    in >> std::ws;
    require(in.eof(), "Unexpected trailing data in " + label);
}
inline double selected_rate(const npsat_flow::TimeSeriesData<double> &rates,
                            int row, int step, const std::string &label) {
    require(row >= 0 && row < rates.n_points(), label + ": invalid zero-based rate row");
    require(rates.n_times() == 1 || step < rates.n_times(), label + ": unavailable stress column");
    return rates.get_value_at_step(row, rates.n_times() == 1 ? 0 : step);
}
struct WellRecord {
    int id, row;
    double x, y, top, bottom, radius, skin_radius, skin_k, rate;
};
struct StreamRecord {
    int row;
    double width, rate;
    std::vector<std::pair<double, double>> vertices;
};
struct SharedSources {
    std::vector<WellRecord> wells;
    std::vector<StreamRecord> streams;

    void read_wells(const std::string &geometry, const std::string &rates_file,
                    int step, MPI_Comm comm) {
        npsat_flow::TimeSeriesData<double> rates;
        rates.read_data(rates_file, comm);
        std::string text;
        npsat_flow::read_text_file_mpi(geometry, comm, text);
        std::istringstream in(text);
        std::string line;
        require(static_cast<bool>(std::getline(in, line)), "Empty well geometry: " + geometry);
        // v2 consumes a header followed by Eid,x,y,top,bottom,q_row,rw,Rskin,Kskin.
        npsat_flow::normalize_separators(line);
        std::istringstream header(line);
        std::string token;
        const char *names[] = {"Eid", "x", "y", "top", "bottom", "q_row", "rw", "Rskin", "Kskin"};
        for (unsigned i = 0; i < 9; ++i) {
            require(static_cast<bool>(header >> token) && token == names[i], "Invalid well CSV header");
        }
        finish_record(header, "well header");
        std::set<int> ids;
        while (std::getline(in, line)) {
            if (npsat_flow::is_comment_or_empty_line(line)) continue;
            npsat_flow::normalize_separators(line);
            std::istringstream record(line);
            WellRecord w;
            record >> w.id >> w.x >> w.y >> w.top >> w.bottom >> w.row
                   >> w.radius >> w.skin_radius >> w.skin_k;
            finish_record(record, "well geometry");
            require(w.id >= 0 && ids.insert(w.id).second, "Duplicate or negative well Eid");
            require(std::isfinite(w.x) && std::isfinite(w.y) && std::isfinite(w.top) &&
                    std::isfinite(w.bottom) && std::isfinite(w.radius) &&
                    std::isfinite(w.skin_radius) && std::isfinite(w.skin_k), "Nonfinite well geometry");
            require(w.top > w.bottom && w.radius > 0 && w.skin_radius >= w.radius && w.skin_k > 0,
                    "Invalid well screen/radius/skin parameters");
            w.rate = selected_rate(rates, w.row, step, "well " + std::to_string(w.id));
            wells.push_back(w);
        }
        require(!wells.empty(), "Well file contains no wells");
    }

    void read_streams(const std::string &geometry, const std::string &rates_file,
                      int step, MPI_Comm comm) {
        npsat_flow::TimeSeriesData<double> rates;
        rates.read_data(rates_file, comm);
        std::string text;
        npsat_flow::read_text_file_mpi(geometry, comm, text);
        std::istringstream in(text);
        int count = 0;
        require(static_cast<bool>(in >> count) && count > 0, "Invalid stream count");
        for (int i = 0; i < count; ++i) {
            StreamRecord s;
            int n = 0;
            s.width = 0;
            require(static_cast<bool>(in >> n >> s.row) && n >= 2 && n <= 4, "Invalid stream record");
            if (n == 2) require(static_cast<bool>(in >> s.width) && std::isfinite(s.width) && s.width > 0,
                                "Invalid stream width");
            for (int j = 0; j < n; ++j) {
                double x, y;
                require(static_cast<bool>(in >> x >> y) && std::isfinite(x) && std::isfinite(y),
                        "Invalid stream vertex");
                s.vertices.push_back(std::make_pair(x, y));
            }
            double area2 = 0;
            for (int j = 0; j < n; ++j) {
                const std::pair<double,double> &a = s.vertices[j], &b = s.vertices[(j+1)%n];
                area2 += a.first*b.second - b.first*a.second;
            }
            require(n == 2 ? s.vertices[0] != s.vertices[1] : std::abs(area2) > 1e-14,
                    "Degenerate stream geometry");
            s.rate = selected_rate(rates, s.row, step, "stream " + std::to_string(i));
            streams.push_back(s);
        }
        finish_record(in, "stream geometry");
    }

    // Adapt already validated data to the legacy stream geometry constructor in memory.
    std::string legacy_stream_text() const {
        std::ostringstream out;
        out << std::setprecision(17) << streams.size() << '\n';
        for (const auto &s : streams) {
            out << s.vertices.size() << ' ' << s.rate;
            if (s.vertices.size() == 2) out << ' ' << s.width;
            out << '\n';
            for (const auto &p : s.vertices) out << p.first << ' ' << p.second << '\n';
        }
        return out.str();
    }
};
} // namespace npsat_input
#endif
