#ifndef NPSAT_FLOW_OPTIONS_H
#define NPSAT_FLOW_OPTIONS_H

#include <boost/program_options.hpp>
#include "shared_sources.h"
#include <map>

namespace npsat_input {
namespace po = boost::program_options;

// Typed INI schema plus a small adapter for the existing AquiferProperties loader.
// The legacy labels are private adapter keys, never an alternative input grammar.
class FlowOptions {
public:
    po::options_description schema;
    po::variables_map values;
    std::map<std::string, std::string> legacy_keys;
    std::map<std::string, char> types;
    std::string input_path, output_path;
    SharedSources sources;

    FlowOptions() : schema("Steady FEM flow configuration") { declare_options(); }
    template<class T>
    T value(const std::string &key) const { return values.at(key).as<T>(); }
    void enter_subsection(const std::string &) {}
    void leave_subsection() {}
    std::string get(const std::string &legacy) const {
        if (legacy == "a Input directory") return input_path;
        if (legacy == "b Output directory") return output_path;
        // Source geometry/rate adapters are applied after common initialization.
        if (legacy == "b Stream recharge" || legacy == "c Wells") return "";
        if (legacy == "b Vertical discretization" && value<std::string>("Discretization.Vertical").empty()) {
            const int cells = static_cast<int>(list(value<std::string>("Discretization.Nxyz"), "Nxyz").back());
            std::ostringstream out;
            out << std::setprecision(17);
            for (int i = 0; i <= cells; ++i) {
                if (i) out << ',';
                out << static_cast<double>(i)/cells;
            }
            return out.str();
        }
        if (legacy == "a XYZ dimensions" || legacy == "b Left lower point" ||
            legacy == "a Nxyz" || legacy == "b Vertical discretization") {
            const std::vector<double> numbers = list(value<std::string>(legacy_keys.at(legacy)), legacy);
            std::ostringstream out;
            out << std::setprecision(17);
            for (unsigned i = 0; i < numbers.size(); ++i) { if (i) out << ','; out << numbers[i]; }
            return out.str();
        }
        return value<std::string>(legacy_keys.at(legacy));
    }
    int get_integer(const std::string &key) const { return value<int>(legacy_keys.at(key)); }
    double get_double(const std::string &key) const { return value<double>(legacy_keys.at(key)); }
    void add_string(const char *old, const char *key, const char *initial) {
        legacy_keys[old] = key; types[key] = 's';
        schema.add_options()(key, po::value<std::string>()->default_value(initial), old);
    }
    void add_int(const char *old, const char *key, int initial) {
        legacy_keys[old] = key; types[key] = 'i';
        schema.add_options()(key, po::value<int>()->default_value(initial), old);
    }
    void add_double(const char *old, const char *key, double initial) {
        legacy_keys[old] = key; types[key] = 'd';
        schema.add_options()(key, po::value<double>()->default_value(initial), old);
    }
    void declare_options();
    void read(const std::string &file, MPI_Comm comm) {
        std::string text;
        npsat_flow::read_text_file_mpi(file, comm, text);
        std::istringstream in(text);
        po::store(po::parse_config_file(in, schema, false), values);
        po::notify(values);
        validate();
        const std::string main = value<std::string>("Paths.Main");
        input_path = npsat_flow::resolve_relative_path(main, value<std::string>("Paths.Input"));
        output_path = npsat_flow::resolve_relative_path(main, value<std::string>("Paths.Output"));
        require(!input_path.empty() && !output_path.empty(), "Resolved input/output paths must not be empty");
        require(!input_path.empty() && !output_path.empty(), "Resolved input/output paths must not be empty");
        if (input_path.back() != '/') input_path += '/';
        if (output_path.back() != '/') output_path += '/';
        const int step = value<int>("Simulation.Start_step");
        if (!value<std::string>("Sources.Well_Data").empty())
            sources.read_wells(path("Sources.Well_Data"), path("Sources.Well_Rates"), step, comm);
        if (!value<std::string>("Sources.Stream_Data").empty())
            sources.read_streams(path("Sources.Stream_Data"), path("Sources.Stream_Rates"), step, comm);
    }
    std::string path(const std::string &key) const {
        return npsat_flow::resolve_relative_path(input_path, value<std::string>(key));
    }
    static std::vector<double> list(std::string text, const std::string &key) {
        npsat_flow::normalize_separators(text);
        std::istringstream in(text);
        std::vector<double> out;
        double number;
        while (in >> number) {
            require(std::isfinite(number), key + ": nonfinite number");
            out.push_back(number);
        }
        require(in.eof(), key + ": invalid numeric list");
        return out;
    }
    void validate() const {
        for (const auto &entry : types) {
            if (entry.second == 'd') require(std::isfinite(value<double>(entry.first)), entry.first + ": nonfinite value");
            if (entry.second == 'i') require(value<int>(entry.first) >= 0, entry.first + ": must be nonnegative");
        }
        for (const char *key : {"Paths.Main", "Paths.Input", "Paths.Output", "Geometry.Top", "Geometry.Bottom",
                               "Properties.Kxy", "FEM.Porosity", "Sources.Rch_Data", "BC.Dirichlet", "Output.Prefix"})
            require(!value<std::string>(key).empty(), std::string(key) + " must not be empty");
        require(value<int>("Simulation.Steady_state") == 1, "v1 only supports Simulation.Steady_state=1");
        require(value<int>("Simulation.Nsteps") >= 1, "Simulation.Nsteps must be positive (metadata only)");
        require(value<int>("Simulation.Confined") <= 1, "Simulation.Confined must be 0 or 1");
        require(value<int>("FEM.MeshIterations") >= 1, "FEM.MeshIterations must be positive");
        require(value<int>("Simulation.Confined") == 0 || value<int>("FEM.MeshIterations") == 1,
                "Confined v1 flow requires FEM.MeshIterations=1 (fixed mesh)");
        require(value<double>("Solver.System_tol") > 0 && value<int>("Solver.System_iterations") > 0,
                "Solver tolerance and iteration limit must be positive");
        require(value<std::string>("BC.GHB").empty(), "BC.GHB assembly is not implemented; milestone 2");
        require(value<std::string>("FEM.WellModel") == "legacy" || value<std::string>("FEM.WellModel") == "mnw",
                "FEM.WellModel must be legacy or mnw");
        for (const char *kind : {"Well", "Stream"}) {
            const std::string prefix = std::string("Sources.") + kind;
            require(value<std::string>(prefix + "_Data").empty() == value<std::string>(prefix + "_Rates").empty(),
                    prefix + "_Data and _Rates must be supplied together");
        }
        const std::string geometry = value<std::string>("Geometry.Type");
        require(geometry == "BOX" || geometry == "FILE", "Geometry.Type must be BOX or FILE");
        if (geometry == "FILE") require(!value<std::string>("Geometry.MeshFileName").empty(), "Missing mesh filename");
        for (const char *key : {"Geometry.BoxDims", "Geometry.BoxLLP", "Discretization.Nxyz"}) {
            const std::vector<double> numbers = list(value<std::string>(key), key);
            require(numbers.size() == 3, std::string(key) + " requires three entries");
            for (double x : numbers) {
                if (std::string(key) == "Geometry.BoxDims") require(x > 0, "Box dimensions must be positive");
                if (std::string(key) == "Discretization.Nxyz")
                    require(x >= 1 && x < std::numeric_limits<int>::max() && x == std::floor(x),
                            "INI Nxyz entries are cell counts and must be positive integers");
            }
        }
        const std::string vertical = value<std::string>("Discretization.Vertical");
        if (!vertical.empty()) {
            const std::vector<double> z = list(vertical, "Discretization.Vertical");
            require(z.size() >= 2 && z.front() == 0 && z.back() == 1, "Vertical list must run from 0 to 1");
            for (unsigned i = 1; i < z.size(); ++i) require(z[i] > z[i-1], "Vertical list must increase strictly");
            if (geometry == "BOX")
                require(z.size() == static_cast<std::size_t>(list(value<std::string>("Discretization.Nxyz"), "Nxyz").back()) + 1,
                        "Vertical list must contain Nz+1 elevations for the shared cell-count convention");
        }
        require(value<double>("FEM.TopFraction") >= 0 && value<double>("FEM.BottomFraction") >= 0 &&
                value<double>("FEM.TopFraction") + value<double>("FEM.BottomFraction") <= 1,
                "Invalid refinement fractions");
        require(value<double>("FEM.MinElementSize") > 0, "Minimum element size must be positive");
        require(value<double>("BC.HalfWidth") > 0 && value<double>("BC.MinOverlap") >= 0,
                "Invalid lateral matching tolerances");
        for (const char *key : {"FEM.XYThreshold", "FEM.ZThreshold", "FEM.TopRadius", "FEM.BottomRadius",
                               "FEM.TopPower", "FEM.BottomPower"})
            require(value<double>(key) > 0, std::string(key) + " must be positive");
    }
    void print(std::ostream &out) const {
        out << "INPUT_SCHEMA=npsat-flow-1\n" << std::setprecision(17);
        for (const auto &entry : types) {
            out << entry.first << '=';
            if (entry.second == 's') out << value<std::string>(entry.first);
            else if (entry.second == 'i') out << value<int>(entry.first);
            else out << value<double>(entry.first);
            out << '\n';
        }
        out << "Resolved.Input=" << input_path << "\nResolved.Output=" << output_path << '\n';
        for (const auto &w : sources.wells)
            out << "Selected.Well=" << w.id << ',' << w.row << ',' << w.rate << '\n';
        for (unsigned i = 0; i < sources.streams.size(); ++i)
            out << "Selected.Stream=" << i << ',' << sources.streams[i].row << ',' << sources.streams[i].rate << '\n';
    }
};

// Schema definitions are kept separate to make the old-to-new mapping reviewable.
#include "flow_option_schema.h"
} // namespace npsat_input
#endif
