//
// Created by giorgk on 6/22/2026.
//
#include <string>
#include <vector>
#include <cstdint>
#include <type_traits>
#include <deal.II/base/types.h>

#ifndef NPSAT_V2_FLOW_STRUCTURES_H
#define NPSAT_V2_FLOW_STRUCTURES_H

namespace npsat_flow {
    using namespace dealii;

    static const std::vector<unsigned int> face_order = {0, 1, 3, 2};

    struct Refinement_options
    {
        int initial = 0;
        int wells = 0;
        int streams = 0;
        int top = 0;
        int top_depth = 0;
        int dirichlet = 0;
        int GHB = 0;
        int neumann = 0;
        double well_min_screen_length = 0.5;
        double well_min_screen_cell_fraction = 0.1;
    };

    struct Sources_uo
    {
        std::string rch_file;
        double rch_factor = 1.0;

        std::string stream_file;
        std::string stream_rates_file;
        double stream_factor = 1.0;

        std::string well_file;
        std::string well_rates_file;
        double well_factor = 1.0;
    };

    struct Hydrogeology_options {
        std::string kx_file;
        std::string kz_file;
        std::string ss_file;
        std::string sy_file;
    };

    struct Time_step_uo
    {
        //std::string time_file;
        unsigned int n_sim_steps;
        unsigned int start_step;
    };

    struct BC_uo
    {
        std::string dirichlet_file;
        std::string ghb_file;
        std::string neumann_file;
        double half_with;
        double min_overlap;
    };

    struct Sim_uo {
        int n_steps = 1;
        int Start_step = 0;
        std::string delta_time_file;
        bool confined = false;
        // Steady mode solves exactly one stress scenario selected by Start_step.
        // Confined remains an independent option, giving the four combinations
        // confined/unconfined and steady/transient.
        bool steady_state = false;
        bool restart_from_checkpoint = false;
        std::string checkpoint_file = "npsat_flow.chk";
        std::string restart_checkpoint_file;
    };

    struct Spinup_uo {
        unsigned int iterations = 30;                    // hard maximum solves
        double tolerance = 1.0;                          // maximum-head guard [L]
        unsigned int minimum_solves = 5;
        double flux_relative_l2_tolerance = 1.0e-2;
        double rms_head_tolerance = 1.0e-1;              // [L]
        unsigned int consecutive_passes = 3;
        double pumping_loss_fraction_tolerance = 5.0e-3; // fraction of requested pumping
        double pumping_loss_stability_tolerance = 1.0e-4;// solve-to-solve fraction change
        bool exit_after_convergence = false;
    };

    struct Solver_uo {
        int System_iterations = 15000;
        double System_tol = 1e-12;
    };

    struct NonlinearControls
    {
        enum class RechargeStabilizationMode
        {
            HysteresisOnly,
            EffectiveTop
        };
        enum class EffectiveTopMode
        {
            Off,
            RechargeReceivers,
            AllWaterTableCells
        };

        unsigned int max_picard_iters = 25;
        double       abs_tol_update   = 1e-2;   // e.g. meters
        double       rel_tol_update   = 1e-4;
        double       damping_omega    = 0.7;    // Picard damping
        bool         use_anderson     = true;
        unsigned int anderson_start  = 2;      // after 1–2 Picard steps
        unsigned int anderson_m      = 5;      // memory depth <= 5
        double       anderson_reg    = 1e-11;  // small regularization for LS
        bool         carry_history_across_timesteps = false;
        bool         use_recharge_hysteresis = true;
        double       recharge_drying_saturated_fraction = 5.0e-3;
        double       recharge_wetting_saturated_fraction = 5.0e-2;
        double       recharge_min_relative_k = 1.0e-4;
        double anderson_beta = 0.7;
        double anderson_max_alpha = 5.0;
        double anderson_max_step_factor = 2.0;
        RechargeStabilizationMode recharge_stabilization_mode =
            RechargeStabilizationMode::EffectiveTop;
        EffectiveTopMode effective_top_mode = EffectiveTopMode::RechargeReceivers;
    };


    struct user_options
    {
        std::string main_path;
        std::string output_path;
        std::string checkpoint_folder;
        std::string input_path;
        std::string output_prefix;
        std::string initial_head_file;
        std::string log_file;
        int dry_well_log = 0;

        bool isBox = false;
        std::vector<double> box_dims;
        std::vector<double> box_llp;
        std::vector<unsigned int> box_nxyz;
        std::vector<double> vert_discr;
        std::string vert_file;
        std::string mesh_file;
        std::string top_fnc;
        std::string bot_fnc;

        bool print_matrices;
        bool print_initial_mesh = false;
        bool print_mesh_with_prop = false;
        bool print_mesh_exit = false;
        bool save_trace_data = false;
        bool save_fine_velocities = true;
        bool save_coarse_velocities = false;
        bool print_vtk = false;
        bool print_water_table = false;
        bool print_q_to_vtu = false;
        bool print_lambda_to_vtu = false;
        bool print_area_to_vtu = false;
        bool print_cell_budget_csv = false;
        bool print_solution_cell_vtu = false;
        bool print_cellcenters_vtk = false;
        bool print_cell_well_map_csv = false;
        bool print_trace_well_maps_csv = false;
        bool print_wellbore_segments_csv = false;
        bool print_wellbore_segments_vtu = false;
        bool print_wellbore_segments_legacy_vtk = false;
        bool print_wellboreflow_csv = false;
        bool print_well_resid_csv = false;

        int n_steps = 1;
        int start_step = 0;

        Refinement_options ref_opt;
        Hydrogeology_options hgeo;
        Sources_uo sources;

        Time_step_uo time_tracking;

        BC_uo bndr_cond;
        Sim_uo sim_opt;
        Spinup_uo spin_uo;
        Solver_uo solver_opt;
        NonlinearControls NLC;
        int verbose_level = 0;
        int assembly_progress_frequency = 10;
    };

    template <typename T>
    struct MatrixView
    {
        static_assert(std::is_arithmetic<T>::value, "MatrixView<T>: T must be arithmetic.");
        T *data = nullptr;
        std::int64_t nrows = 0;
        std::int64_t ncols = 0;

        inline T &operator()(std::int64_t i, std::int64_t j)
        {
            return data[static_cast<std::size_t>(i) * static_cast<std::size_t>(ncols) +
                        static_cast<std::size_t>(j)];
        }

        inline const T &operator()(std::int64_t i, std::int64_t j) const
        {
            return data[static_cast<std::size_t>(i) * static_cast<std::size_t>(ncols) +
                        static_cast<std::size_t>(j)];
        }
        inline bool empty() const { return (data == nullptr) || (nrows == 0) || (ncols == 0); }
    };

    struct ScreenClip
    {
        double z_low = 0.0;
        double z_high = 0.0;
        double L = 0.0;
    };

    struct WellRef
    {
        unsigned int well_id;    // global well DoF id (0..n_wells-1)
        unsigned int well_rank;  // owner rank of that well DoF
    };

    struct TraceRef
    {
        types::global_dof_index trace_id;   // global trace DoF id
        unsigned int            trace_rank; // owner rank of that trace DoF
    };

    struct TraceWellEdgeMsg
    {
        types::global_dof_index trace_id;
        unsigned int            well_id;
        unsigned int            well_rank; // owner of well_id

        template <class Archive>
        void serialize(Archive &ar, const unsigned int)
        { ar & trace_id; ar & well_id; ar & well_rank; }
    };

    struct WellTraceEdgeMsg
    {
        unsigned int            well_id;
        types::global_dof_index trace_id;
        unsigned int            trace_rank; // owner of trace_id

        template <class Archive>
        void serialize(Archive &ar, const unsigned int)
        { ar & well_id; ar & trace_id; ar & trace_rank; }
    };

    struct RechargeRouteRequest
    {
        std::uint64_t cell_gid;
        double source_area;
    };

    struct CellNonlinearData
    {
        double z_bot = 0.0;
        double z_top = 0.0;
        double assembly_z_top = 0.0;
        double thickness = 0.0;

        double h_e   = 0.0;   // representative head in cell
        double psi   = 0.0;   // h - z_bot

        double r     = 1.0;   // relative conductivity factor
        double S_eff = 0.0;   // effective storage coefficient for this cell (scalar)

        bool is_partially_saturated = false;
        bool is_fully_dry           = false;
        bool is_fully_saturated     = false;

        bool is_top_layer_cell      = false; // inferred from boundary top face
        bool uses_effective_top     = false; // nonlinear quantities use assembly_z_top instead of geometric z_top
    };

    using dof_t  = types::global_dof_index;
    using well_t = unsigned int;
    struct Well10Entry { well_t well; dof_t col; double val; }; // (1,0): row=well, col=trace/master
    struct Well11Entry { well_t well; double val; };            // (1,1): diagonal only in your code
    struct WellRhsEntry{ well_t well; double val; };            // rhs block(1)
    struct Trace01Entry
    {
        dof_t  row;   // trace row (master)
        well_t col;   // well column
        double val;
    };

    struct WellIdentityPartial
    {
        std::uint32_t well_id;      // global well index
        std::uint32_t count_cells;  // number of contributing cells on this rank
        double        sum_Qe;
        double        sum_cwc;
        double        sum_cwc_he;
    };
    static_assert(std::is_trivially_copyable<WellIdentityPartial>::value, "WellIdentityPartial must be trivially copyable for MPI_BYTE exchange.");

    // One well segment contribution coming from a cell on some rank.
    // We only need (well_id, ze, Qe) to reproduce the tested logic.
    struct WellSegmentMsg
    {
        std::uint32_t well_id; // global well index
        double        ze;      // ordering key (bottom->top)
        double        Qe;      // exchange for this segment (must be computed already)
    };

}

#endif //NPSAT_V2_FLOW_STRUCTURES_H
