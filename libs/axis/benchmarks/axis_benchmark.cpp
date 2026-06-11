// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

/// @file benchmarks/axis_benchmark.cpp
/// @brief Performance benchmark harness for AXIS CI regression detection.
///
/// Benchmark cases (Requirement 13.1):
///   1. bilinear_O48_to_O96       — Bilinear interpolation O48→O96
///   2. conservative_O48_to_O96   — Conservative 1st-order O48→O96
///   3. batch_apply_10vars_O96    — Batch apply 10 variables O96→O96
///   4. csr_apply_O96             — CSR-format apply O96→O96
///
/// All benchmarks use Kokkos::HostSpace for reproducible timing across CI
/// environments (Requirement 13.5).
///
/// Output: JSON array of objects with {name, wall_clock_ms, date, commit}
/// (Requirement 13.3).

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include <axis/solver/apply.hpp>
#include <axis/solver/interpolation_matrix.hpp>
#include <axis/solver/regrid_config.hpp>
#include <axis/solver/weight_generator.hpp>
#include <axis/topology/named_grid_registry.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark result structure (Requirement 13.3)
// ─────────────────────────────────────────────────────────────────────────────

struct BenchmarkResult {
    std::string name;
    double      wall_clock_ms;
    std::string date;
    std::string commit;
};

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Get current UTC date as ISO 8601 string.
std::string current_date_utc() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm utc_tm{};
    gmtime_r(&time_t_now, &utc_tm);
    std::ostringstream oss;
    oss << std::put_time(&utc_tm, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

/// Write results as JSON to a stream (Requirement 13.3).
void write_json(std::ostream& os, const std::vector<BenchmarkResult>& results) {
    os << "[\n";
    for (std::size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        os << "  {\n"
           << "    \"name\": \"" << r.name << "\",\n"
           << "    \"wall_clock_ms\": " << std::fixed << std::setprecision(3) << r.wall_clock_ms << ",\n"
           << "    \"date\": \"" << r.date << "\",\n"
           << "    \"commit\": \"" << r.commit << "\"\n"
           << "  }";
        if (i + 1 < results.size()) os << ",";
        os << "\n";
    }
    os << "]\n";
}

/// High-resolution wall-clock timer.
class Timer {
public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    void stop()  { end_   = std::chrono::high_resolution_clock::now(); }

    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(end_ - start_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
    std::chrono::high_resolution_clock::time_point end_;
};

/// Number of warmup iterations before timing.
constexpr int WARMUP_ITERS = 2;
/// Number of timed iterations to average.
constexpr int BENCH_ITERS  = 5;

/// Fill a Kokkos view with a smooth test field (cosine bell pattern).
void fill_test_field(Kokkos::View<double*, Kokkos::HostSpace>& field, std::size_t n) {
    Kokkos::resize(field, n);
    for (std::size_t i = 0; i < n; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(n);
        field(i) = std::cos(2.0 * M_PI * t) * std::sin(M_PI * t);
    }
}

/// Fill a rank-2 Kokkos view with test data for batch apply.
void fill_test_field_2d(Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace>& field,
                        std::size_t n_cells, std::size_t n_vars) {
    Kokkos::resize(field, n_cells, n_vars);
    for (std::size_t v = 0; v < n_vars; ++v) {
        for (std::size_t i = 0; i < n_cells; ++i) {
            double t = static_cast<double>(i) / static_cast<double>(n_cells);
            field(i, v) = std::cos(2.0 * M_PI * t + static_cast<double>(v)) *
                          std::sin(M_PI * t);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark cases (Requirement 13.1)
// ─────────────────────────────────────────────────────────────────────────────

/// Case 1: Bilinear O48→O96
BenchmarkResult bench_bilinear_O48_to_O96(const std::string& commit, const std::string& date) {
    using MS = Kokkos::HostSpace;

    // Generate named grids
    auto src_mesh = axis::topology::NamedGridRegistry::generate<MS>("O48");
    auto dst_mesh = axis::topology::NamedGridRegistry::generate<MS>("O96");

    // Configure bilinear regridding
    axis::solver::RegridConfig config;
    config.method   = axis::solver::InterpolationMethod::Bilinear;
    config.unmapped = axis::solver::UnmappedAction::Ignore;

    // Warmup: generate weights (includes BVH construction)
    axis::solver::InterpolationMatrix<MS> matrix;
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        matrix = axis::solver::WeightGenerator::generate<MS>(src_mesh, dst_mesh, config);
    }

    // Timed iterations: weight generation
    Timer timer;
    double total_ms = 0.0;
    for (int i = 0; i < BENCH_ITERS; ++i) {
        timer.start();
        matrix = axis::solver::WeightGenerator::generate<MS>(src_mesh, dst_mesh, config);
        Kokkos::fence("bench_bilinear_complete");
        timer.stop();
        total_ms += timer.elapsed_ms();
    }

    return BenchmarkResult{
        "bilinear_O48_to_O96",
        total_ms / BENCH_ITERS,
        date,
        commit
    };
}

/// Case 2: Conservative O48→O96
BenchmarkResult bench_conservative_O48_to_O96(const std::string& commit, const std::string& date) {
    using MS = Kokkos::HostSpace;

    auto src_mesh = axis::topology::NamedGridRegistry::generate<MS>("O48");
    auto dst_mesh = axis::topology::NamedGridRegistry::generate<MS>("O96");

    axis::solver::RegridConfig config;
    config.method    = axis::solver::InterpolationMethod::Conservative1stOrder;
    config.norm_type = axis::solver::NormType::DstArea;
    config.unmapped  = axis::solver::UnmappedAction::Ignore;

    // Warmup
    axis::solver::InterpolationMatrix<MS> matrix;
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        matrix = axis::solver::WeightGenerator::generate<MS>(src_mesh, dst_mesh, config);
    }

    // Timed iterations
    Timer timer;
    double total_ms = 0.0;
    for (int i = 0; i < BENCH_ITERS; ++i) {
        timer.start();
        matrix = axis::solver::WeightGenerator::generate<MS>(src_mesh, dst_mesh, config);
        Kokkos::fence("bench_conservative_complete");
        timer.stop();
        total_ms += timer.elapsed_ms();
    }

    return BenchmarkResult{
        "conservative_O48_to_O96",
        total_ms / BENCH_ITERS,
        date,
        commit
    };
}

/// Case 3: Batch apply 10 variables O96→O96
BenchmarkResult bench_batch_apply_10vars_O96(const std::string& commit, const std::string& date) {
    using MS = Kokkos::HostSpace;

    auto src_mesh = axis::topology::NamedGridRegistry::generate<MS>("O96");
    auto dst_mesh = axis::topology::NamedGridRegistry::generate<MS>("O96");

    axis::solver::RegridConfig config;
    config.method   = axis::solver::InterpolationMethod::Bilinear;
    config.unmapped = axis::solver::UnmappedAction::Ignore;

    // Generate weights once (not timed — we're benchmarking apply)
    auto matrix = axis::solver::WeightGenerator::generate<MS>(src_mesh, dst_mesh, config);

    const std::size_t n_src  = matrix.n_src();
    const std::size_t n_dst  = matrix.n_dst();
    const std::size_t n_vars = 10;

    // Prepare source and destination fields
    Kokkos::View<double**, Kokkos::LayoutLeft, MS> src_data("src", n_src, n_vars);
    Kokkos::View<double**, Kokkos::LayoutLeft, MS> dst_data("dst", n_dst, n_vars);
    fill_test_field_2d(src_data, n_src, n_vars);

    // Create field_views over the Kokkos data
    axis::field_view<const double, 2> src_view(src_data.data(), n_src, n_vars);
    axis::field_view<double, 2>       dst_view(dst_data.data(), n_dst, n_vars);

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        axis::solver::batch_apply(matrix, src_view, dst_view);
    }

    // Timed iterations: batch apply only
    Timer timer;
    double total_ms = 0.0;
    for (int i = 0; i < BENCH_ITERS; ++i) {
        timer.start();
        axis::solver::batch_apply(matrix, src_view, dst_view);
        Kokkos::fence("bench_batch_apply_complete");
        timer.stop();
        total_ms += timer.elapsed_ms();
    }

    return BenchmarkResult{
        "batch_apply_10vars_O96_to_O96",
        total_ms / BENCH_ITERS,
        date,
        commit
    };
}

/// Case 4: CSR apply O96→O96
BenchmarkResult bench_csr_apply_O96(const std::string& commit, const std::string& date) {
    using MS = Kokkos::HostSpace;

    auto src_mesh = axis::topology::NamedGridRegistry::generate<MS>("O96");
    auto dst_mesh = axis::topology::NamedGridRegistry::generate<MS>("O96");

    axis::solver::RegridConfig config;
    config.method   = axis::solver::InterpolationMethod::Bilinear;
    config.unmapped = axis::solver::UnmappedAction::Ignore;

    // Generate weights and convert to CSR (not timed)
    auto matrix = axis::solver::WeightGenerator::generate<MS>(src_mesh, dst_mesh, config);
    matrix.to_csr();

    const std::size_t n_src = matrix.n_src();
    const std::size_t n_dst = matrix.n_dst();

    // Prepare source and destination fields
    Kokkos::View<double*, MS> src_data("src", n_src);
    Kokkos::View<double*, MS> dst_data("dst", n_dst);
    fill_test_field(src_data, n_src);

    axis::field_view<const double, 1> src_view(src_data.data(), n_src);
    axis::field_view<double, 1>       dst_view(dst_data.data(), n_dst);

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; ++i) {
        axis::solver::apply(matrix, src_view, dst_view);
    }

    // Timed iterations: CSR apply only
    Timer timer;
    double total_ms = 0.0;
    for (int i = 0; i < BENCH_ITERS; ++i) {
        timer.start();
        axis::solver::apply(matrix, src_view, dst_view);
        Kokkos::fence("bench_csr_apply_complete");
        timer.stop();
        total_ms += timer.elapsed_ms();
    }

    return BenchmarkResult{
        "csr_apply_O96_to_O96",
        total_ms / BENCH_ITERS,
        date,
        commit
    };
}

// ─────────────────────────────────────────────────────────────────────────────
// CLI argument parsing
// ─────────────────────────────────────────────────────────────────────────────

struct CliArgs {
    std::string commit = "unknown";
    std::string output_path;  // empty = stdout
};

CliArgs parse_args(int argc, char* argv[]) {
    CliArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--commit" && i + 1 < argc) {
            args.commit = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            args.output_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: axis_benchmarks [--commit HASH] [--output PATH]\n"
                      << "  --commit HASH   Git commit hash for result metadata\n"
                      << "  --output PATH   Write JSON results to file (default: stdout)\n";
            std::exit(0);
        }
    }
    return args;
}

} // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    // Parse CLI args before Kokkos (Kokkos consumes its own args)
    CliArgs cli = parse_args(argc, argv);

    Kokkos::initialize(argc, argv);
    {
        std::string date = current_date_utc();
        std::vector<BenchmarkResult> results;

        std::cout << "AXIS Benchmark Suite\n";
        std::cout << "====================\n";
        std::cout << "Commit: " << cli.commit << "\n";
        std::cout << "Date:   " << date << "\n";
        std::cout << "Warmup: " << WARMUP_ITERS << " iters, Timed: " << BENCH_ITERS << " iters\n";
        std::cout << "\n";

        // Run benchmark cases
        std::cout << "[1/4] bilinear_O48_to_O96 ..." << std::flush;
        auto r1 = bench_bilinear_O48_to_O96(cli.commit, date);
        std::cout << " " << std::fixed << std::setprecision(2) << r1.wall_clock_ms << " ms\n";
        results.push_back(r1);

        std::cout << "[2/4] conservative_O48_to_O96 ..." << std::flush;
        auto r2 = bench_conservative_O48_to_O96(cli.commit, date);
        std::cout << " " << std::fixed << std::setprecision(2) << r2.wall_clock_ms << " ms\n";
        results.push_back(r2);

        std::cout << "[3/4] batch_apply_10vars_O96_to_O96 ..." << std::flush;
        auto r3 = bench_batch_apply_10vars_O96(cli.commit, date);
        std::cout << " " << std::fixed << std::setprecision(2) << r3.wall_clock_ms << " ms\n";
        results.push_back(r3);

        std::cout << "[4/4] csr_apply_O96_to_O96 ..." << std::flush;
        auto r4 = bench_csr_apply_O96(cli.commit, date);
        std::cout << " " << std::fixed << std::setprecision(2) << r4.wall_clock_ms << " ms\n";
        results.push_back(r4);

        std::cout << "\nDone.\n";

        // Output JSON results (Requirement 13.3)
        if (cli.output_path.empty()) {
            std::cout << "\n--- JSON Results ---\n";
            write_json(std::cout, results);
        } else {
            std::ofstream ofs(cli.output_path);
            if (!ofs) {
                std::cerr << "ERROR: Cannot open output file: " << cli.output_path << "\n";
                Kokkos::finalize();
                return 1;
            }
            write_json(ofs, results);
            std::cout << "Results written to: " << cli.output_path << "\n";
        }
    }
    Kokkos::finalize();
    return 0;
}
