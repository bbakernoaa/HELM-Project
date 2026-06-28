/// @file example_batch_apply.cpp
/// @brief Demonstrates batch apply for multi-variable regridding.
///
/// This example:
/// 1. Creates an interpolation matrix
/// 2. Creates a multi-variable source field (rank-2 view)
/// 3. Applies weights to all variables in a single kernel launch
/// 4. Verifies results match individual apply calls

#include <Kokkos_Core.hpp>
#include <axis/axis.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>

int main(int argc, char *argv[]) {
    Kokkos::ScopeGuard kokkos(argc, argv);

    constexpr int N_VARS = 5;  // T, u, v, q, ps

    // =========================================================================
    // Step 1: Create meshes and generate weights
    // =========================================================================
    auto src_mesh = axis::topology::NamedGridRegistry::generate("O48");
    auto dst_mesh = axis::topology::NamedGridRegistry::generate("O96");

    axis::solver::RegridConfig config;
    config.method = axis::solver::RegridMethod::Bilinear;

    axis::solver::WeightGenerator gen;
    auto matrix = gen.generate(src_mesh, dst_mesh, config);
    matrix.to_csr();

    auto n_src = src_mesh.n_cells();
    auto n_dst = dst_mesh.n_cells();

    std::printf("Matrix: %zu src, %zu dst, %zu nnz, %d vars\n", static_cast<std::size_t>(n_src), static_cast<std::size_t>(n_dst),
                static_cast<std::size_t>(matrix.nnz()), N_VARS);

    // =========================================================================
    // Step 2: Create multi-variable source field [n_src, N_VARS]
    // =========================================================================
    Kokkos::View<double **> src_fields("src_fields", n_src, N_VARS);
    Kokkos::parallel_for(
        "init_src", n_src, KOKKOS_LAMBDA(int i) {
            for (int v = 0; v < N_VARS; ++v) {
                src_fields(i, v) = static_cast<double>(i * N_VARS + v) * 0.001;
            }
        });

    // =========================================================================
    // Step 3: Batch apply — single kernel for all variables
    // =========================================================================
    Kokkos::View<double **> dst_batch("dst_batch", n_dst, N_VARS);
    axis::solver::batch_apply(matrix, src_fields, dst_batch);

    // =========================================================================
    // Step 4: Verify against individual apply calls
    // =========================================================================
    double max_error = 0.0;

    for (int v = 0; v < N_VARS; ++v) {
        auto src_v = Kokkos::subview(src_fields, Kokkos::ALL, v);
        Kokkos::View<double *> dst_single("dst_single", n_dst);
        axis::solver::apply(matrix, src_v, dst_single);

        double var_error = 0.0;
        auto dst_batch_v = Kokkos::subview(dst_batch, Kokkos::ALL, v);
        Kokkos::parallel_reduce(
            "check_var", n_dst,
            KOKKOS_LAMBDA(int i, double &err) {
                double e = Kokkos::abs(dst_batch_v(i) - dst_single(i));
                if (e > err) err = e;
            },
            Kokkos::Max<double>(var_error));

        if (var_error > max_error) max_error = var_error;
    }

    std::printf("Batch vs individual apply max error: %.2e\n", max_error);

    if (max_error > 1e-14) {
        std::fprintf(stderr, "ERROR: Batch apply does not match individual apply!\n");
        return EXIT_FAILURE;
    }

    std::printf("SUCCESS: Batch apply matches individual apply for %d variables.\n", N_VARS);
    return EXIT_SUCCESS;
}
