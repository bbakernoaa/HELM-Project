/// @file example_bilinear_regrid.cpp
/// @brief Demonstrates basic bilinear interpolation between two grids.
///
/// This example:
/// 1. Creates source and destination meshes using the NamedGridRegistry
/// 2. Generates bilinear interpolation weights
/// 3. Creates a test field (constant = 1.0)
/// 4. Applies weights and verifies partition of unity

#include <Kokkos_Core.hpp>
#include <axis/axis.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>

int main(int argc, char *argv[]) {
    Kokkos::ScopeGuard kokkos(argc, argv);

    // =========================================================================
    // Step 1: Create source and destination meshes
    // =========================================================================
    auto src_mesh = axis::topology::NamedGridRegistry::generate("O48");
    auto dst_mesh = axis::topology::NamedGridRegistry::generate("O96");

    std::printf("Source mesh: %zu cells\n", static_cast<std::size_t>(src_mesh.n_cells()));
    std::printf("Destination mesh: %zu cells\n", static_cast<std::size_t>(dst_mesh.n_cells()));

    // =========================================================================
    // Step 2: Generate bilinear interpolation weights
    // =========================================================================
    axis::solver::RegridConfig config;
    config.method = axis::solver::RegridMethod::Bilinear;

    axis::solver::WeightGenerator gen;
    auto matrix = gen.generate(src_mesh, dst_mesh, config);

    std::printf("Weight matrix: nnz = %zu\n", static_cast<std::size_t>(matrix.nnz()));

    // =========================================================================
    // Step 3: Convert to CSR for efficient apply
    // =========================================================================
    matrix.to_csr();

    // =========================================================================
    // Step 4: Create a constant source field (f = 1.0)
    // =========================================================================
    Kokkos::View<double *> src_field("src_field", src_mesh.n_cells());
    Kokkos::deep_copy(src_field, 1.0);

    // =========================================================================
    // Step 5: Apply weights
    // =========================================================================
    Kokkos::View<double *> dst_field("dst_field", dst_mesh.n_cells());
    axis::solver::apply(matrix, src_field, dst_field);

    // =========================================================================
    // Step 6: Verify partition of unity (constant field preserved)
    // =========================================================================
    double max_error = 0.0;
    Kokkos::parallel_reduce(
        "check_partition_of_unity", dst_mesh.n_cells(),
        KOKKOS_LAMBDA(int i, double &err) {
            double e = Kokkos::abs(dst_field(i) - 1.0);
            if (e > err) err = e;
        },
        Kokkos::Max<double>(max_error));

    std::printf("Partition of unity max error: %.2e\n", max_error);

    if (max_error > 1e-12) {
        std::fprintf(stderr, "ERROR: Partition of unity violated!\n");
        return EXIT_FAILURE;
    }

    std::printf("SUCCESS: Bilinear regridding preserves constant field.\n");
    return EXIT_SUCCESS;
}
