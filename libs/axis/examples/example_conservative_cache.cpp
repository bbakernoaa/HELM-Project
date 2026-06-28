/// @file example_conservative_cache.cpp
/// @brief Demonstrates conservative remapping with weight caching.
///
/// This example:
/// 1. Generates conservative interpolation weights
/// 2. Serializes the weights to a binary buffer
/// 3. Deserializes and verifies bitwise-identical apply results
/// 4. Verifies conservation (integral preservation)

#include <Kokkos_Core.hpp>
#include <axis/axis.hpp>
#include <axis/solver/weight_cache.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char *argv[]) {
    Kokkos::ScopeGuard kokkos(argc, argv);

    // =========================================================================
    // Step 1: Create meshes and generate conservative weights
    // =========================================================================
    auto src_mesh = axis::topology::NamedGridRegistry::generate("O48");
    auto dst_mesh = axis::topology::NamedGridRegistry::generate("O96");

    axis::solver::RegridConfig config;
    config.method = axis::solver::RegridMethod::Conservative1stOrder;
    config.normalization = axis::solver::Normalization::DstArea;

    axis::solver::WeightGenerator gen;
    auto matrix = gen.generate(src_mesh, dst_mesh, config);

    std::printf("Generated conservative weights: nnz = %zu\n", static_cast<std::size_t>(matrix.nnz()));

    // =========================================================================
    // Step 2: Serialize weights to a buffer
    // =========================================================================
    auto buf_size = axis::solver::WeightCache::serialize(matrix, nullptr, 0);
    std::vector<char> cache(buf_size);
    axis::solver::WeightCache::serialize(matrix, cache.data(), cache.size());

    std::printf("Serialized to %zu bytes (%.1f KB)\n", buf_size, buf_size / 1024.0);

    // =========================================================================
    // Step 3: Deserialize and verify round-trip
    // =========================================================================
    auto matrix2 = axis::solver::WeightCache::deserialize<Kokkos::HostSpace>(cache.data(), cache.size());

    // Create a test field
    Kokkos::View<double *> src_field("src_field", src_mesh.n_cells());
    Kokkos::parallel_for("init_field", src_mesh.n_cells(), KOKKOS_LAMBDA(int i) { src_field(i) = static_cast<double>(i) * 0.01; });

    // Apply with original matrix
    Kokkos::View<double *> dst_a("dst_a", dst_mesh.n_cells());
    axis::solver::apply(matrix, src_field, dst_a);

    // Apply with deserialized matrix
    Kokkos::View<double *> dst_b("dst_b", dst_mesh.n_cells());
    axis::solver::apply(matrix2, src_field, dst_b);

    // Verify bitwise-identical results
    double max_diff = 0.0;
    Kokkos::parallel_reduce(
        "check_roundtrip", dst_mesh.n_cells(),
        KOKKOS_LAMBDA(int i, double &diff) {
            double d = Kokkos::abs(dst_a(i) - dst_b(i));
            if (d > diff) diff = d;
        },
        Kokkos::Max<double>(max_diff));

    std::printf("Cache round-trip max difference: %.2e\n", max_diff);

    if (max_diff > 0.0) {
        std::fprintf(stderr, "ERROR: Cache round-trip not bitwise identical!\n");
        return EXIT_FAILURE;
    }

    // =========================================================================
    // Step 4: Verify conservation
    // =========================================================================
    double src_integral = 0.0;
    Kokkos::parallel_reduce(
        "src_integral", src_mesh.n_cells(), KOKKOS_LAMBDA(int i, double &sum) { sum += src_field(i) * src_mesh.cell_area(i); }, src_integral);

    double dst_integral = 0.0;
    Kokkos::parallel_reduce(
        "dst_integral", dst_mesh.n_cells(), KOKKOS_LAMBDA(int i, double &sum) { sum += dst_a(i) * dst_mesh.cell_area(i); }, dst_integral);

    double conservation_error = std::abs(src_integral - dst_integral) / std::abs(src_integral);
    std::printf("Conservation error: %.2e\n", conservation_error);

    if (conservation_error > 1e-12) {
        std::fprintf(stderr, "ERROR: Conservation violated!\n");
        return EXIT_FAILURE;
    }

    std::printf("SUCCESS: Conservative remapping with caching verified.\n");
    return EXIT_SUCCESS;
}
