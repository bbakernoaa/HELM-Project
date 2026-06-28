// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

/// @file src/solver/gradient_reconstructor.cpp
/// @brief Least-squares gradient reconstruction with optional Barth-Jespersen limiter.
///
/// Implementation uses Kokkos::parallel_for over cells for both gradient
/// computation and limiter application. The 3×3 normal-equations solve uses
/// Cramer's rule (device-portable, no LAPACK dependency).

#include <Kokkos_Core.hpp>
#include <axis/solver/gradient_reconstructor.hpp>
#include <cmath>

namespace axis::solver {

namespace {

/// Singularity threshold for determinant check.
constexpr double SINGULAR_EPS = 1.0e-30;

/// @brief Solve 3×3 symmetric positive-definite system via Cramer's rule.
///
/// Solves (A^T A) g = A^T b where the normal matrix N = A^T A is symmetric.
/// Returns false if |det(N)| < eps (singular/degenerate).
///
/// @param[in]  n00..n22  Upper-triangle entries of the 3×3 normal matrix
/// @param[in]  r0,r1,r2  Right-hand side vector (A^T b)
/// @param[out] g0,g1,g2  Solution gradient components
/// @return true if solve succeeded, false if singular
KOKKOS_INLINE_FUNCTION
bool solve_3x3_cramer(double n00, double n01, double n02, double n11, double n12, double n22, double r0, double r1, double r2, double &g0, double &g1,
                      double &g2) noexcept {
    // Determinant of the symmetric matrix:
    // | n00 n01 n02 |
    // | n01 n11 n12 |
    // | n02 n12 n22 |
    const double det = n00 * (n11 * n22 - n12 * n12) - n01 * (n01 * n22 - n12 * n02) + n02 * (n01 * n12 - n11 * n02);

    if (Kokkos::fabs(det) < SINGULAR_EPS) {
        g0 = g1 = g2 = 0.0;
        return false;
    }

    const double inv_det = 1.0 / det;

    // Cramer's rule: replace each column of N with rhs
    // g0 = det(N with col0 replaced by rhs) / det(N)
    g0 = inv_det * (r0 * (n11 * n22 - n12 * n12) - n01 * (r1 * n22 - n12 * r2) + n02 * (r1 * n12 - n11 * r2));

    g1 = inv_det * (n00 * (r1 * n22 - n12 * r2) - r0 * (n01 * n22 - n12 * n02) + n02 * (n01 * r2 - r1 * n02));

    g2 = inv_det * (n00 * (n11 * r2 - r1 * n12) - n01 * (n01 * r2 - r1 * n02) + r0 * (n01 * n12 - n11 * n02));

    return true;
}

}  // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// GradientReconstructor::compute — explicit specialization for HostSpace
// ─────────────────────────────────────────────────────────────────────────────

template <class MemorySpace>
void GradientReconstructor<MemorySpace>::compute(Kokkos::View<const double *, MemorySpace> cell_values,
                                                 Kokkos::View<const double *[3], MemorySpace> centroids,
                                                 Kokkos::View<const index_t *, MemorySpace> adj_offsets,
                                                 Kokkos::View<const index_t *, MemorySpace> adj_indices, Kokkos::View<double *[3], MemorySpace> grad,
                                                 bool use_limiter) {
    using execution_space = typename MemorySpace::execution_space;

    const auto n_cells = static_cast<int>(cell_values.extent(0));

    // ── Phase 1: Compute raw least-squares gradients ────────────────────────
    Kokkos::parallel_for(
        "GradientReconstructor::compute_gradients", Kokkos::RangePolicy<execution_space>(0, n_cells), KOKKOS_LAMBDA(const int i) {
            const auto start = adj_offsets(i);
            const auto end = adj_offsets(i + 1);
            const auto n_nbr = end - start;

            // Fallback: zero gradient for cells with < 3 neighbors
            if (n_nbr < 3) {
                grad(i, 0) = 0.0;
                grad(i, 1) = 0.0;
                grad(i, 2) = 0.0;
                return;
            }

            const double val_i = cell_values(i);
            const double cx_i = centroids(i, 0);
            const double cy_i = centroids(i, 1);
            const double cz_i = centroids(i, 2);

            // Build normal equations: N = A^T A, rhs = A^T b
            // N is symmetric 3×3, only store upper triangle
            double n00 = 0.0, n01 = 0.0, n02 = 0.0;
            double n11 = 0.0, n12 = 0.0, n22 = 0.0;
            double r0 = 0.0, r1 = 0.0, r2 = 0.0;

            for (auto k = start; k < end; ++k) {
                const auto j = adj_indices(k);

                const double dx = centroids(j, 0) - cx_i;
                const double dy = centroids(j, 1) - cy_i;
                const double dz = centroids(j, 2) - cz_i;
                const double dv = cell_values(j) - val_i;

                // Accumulate A^T A
                n00 += dx * dx;
                n01 += dx * dy;
                n02 += dx * dz;
                n11 += dy * dy;
                n12 += dy * dz;
                n22 += dz * dz;

                // Accumulate A^T b
                r0 += dx * dv;
                r1 += dy * dv;
                r2 += dz * dv;
            }

            // Solve via Cramer's rule
            double gx, gy, gz;
            const bool ok = solve_3x3_cramer(n00, n01, n02, n11, n12, n22, r0, r1, r2, gx, gy, gz);

            if (ok) {
                grad(i, 0) = gx;
                grad(i, 1) = gy;
                grad(i, 2) = gz;
            } else {
                // Singular matrix fallback: zero gradient
                grad(i, 0) = 0.0;
                grad(i, 1) = 0.0;
                grad(i, 2) = 0.0;
            }
        });

    Kokkos::fence("GradientReconstructor::compute_gradients fence");

    // ── Phase 2: Barth-Jespersen limiter (optional) ─────────────────────────
    if (!use_limiter) return;

    Kokkos::parallel_for(
        "GradientReconstructor::apply_limiter", Kokkos::RangePolicy<execution_space>(0, n_cells), KOKKOS_LAMBDA(const int i) {
            const auto start = adj_offsets(i);
            const auto end = adj_offsets(i + 1);
            const auto n_nbr = end - start;

            // No neighbors → gradient is already zero, nothing to limit
            if (n_nbr == 0) return;

            const double val_i = cell_values(i);

            // Find min/max among neighbors (and self for safety)
            double val_min = val_i;
            double val_max = val_i;
            for (auto k = start; k < end; ++k) {
                const auto j = adj_indices(k);
                const double vj = cell_values(j);
                val_min = Kokkos::fmin(val_min, vj);
                val_max = Kokkos::fmax(val_max, vj);
            }

            // Compute limiter phi_i = min over neighbors of limiter function
            double phi = 1.0;

            for (auto k = start; k < end; ++k) {
                const auto j = adj_indices(k);

                // Reconstructed value at neighbor centroid
                const double dx = centroids(j, 0) - centroids(i, 0);
                const double dy = centroids(j, 1) - centroids(i, 1);
                const double dz = centroids(j, 2) - centroids(i, 2);

                const double delta_f = grad(i, 0) * dx + grad(i, 1) * dy + grad(i, 2) * dz;

                if (delta_f > 1.0e-30) {
                    // Positive excursion: limit to (val_max - val_i)
                    const double phi_k = Kokkos::fmin(1.0, (val_max - val_i) / delta_f);
                    phi = Kokkos::fmin(phi, phi_k);
                } else if (delta_f < -1.0e-30) {
                    // Negative excursion: limit to (val_min - val_i)
                    const double phi_k = Kokkos::fmin(1.0, (val_min - val_i) / delta_f);
                    phi = Kokkos::fmin(phi, phi_k);
                }
                // delta_f ≈ 0: no limiting needed for this neighbor
            }

            // Clamp phi to [0, 1] for numerical safety
            phi = Kokkos::fmax(0.0, Kokkos::fmin(1.0, phi));

            // Apply limiter: scale gradient uniformly
            grad(i, 0) *= phi;
            grad(i, 1) *= phi;
            grad(i, 2) *= phi;
        });

    Kokkos::fence("GradientReconstructor::apply_limiter fence");
}

// ─────────────────────────────────────────────────────────────────────────────
// Explicit template instantiations
// ─────────────────────────────────────────────────────────────────────────────

template struct GradientReconstructor<Kokkos::HostSpace>;

#ifdef KOKKOS_ENABLE_CUDA
template struct GradientReconstructor<Kokkos::CudaSpace>;
#endif

#ifdef KOKKOS_ENABLE_HIP
template struct GradientReconstructor<Kokkos::HIPSpace>;
#endif

}  // namespace axis::solver
