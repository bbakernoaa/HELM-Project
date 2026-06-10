// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_SOLVER_APPLY_HPP
#define AXIS_SOLVER_APPLY_HPP

/// @file axis/solver/apply.hpp
/// @brief Kokkos-parallel sparse matrix-vector apply (SpMV) for field regridding.
///
/// Implements dst = S · src via scatter-add SpMV over non-owning layout_left
/// field views, exactly mirroring the ESMF apply loop:
///   dst(row(k)) += S(k) * src(col(k))
///
/// Three overloads are provided:
///   1. Local apply — single-rank, matrix + src + dst
///   2. Distributed apply — matrix + pattern + local_src + gathered_halo_src + dst
///   3. Callback-based distributed apply — uses a user-provided gather functor
///
/// Header-only (template) since it is parameterized on MemorySpace.
/// Uses Kokkos::parallel_for with Kokkos::atomic_add for thread-safe accumulation.

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include <axis/solver/halo_pattern.hpp>
#include <axis/solver/interpolation_matrix.hpp>
#include <axis/types.hpp>

namespace axis::solver {

// ─────────────────────────────────────────────────────────────────────────────
// 1. Local apply: dst = S · src (single-rank)
// ─────────────────────────────────────────────────────────────────────────────

/// Apply the interpolation matrix: dst = S · src.
///
/// Implements, in parallel over k in [0, nnz):
///     Kokkos::atomic_add(&dst(row_k), S(k) * src(col_k));
/// after zero-initializing dst.
///
/// src and dst are non-owning layout_left views whose memory MUST reside in
/// MemorySpace (HELM Law #1: no copy of field data).
///
/// @tparam MemorySpace Kokkos memory space of the InterpolationMatrix
/// @param matrix  Sparse interpolation operator (COO: factorList + row/col)
/// @param src     Source field [n_src]
/// @param dst     Destination field [n_dst] — overwritten with result
///
/// @throws std::invalid_argument if src.extent(0) != matrix.n_src() or
///         dst.extent(0) != matrix.n_dst(). dst is NOT written on validation
///         failure.
template <class MemorySpace>
void apply(const InterpolationMatrix<MemorySpace>& matrix,
           field_view<const double, 1>             src,
           field_view<double, 1>                   dst)
{
    // ── Extent validation (throw BEFORE touching dst) ────────────────────────
    if (src.extent(0) != matrix.n_src()) {
        throw std::invalid_argument(
            "axis::solver::apply: src.extent(0) (" +
            std::to_string(src.extent(0)) + ") != matrix.n_src() (" +
            std::to_string(matrix.n_src()) + ")");
    }
    if (dst.extent(0) != matrix.n_dst()) {
        throw std::invalid_argument(
            "axis::solver::apply: dst.extent(0) (" +
            std::to_string(dst.extent(0)) + ") != matrix.n_dst() (" +
            std::to_string(matrix.n_dst()) + ")");
    }

    // ── Obtain internal Kokkos Views for the parallel kernel ─────────────────
    const auto& S   = matrix.factor_list_view();  // [nnz]
    const auto& row = matrix.factor_row_view();   // [nnz]
    const auto& col = matrix.factor_col_view();   // [nnz]
    const std::size_t nnz   = matrix.nnz();
    const std::size_t n_dst = matrix.n_dst();

    // Wrap the raw dst pointer in a Kokkos unmanaged View for the kernel.
    Kokkos::View<double*, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        dst_view(dst.data_handle(), n_dst);

    // Wrap the raw src pointer in a Kokkos unmanaged View for the kernel.
    Kokkos::View<const double*, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        src_view(src.data_handle(), src.extent(0));

    // ── Zero-initialize dst ──────────────────────────────────────────────────
    Kokkos::parallel_for(
        "axis::apply::zero_dst",
        Kokkos::RangePolicy<typename MemorySpace::execution_space>(0, n_dst),
        KOKKOS_LAMBDA(const std::size_t j) {
            dst_view(j) = 0.0;
        });

    // ── SpMV scatter-add: dst(row_k) += S(k) * src(col_k) ───────────────────
    Kokkos::parallel_for(
        "axis::apply::spmv",
        Kokkos::RangePolicy<typename MemorySpace::execution_space>(0, nnz),
        KOKKOS_LAMBDA(const std::size_t k) {
            const auto r = row(k);
            const auto c = col(k);
            Kokkos::atomic_add(&dst_view(r), S(k) * src_view(c));
        });

    Kokkos::fence("axis::apply::complete");
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Distributed apply: local_src + gathered_halo_src → dst
// ─────────────────────────────────────────────────────────────────────────────

/// Distributed apply (ESMF ASMM). Reads locally-owned source cells from
/// `local_src` and off-rank source cells from `gathered_halo_src` — the
/// contiguous buffer HALO produced by exchanging the published HaloPattern.
///
/// For each nonzero k, the source value is:
///   - local_src(col_k)  if col_k < local_src.extent(0)  (locally-owned)
///   - gathered_halo_src(gather_slot mapping)  otherwise  (off-rank)
///
/// The column indices in the matrix are encoded as:
///   - [0, local_src.extent(0)) → local source cells
///   - [local_src.extent(0), ...) → offset into gathered_halo_src
///
/// AXIS performs NO MPI; it only reads the buffer the caller (via HALO)
/// already gathered in gather_slot order.
///
/// @tparam MemorySpace Kokkos memory space of the InterpolationMatrix
/// @param matrix             Sparse interpolation operator
/// @param pattern            HaloPattern describing off-rank dependencies
/// @param local_src          Locally-owned source values [n_local_src]
/// @param gathered_halo_src  Off-rank source values from HALO [num_remote()]
/// @param dst                Destination field [n_dst] — overwritten with result
///
/// @throws std::invalid_argument if:
///   - dst.extent(0) != matrix.n_dst()
///   - gathered_halo_src.extent(0) != pattern.num_remote()
///   - local_src extent disagrees with matrix dimensions
template <class MemorySpace>
void apply(const InterpolationMatrix<MemorySpace>& matrix,
           const HaloPattern&                      pattern,
           field_view<const double, 1>             local_src,
           field_view<const double, 1>             gathered_halo_src,
           field_view<double, 1>                   dst)
{
    // ── Extent validation (throw BEFORE touching dst) ────────────────────────
    if (dst.extent(0) != matrix.n_dst()) {
        throw std::invalid_argument(
            "axis::solver::apply (distributed): dst.extent(0) (" +
            std::to_string(dst.extent(0)) + ") != matrix.n_dst() (" +
            std::to_string(matrix.n_dst()) + ")");
    }
    if (gathered_halo_src.extent(0) != pattern.num_remote()) {
        throw std::invalid_argument(
            "axis::solver::apply (distributed): gathered_halo_src.extent(0) (" +
            std::to_string(gathered_halo_src.extent(0)) +
            ") != pattern.num_remote() (" +
            std::to_string(pattern.num_remote()) + ")");
    }

    const std::size_t n_local_src = local_src.extent(0);

    // Validate: local_src + num_remote should cover the matrix's source space
    if (n_local_src + pattern.num_remote() != matrix.n_src()) {
        throw std::invalid_argument(
            "axis::solver::apply (distributed): local_src.extent(0) (" +
            std::to_string(n_local_src) + ") + pattern.num_remote() (" +
            std::to_string(pattern.num_remote()) +
            ") != matrix.n_src() (" +
            std::to_string(matrix.n_src()) + ")");
    }

    // ── Obtain internal Kokkos Views ─────────────────────────────────────────
    const auto& S   = matrix.factor_list_view();
    const auto& row = matrix.factor_row_view();
    const auto& col = matrix.factor_col_view();
    const std::size_t nnz   = matrix.nnz();
    const std::size_t n_dst = matrix.n_dst();

    // Wrap raw pointers in unmanaged Kokkos Views
    Kokkos::View<double*, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        dst_view(dst.data_handle(), n_dst);

    Kokkos::View<const double*, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        local_src_view(local_src.data_handle(), n_local_src);

    Kokkos::View<const double*, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        halo_src_view(gathered_halo_src.data_handle(), pattern.num_remote());

    // ── Zero-initialize dst ──────────────────────────────────────────────────
    Kokkos::parallel_for(
        "axis::apply_distributed::zero_dst",
        Kokkos::RangePolicy<typename MemorySpace::execution_space>(0, n_dst),
        KOKKOS_LAMBDA(const std::size_t j) {
            dst_view(j) = 0.0;
        });

    // ── SpMV with local/remote source dispatch ───────────────────────────────
    // Column indices < n_local_src refer to locally-owned cells;
    // Column indices >= n_local_src refer to gathered halo buffer.
    const auto n_local = static_cast<index_t>(n_local_src);
    Kokkos::parallel_for(
        "axis::apply_distributed::spmv",
        Kokkos::RangePolicy<typename MemorySpace::execution_space>(0, nnz),
        KOKKOS_LAMBDA(const std::size_t k) {
            const auto r = row(k);
            const auto c = col(k);
            double src_val;
            if (c < n_local) {
                src_val = local_src_view(c);
            } else {
                src_val = halo_src_view(c - n_local);
            }
            Kokkos::atomic_add(&dst_view(r), S(k) * src_val);
        });

    Kokkos::fence("axis::apply_distributed::complete");
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Callback-based distributed apply
// ─────────────────────────────────────────────────────────────────────────────

/// Distributed apply via a caller-supplied abstract gather callback/functor.
///
/// AXIS invokes `gather` exactly once with the published pattern to obtain the
/// off-rank source values, then proceeds as the gathered-buffer overload.
/// AXIS still performs NO MPI and knows nothing of HALO — the callback owns
/// all communication. Useful for a Python layer or a test stub.
///
/// @tparam MemorySpace Kokkos memory space of the InterpolationMatrix
/// @param matrix     Sparse interpolation operator
/// @param pattern    HaloPattern describing off-rank dependencies
/// @param local_src  Locally-owned source values [n_local_src]
/// @param gather     Callback that receives the HaloPattern and returns a
///                   std::vector<double> of size pattern.num_remote() with the
///                   off-rank source values in gather_slot order
/// @param dst        Destination field [n_dst] — overwritten with result
///
/// @throws std::invalid_argument if extent validation fails (same as
///         distributed overload)
/// @throws Any exception the gather callback throws
template <class MemorySpace>
void apply(const InterpolationMatrix<MemorySpace>&                        matrix,
           const HaloPattern&                                             pattern,
           field_view<const double, 1>                                    local_src,
           const std::function<std::vector<double>(const HaloPattern&)>&  gather,
           field_view<double, 1>                                          dst)
{
    // Invoke the gather callback to obtain off-rank source values
    std::vector<double> halo_buffer = gather(pattern);

    if (halo_buffer.size() != pattern.num_remote()) {
        throw std::invalid_argument(
            "axis::solver::apply (callback): gather returned " +
            std::to_string(halo_buffer.size()) + " values but pattern.num_remote() is " +
            std::to_string(pattern.num_remote()));
    }

    // Wrap the gathered buffer as a field_view and delegate to the buffer overload
    field_view<const double, 1> halo_view(halo_buffer.data(), halo_buffer.size());
    apply(matrix, pattern, local_src, halo_view, dst);
}

} // namespace axis::solver

#endif // AXIS_SOLVER_APPLY_HPP
