// SPDX-License-Identifier: Apache-2.0
// BLEND — Stateless Array Blending Kernels
// Copyright (c) HELM Project Contributors

#ifndef BLEND_HELM_MATH_BLEND_HPP
#define BLEND_HELM_MATH_BLEND_HPP

/// @file blend/helm_math_blend.hpp
/// @brief High-speed stateless array blending kernels for the HELM ecosystem.
///
/// BLEND is a Tier 1 micro-library providing BLAS-style element-wise array
/// mathematics on Kokkos device execution spaces. It is entirely stateless,
/// has no knowledge of time, calendars, grids, or model physics, and depends
/// only on Kokkos and the SPAN interface types.
///
/// ## Memory Bandwidth Analysis
///
/// Both kernels are **memory-bandwidth-bound** on modern hardware:
///
/// | Kernel            | Reads             | Writes        | Arithmetic Intensity |
/// |-------------------|-------------------|---------------|----------------------|
/// | LinearBlendKernel | 2 × N × 8 bytes  | 1 × N × 8 bytes | ~2 FLOP / 24 bytes  |
/// | StepBlendKernel   | 2 × N × 8 bytes  | 1 × N × 8 bytes | ~1 FLOP / 24 bytes  |
///
/// At 24 bytes per element touched (two reads + one write of fp64), the kernels
/// achieve peak throughput when the array length exceeds L2 capacity, making
/// memory bus saturation the limiting factor. For short arrays that fit in L1/L2,
/// kernel launch overhead dominates.
///
/// @note All parallel dispatch uses Kokkos execution spaces (HELM Law #2).
///       No raw CUDA, HIP, or OpenMP constructs appear in this file.

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <stdexcept>
#include <string>

// ─────────────────────────────────────────────────────────────────────────────
// Forward-compatible SPAN interface type
// ─────────────────────────────────────────────────────────────────────────────

namespace span {

/// @brief Non-owning rank-1 device-portable view of a contiguous double array.
///
/// This is the SPAN layer's canonical field view type for BLEND consumption.
/// It wraps a Kokkos::View with unmanaged memory semantics, satisfying
/// HELM Law #1 (zero-copy): the underlying memory is owned by the Fortran
/// domain model or caller; BLEND never allocates or copies field data.
///
/// @note When the full SPAN library materializes, this alias will be replaced
///       by the authoritative definition. The interface contract (pointer +
///       extent, unmanaged, device-accessible) remains stable.
using FieldView = Kokkos::View<double *, Kokkos::MemoryUnmanaged>;

}  // namespace span

// ─────────────────────────────────────────────────────────────────────────────
// BLEND namespace
// ─────────────────────────────────────────────────────────────────────────────

namespace blend {

// ═════════════════════════════════════════════════════════════════════════════
// BlendProfile — static routing enum
// ═════════════════════════════════════════════════════════════════════════════

/// @brief Selects the blending algorithm dispatched by execute_blend().
///
/// The profile acts as a compile-time-friendly static switch that routes to
/// the appropriate kernel without virtual dispatch or function pointers.
enum class BlendProfile {
    Linear,  ///< Weighted linear interpolation: target = left*(1-α) + right*α
    Step     ///< Nearest-neighbor snap: target = (α < 0.5) ? left : right
};

// ═════════════════════════════════════════════════════════════════════════════
// LinearBlendKernel
// ═════════════════════════════════════════════════════════════════════════════

/// @brief Element-wise linear interpolation kernel (AXPY-style).
///
/// Computes: `target[i] = left[i] * (1.0 - alpha) + right[i] * alpha`
///
/// This is mathematically equivalent to a fused multiply-add (FMA) pair and
/// maps directly to hardware FMA units on GPU and SIMD lanes on CPU.
///
/// @par Memory Traffic (per element)
/// - Reads:  16 bytes (left[i] + right[i])
/// - Writes:  8 bytes (target[i])
/// - Total:  24 bytes / element → bandwidth-bound for N > L2 capacity
///
/// @par Execution Model
/// Dispatches a Kokkos::parallel_for over the default execution space.
/// Thread mapping is 1:1 (one element per thread/vector lane).
///
/// @param left   Source field A (unmanaged, device-accessible).
/// @param right  Source field B (unmanaged, device-accessible).
/// @param target Output field (unmanaged, device-accessible, may alias neither input).
/// @param alpha  Blending weight in [0.0, 1.0]. Values outside this range
///              extrapolate linearly (no clamping).
///
/// @pre left.extent(0) == right.extent(0) == target.extent(0)
/// @post target[i] == left[i] * (1.0 - alpha) + right[i] * alpha  ∀ i
///
/// @throws std::invalid_argument if extents are mismatched.
struct LinearBlendKernel {
    static void apply(const span::FieldView &left, const span::FieldView &right, const span::FieldView &target, const double alpha) {
        const std::size_t n = left.extent(0);

        if (right.extent(0) != n || target.extent(0) != n) {
            throw std::invalid_argument("LinearBlendKernel: extent mismatch — left(" + std::to_string(n) + "), right(" +
                                        std::to_string(right.extent(0)) + "), target(" + std::to_string(target.extent(0)) + ")");
        }

        const double one_minus_alpha = 1.0 - alpha;

        Kokkos::parallel_for(
            "BLEND::LinearBlend", Kokkos::RangePolicy<>(0, n),
            KOKKOS_LAMBDA(const std::size_t i) { target(i) = left(i) * one_minus_alpha + right(i) * alpha; });

        Kokkos::fence("BLEND::LinearBlend::fence");
    }
};

// ═════════════════════════════════════════════════════════════════════════════
// StepBlendKernel
// ═════════════════════════════════════════════════════════════════════════════

/// @brief Element-wise nearest-neighbor snap kernel.
///
/// Computes:
/// ```
/// target[i] = (alpha < 0.5) ? left[i] : right[i]
/// ```
///
/// This kernel performs a branchless select on GPU architectures (maps to
/// a predicated move or conditional select instruction). On CPUs it compiles
/// to a CMOV or blend intrinsic with appropriate optimization flags.
///
/// @par Memory Traffic (per element)
/// - Reads:  16 bytes (left[i] + right[i]) — both loaded regardless of branch
/// - Writes:  8 bytes (target[i])
/// - Total:  24 bytes / element → identical bandwidth profile to LinearBlendKernel
///
/// @par Design Note
/// Both inputs are unconditionally loaded to maintain uniform memory access
/// patterns and avoid warp divergence on GPU. The conditional select is
/// purely arithmetic with no control-flow branching.
///
/// @param left   Source field A (unmanaged, device-accessible).
/// @param right  Source field B (unmanaged, device-accessible).
/// @param target Output field (unmanaged, device-accessible).
/// @param alpha  Threshold weight. If < 0.5, snaps to left; otherwise right.
///
/// @pre left.extent(0) == right.extent(0) == target.extent(0)
/// @post target[i] == left[i] if alpha < 0.5, else target[i] == right[i]  ∀ i
///
/// @throws std::invalid_argument if extents are mismatched.
struct StepBlendKernel {
    static void apply(const span::FieldView &left, const span::FieldView &right, const span::FieldView &target, const double alpha) {
        const std::size_t n = left.extent(0);

        if (right.extent(0) != n || target.extent(0) != n) {
            throw std::invalid_argument("StepBlendKernel: extent mismatch — left(" + std::to_string(n) + "), right(" +
                                        std::to_string(right.extent(0)) + "), target(" + std::to_string(target.extent(0)) + ")");
        }

        // Precompute the selection flag outside the kernel to avoid
        // per-element branch evaluation (it's uniform across all elements).
        const bool snap_right = (alpha >= 0.5);

        Kokkos::parallel_for(
            "BLEND::StepBlend", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const std::size_t i) { target(i) = snap_right ? right(i) : left(i); });

        Kokkos::fence("BLEND::StepBlend::fence");
    }
};

// ═════════════════════════════════════════════════════════════════════════════
// execute_blend — public entry point
// ═════════════════════════════════════════════════════════════════════════════

/// @brief Unified entry point for all blending operations.
///
/// Routes to the appropriate kernel based on the BlendProfile enum, providing
/// a single narrow API surface for callers. The static switch is resolved at
/// compile time when the profile is a constexpr, or at negligible cost (single
/// branch) when determined at runtime.
///
/// @par Usage Example
/// @code
/// span::FieldView left(ptr_left, n);
/// span::FieldView right(ptr_right, n);
/// span::FieldView target(ptr_target, n);
///
/// // Linear interpolation at 30% blend
/// blend::execute_blend(left, right, target, 0.3, blend::BlendProfile::Linear);
///
/// // Nearest-neighbor snap (picks right since 0.7 >= 0.5)
/// blend::execute_blend(left, right, target, 0.7, blend::BlendProfile::Step);
/// @endcode
///
/// @param left    Source field A (zero-copy view, device-accessible).
/// @param right   Source field B (zero-copy view, device-accessible).
/// @param target  Output field (zero-copy view, device-accessible).
/// @param alpha   Blending parameter in [0.0, 1.0].
/// @param profile Algorithm selector (Linear or Step).
///
/// @throws std::invalid_argument if field extents are mismatched.
/// @throws std::invalid_argument if profile is not a recognized BlendProfile value.
inline void execute_blend(const span::FieldView &left, const span::FieldView &right, const span::FieldView &target, const double alpha,
                          const BlendProfile profile) {
    switch (profile) {
        case BlendProfile::Linear:
            LinearBlendKernel::apply(left, right, target, alpha);
            break;
        case BlendProfile::Step:
            StepBlendKernel::apply(left, right, target, alpha);
            break;
        default:
            throw std::invalid_argument("execute_blend: unknown BlendProfile value (" + std::to_string(static_cast<int>(profile)) + ")");
    }
}

}  // namespace blend

#endif  // BLEND_HELM_MATH_BLEND_HPP
