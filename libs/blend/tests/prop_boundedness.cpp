// SPDX-License-Identifier: Apache-2.0
// BLEND Property Test — Interpolation Boundedness
//
// Property 3: LinearBlendKernel Interpolation Boundedness
//   For alpha in [0.0, 1.0], output is element-wise bounded between
//   min(left[i], right[i]) and max(left[i], right[i]):
//   min(left[i], right[i]) <= target[i] <= max(left[i], right[i])
//
// Requirements: 1.9

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <blend/helm_math_blend.hpp>
#include <vector>

#include "generators.hpp"

// ── Kokkos lifecycle management ───────────────────────────────────────────────

class KokkosEnvironment : public ::testing::Environment {
   public:
    void SetUp() override {
        Kokkos::initialize();
    }
    void TearDown() override {
        Kokkos::finalize();
    }
};

static ::testing::Environment *const kokkos_env = ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);

// ── Property: output bounded by element-wise min/max of inputs ───────────────

RC_GTEST_PROP(BlendBoundedness, OutputBoundedByInputsForUnitAlpha, ()) {
    const std::size_t n = *blend::gen::array_length();
    const auto left_v = *blend::gen::field_data(n);
    const auto right_v = *blend::gen::field_data(n);
    const double alpha = *blend::gen::alpha_unit();

    std::vector<double> target(n, 0.0);

    span::FieldView v_left(const_cast<double *>(left_v.data()), n);
    span::FieldView v_right(const_cast<double *>(right_v.data()), n);
    span::FieldView v_target(target.data(), n);

    blend::LinearBlendKernel::apply(v_left, v_right, v_target, alpha);
    Kokkos::fence();

    for (std::size_t i = 0; i < n; ++i) {
        const double lo = std::min(left_v[i], right_v[i]);
        const double hi = std::max(left_v[i], right_v[i]);
        // Use a small relative tolerance to absorb floating-point rounding at
        // the boundary (e.g., alpha=1.0 with left > right).
        const double tol = std::abs(hi - lo) * 1e-12 + 1e-15;
        RC_ASSERT(target[i] >= lo - tol);
        RC_ASSERT(target[i] <= hi + tol);
    }
}
