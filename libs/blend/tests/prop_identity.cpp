// SPDX-License-Identifier: Apache-2.0
// BLEND Property Test — Identity Property
//
// Property 1: LinearBlendKernel Identity
//   - alpha = 0.0  →  target[i] == left[i]   for all i
//   - alpha = 1.0  →  target[i] == right[i]  for all i
//
// Requirements: 1.7

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <blend/helm_math_blend.hpp>

#include "generators.hpp"

#include <vector>

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

static ::testing::Environment* const kokkos_env = ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);

// ── Property: alpha=0 produces left unchanged ─────────────────────────────────

RC_GTEST_PROP(BlendIdentity, AlphaZeroReturnsLeft, ()) {
    const std::size_t n       = *blend::gen::array_length();
    const auto        left_v  = *blend::gen::field_data(n);
    const auto        right_v = *blend::gen::field_data(n);

    std::vector<double> target(n, 0.0);

    span::FieldView v_left(const_cast<double*>(left_v.data()), n);
    span::FieldView v_right(const_cast<double*>(right_v.data()), n);
    span::FieldView v_target(target.data(), n);

    blend::LinearBlendKernel::apply(v_left, v_right, v_target, 0.0);
    Kokkos::fence();

    for (std::size_t i = 0; i < n; ++i) {
        RC_ASSERT(target[i] == left_v[i]);
    }
}

// ── Property: alpha=1 produces right unchanged ────────────────────────────────

RC_GTEST_PROP(BlendIdentity, AlphaOneReturnsRight, ()) {
    const std::size_t n       = *blend::gen::array_length();
    const auto        left_v  = *blend::gen::field_data(n);
    const auto        right_v = *blend::gen::field_data(n);

    std::vector<double> target(n, 0.0);

    span::FieldView v_left(const_cast<double*>(left_v.data()), n);
    span::FieldView v_right(const_cast<double*>(right_v.data()), n);
    span::FieldView v_target(target.data(), n);

    blend::LinearBlendKernel::apply(v_left, v_right, v_target, 1.0);
    Kokkos::fence();

    for (std::size_t i = 0; i < n; ++i) {
        RC_ASSERT(target[i] == right_v[i]);
    }
}
