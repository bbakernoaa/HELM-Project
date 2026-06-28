// SPDX-License-Identifier: Apache-2.0
// BLEND Property Test — Step-Select Correctness
//
// Property 2: StepBlendKernel Step-Select Correctness
//   For any alpha, every output element equals exactly one of the two inputs:
//   target[i] == left[i]  OR  target[i] == right[i]
//
// Requirements: 1.8

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
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

// ── Property: output always equals exactly one of the two inputs ──────────────

RC_GTEST_PROP(StepSelectCorrectness, OutputEqualsOneOfInputs, ()) {
    const std::size_t n = *blend::gen::array_length();
    const auto left_v = *blend::gen::field_data(n);
    const auto right_v = *blend::gen::field_data(n);
    const double alpha = *blend::gen::alpha_wide();

    std::vector<double> target(n, 0.0);

    span::FieldView v_left(const_cast<double *>(left_v.data()), n);
    span::FieldView v_right(const_cast<double *>(right_v.data()), n);
    span::FieldView v_target(target.data(), n);

    blend::StepBlendKernel::apply(v_left, v_right, v_target, alpha);
    Kokkos::fence();

    for (std::size_t i = 0; i < n; ++i) {
        RC_ASSERT(target[i] == left_v[i] || target[i] == right_v[i]);
    }
}

// ── Property: output is uniformly left when alpha < 0.5 ──────────────────────

RC_GTEST_PROP(StepSelectCorrectness, AlphaLtHalfAlwaysLeft, ()) {
    const std::size_t n = *blend::gen::array_length();
    const auto left_v = *blend::gen::field_data(n);
    const auto right_v = *blend::gen::field_data(n);

    // Generate alpha strictly in [0, 0.5)
    const int raw = *rc::gen::inRange<int>(0, 5000);
    const double alpha = static_cast<double>(raw) / 10000.0;  // [0.0, 0.4999]

    std::vector<double> target(n, 0.0);

    span::FieldView v_left(const_cast<double *>(left_v.data()), n);
    span::FieldView v_right(const_cast<double *>(right_v.data()), n);
    span::FieldView v_target(target.data(), n);

    blend::StepBlendKernel::apply(v_left, v_right, v_target, alpha);
    Kokkos::fence();

    for (std::size_t i = 0; i < n; ++i) {
        RC_ASSERT(target[i] == left_v[i]);
    }
}

// ── Property: output is uniformly right when alpha >= 0.5 ────────────────────

RC_GTEST_PROP(StepSelectCorrectness, AlphaGeqHalfAlwaysRight, ()) {
    const std::size_t n = *blend::gen::array_length();
    const auto left_v = *blend::gen::field_data(n);
    const auto right_v = *blend::gen::field_data(n);

    // Generate alpha in [0.5, 1.0]
    const int raw = *rc::gen::inRange<int>(5000, 10001);
    const double alpha = static_cast<double>(raw) / 10000.0;

    std::vector<double> target(n, 0.0);

    span::FieldView v_left(const_cast<double *>(left_v.data()), n);
    span::FieldView v_right(const_cast<double *>(right_v.data()), n);
    span::FieldView v_target(target.data(), n);

    blend::StepBlendKernel::apply(v_left, v_right, v_target, alpha);
    Kokkos::fence();

    for (std::size_t i = 0; i < n; ++i) {
        RC_ASSERT(target[i] == right_v[i]);
    }
}
