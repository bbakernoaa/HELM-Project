// SPDX-License-Identifier: Apache-2.0
// BLEND Test Suite — StepBlendKernel unit tests
//
// Verifies:
//   1. Step-select behavior: left when α<0.5, right when α≥0.5
//   2. std::invalid_argument on extent mismatch
//   3. Correct no-op behavior on empty arrays
//
// Requirements: 1.3, 1.5, 1.6

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <blend/helm_math_blend.hpp>

#include <stdexcept>
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

// ── Helper: run StepBlendKernel on host std::vectors ─────────────────────────

static std::vector<double> run_step(const std::vector<double>& left, const std::vector<double>& right, double alpha) {
    const std::size_t n = left.size();
    std::vector<double> target(n, 0.0);

    span::FieldView v_left(const_cast<double*>(left.data()), n);
    span::FieldView v_right(const_cast<double*>(right.data()), n);
    span::FieldView v_target(target.data(), n);

    blend::StepBlendKernel::apply(v_left, v_right, v_target, alpha);
    Kokkos::fence();

    return target;
}

// ── Test 1: alpha < 0.5 → output equals left ─────────────────────────────────

TEST(StepBlendKernel, AlphaLessThanHalfSelectsLeft) {
    const std::vector<double> left  = {1.0, 2.0, 3.0, 4.0};
    const std::vector<double> right = {10.0, 20.0, 30.0, 40.0};

    for (const double alpha : {0.0, 0.1, 0.25, 0.499}) {
        const auto result = run_step(left, right, alpha);
        for (std::size_t i = 0; i < left.size(); ++i) {
            EXPECT_DOUBLE_EQ(result[i], left[i]) << "alpha=" << alpha << " at index " << i;
        }
    }
}

// ── Test 2: alpha >= 0.5 → output equals right ───────────────────────────────

TEST(StepBlendKernel, AlphaGeqHalfSelectsRight) {
    const std::vector<double> left  = {1.0, 2.0, 3.0, 4.0};
    const std::vector<double> right = {10.0, 20.0, 30.0, 40.0};

    for (const double alpha : {0.5, 0.501, 0.75, 1.0}) {
        const auto result = run_step(left, right, alpha);
        for (std::size_t i = 0; i < left.size(); ++i) {
            EXPECT_DOUBLE_EQ(result[i], right[i]) << "alpha=" << alpha << " at index " << i;
        }
    }
}

// ── Test 3: Boundary alpha = 0.5 exactly selects right ───────────────────────

TEST(StepBlendKernel, AlphaExactlyHalfSelectsRight) {
    const std::vector<double> left  = {-5.0, 0.0, 5.0};
    const std::vector<double> right = {100.0, 200.0, 300.0};

    const auto result = run_step(left, right, 0.5);
    for (std::size_t i = 0; i < right.size(); ++i) {
        EXPECT_DOUBLE_EQ(result[i], right[i]) << "at index " << i;
    }
}

// ── Test 4: Extent mismatch throws std::invalid_argument ─────────────────────

TEST(StepBlendKernel, ExtentMismatchThrows) {
    std::vector<double> left   = {1.0, 2.0, 3.0};
    std::vector<double> right  = {4.0, 5.0};          // wrong size
    std::vector<double> target = {0.0, 0.0, 0.0};

    span::FieldView v_left(left.data(), left.size());
    span::FieldView v_right(right.data(), right.size());
    span::FieldView v_target(target.data(), target.size());

    EXPECT_THROW(blend::StepBlendKernel::apply(v_left, v_right, v_target, 0.5), std::invalid_argument);
}

TEST(StepBlendKernel, TargetExtentMismatchThrows) {
    std::vector<double> left   = {1.0, 2.0, 3.0};
    std::vector<double> right  = {4.0, 5.0, 6.0};
    std::vector<double> target = {0.0};                // wrong size

    span::FieldView v_left(left.data(), left.size());
    span::FieldView v_right(right.data(), right.size());
    span::FieldView v_target(target.data(), target.size());

    EXPECT_THROW(blend::StepBlendKernel::apply(v_left, v_right, v_target, 0.5), std::invalid_argument);
}

// ── Test 5: Empty arrays — no-op, no exception ───────────────────────────────

TEST(StepBlendKernel, EmptyArrayNoOp) {
    std::vector<double> left, right, target;

    span::FieldView v_left(left.data(), 0);
    span::FieldView v_right(right.data(), 0);
    span::FieldView v_target(target.data(), 0);

    EXPECT_NO_THROW(blend::StepBlendKernel::apply(v_left, v_right, v_target, 0.0));
    EXPECT_NO_THROW(blend::StepBlendKernel::apply(v_left, v_right, v_target, 1.0));
}
