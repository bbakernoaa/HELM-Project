// SPDX-License-Identifier: Apache-2.0
// BLEND Test Suite — LinearBlendKernel unit tests
//
// Verifies:
//   1. Fixed-input formula correctness: target[i] = left[i]*(1-α) + right[i]*α
//   2. std::invalid_argument on extent mismatch
//   3. Correct no-op behavior on empty arrays
//
// Requirements: 1.2, 1.4, 1.6

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <blend/helm_math_blend.hpp>
#include <cmath>
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

// Register Kokkos environment so it runs before any test.
static ::testing::Environment *const kokkos_env = ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);

// ── Helper: run LinearBlendKernel on host std::vectors ───────────────────────

static std::vector<double> run_linear(const std::vector<double> &left, const std::vector<double> &right, double alpha) {
    const std::size_t n = left.size();
    std::vector<double> target(n, 0.0);

    span::FieldView v_left(const_cast<double *>(left.data()), n);
    span::FieldView v_right(const_cast<double *>(right.data()), n);
    span::FieldView v_target(target.data(), n);

    blend::LinearBlendKernel::apply(v_left, v_right, v_target, alpha);
    Kokkos::fence();

    return target;
}

// ── Test 1: Formula correctness with known fixed inputs ───────────────────────

TEST(LinearBlendKernel, FormulaCorrectness) {
    // left = {0, 2, 4, 6}, right = {4, 6, 8, 10}, alpha = 0.25
    // expected[i] = left[i]*0.75 + right[i]*0.25
    const std::vector<double> left = {0.0, 2.0, 4.0, 6.0};
    const std::vector<double> right = {4.0, 6.0, 8.0, 10.0};
    const double alpha = 0.25;

    const auto result = run_linear(left, right, alpha);

    for (std::size_t i = 0; i < left.size(); ++i) {
        const double expected = left[i] * (1.0 - alpha) + right[i] * alpha;
        EXPECT_DOUBLE_EQ(result[i], expected) << "mismatch at index " << i;
    }
}

// ── Test 2: Formula at alpha = 0.5 ───────────────────────────────────────────

TEST(LinearBlendKernel, AlphaHalf) {
    const std::vector<double> left = {0.0, 10.0, -5.0};
    const std::vector<double> right = {10.0, 0.0, 5.0};
    const double alpha = 0.5;

    const auto result = run_linear(left, right, alpha);

    for (std::size_t i = 0; i < left.size(); ++i) {
        const double expected = (left[i] + right[i]) * 0.5;
        EXPECT_DOUBLE_EQ(result[i], expected) << "mismatch at index " << i;
    }
}

// ── Test 3: Extent mismatch throws std::invalid_argument ─────────────────────

TEST(LinearBlendKernel, ExtentMismatchThrows) {
    std::vector<double> left = {1.0, 2.0, 3.0};
    std::vector<double> right = {4.0, 5.0};  // wrong size
    std::vector<double> target = {0.0, 0.0, 0.0};

    span::FieldView v_left(left.data(), left.size());
    span::FieldView v_right(right.data(), right.size());
    span::FieldView v_target(target.data(), target.size());

    EXPECT_THROW(blend::LinearBlendKernel::apply(v_left, v_right, v_target, 0.5), std::invalid_argument);
}

TEST(LinearBlendKernel, TargetExtentMismatchThrows) {
    std::vector<double> left = {1.0, 2.0, 3.0};
    std::vector<double> right = {4.0, 5.0, 6.0};
    std::vector<double> target = {0.0, 0.0};  // wrong size

    span::FieldView v_left(left.data(), left.size());
    span::FieldView v_right(right.data(), right.size());
    span::FieldView v_target(target.data(), target.size());

    EXPECT_THROW(blend::LinearBlendKernel::apply(v_left, v_right, v_target, 0.5), std::invalid_argument);
}

// ── Test 4: Empty arrays — no-op, no exception ───────────────────────────────

TEST(LinearBlendKernel, EmptyArrayNoOp) {
    std::vector<double> left, right, target;

    span::FieldView v_left(left.data(), 0);
    span::FieldView v_right(right.data(), 0);
    span::FieldView v_target(target.data(), 0);

    EXPECT_NO_THROW(blend::LinearBlendKernel::apply(v_left, v_right, v_target, 0.5));
}
