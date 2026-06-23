// DAGR — test_transform_node.cpp
// Unit tests for Transform_Node dispatch (dispatch_temporal_bookend).
// Verifies call sequencing, profile routing, error propagation, and
// zero-computation guarantee using link-time mock substitution.
//
// Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.8, 4.9, 4.10

#include <gtest/gtest.h>

#include "dagr/pipeline_config.hpp"
#include "dagr/detail/completion_token.hpp"
#include "dagr/detail/task_node.hpp"

#include "tick/aliased_window.hpp"
#include "tick/time_point.hpp"
#include "tick/time_window.hpp"

#include "amio/amio.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// ═════════════════════════════════════════════════════════════════════════════
// Forward declaration of the function under test
// ═════════════════════════════════════════════════════════════════════════════

namespace dagr::detail {
void dispatch_temporal_bookend(
    const dagr::Stream_Descriptor& stream,
    const tick::AliasedWindow& aliased,
    amio_dataset_handle dataset,
    const char* var_name,
    std::int64_t t_left_step,
    std::int64_t t_right_step,
    double* output_ptr,
    std::size_t output_count,
    Completion_Token& token);
} // namespace dagr::detail

// ═════════════════════════════════════════════════════════════════════════════
// Mock infrastructure — global state for tracking calls and configuring faults
// ═════════════════════════════════════════════════════════════════════════════

namespace {

/// Call event recorded by mocks for sequence verification.
enum class MockCall : int {
    amio_read_left   = 0,
    amio_read_right  = 1,
    amio_view_data_left  = 2,
    amio_view_data_right = 3,
    blend_dispatch   = 4
};

/// Global mock state (reset per test).
struct MockState {
    // Call sequence recording
    std::vector<MockCall> call_sequence;

    // Configurable failure injection
    amio_status_t amio_read_left_rc   = AMIO_OK;
    amio_status_t amio_read_right_rc  = AMIO_OK;
    amio_status_t amio_view_data_left_rc  = AMIO_OK;
    amio_status_t amio_view_data_right_rc = AMIO_OK;
    bool blend_should_throw = false;

    // Data buffers returned by AMIO mocks
    static constexpr std::size_t kFieldSize = 4;
    double left_data[kFieldSize]  = {1.0, 2.0, 3.0, 4.0};
    double right_data[kFieldSize] = {5.0, 6.0, 7.0, 8.0};

    // Track which read call we're on (first = left, second = right)
    int amio_read_call_count = 0;
    int amio_view_data_call_count = 0;

    // BLEND dispatch captured arguments
    const double* blend_left_ptr  = nullptr;
    const double* blend_right_ptr = nullptr;
    double* blend_out_ptr         = nullptr;
    std::size_t blend_count       = 0;
    double blend_alpha            = -1.0;
    int blend_profile_tag         = -1;

    // Sentinel view handles (non-null pointers for mock tracking)
    int left_view_sentinel  = 1;
    int right_view_sentinel = 2;

    void reset() {
        call_sequence.clear();
        amio_read_left_rc   = AMIO_OK;
        amio_read_right_rc  = AMIO_OK;
        amio_view_data_left_rc  = AMIO_OK;
        amio_view_data_right_rc = AMIO_OK;
        blend_should_throw = false;
        amio_read_call_count = 0;
        amio_view_data_call_count = 0;
        blend_left_ptr  = nullptr;
        blend_right_ptr = nullptr;
        blend_out_ptr   = nullptr;
        blend_count     = 0;
        blend_alpha     = -1.0;
        blend_profile_tag = -1;
        // Reset data buffers to known values
        left_data[0] = 1.0; left_data[1] = 2.0; left_data[2] = 3.0; left_data[3] = 4.0;
        right_data[0] = 5.0; right_data[1] = 6.0; right_data[2] = 7.0; right_data[3] = 8.0;
    }
};

MockState g_mock;

} // anonymous namespace

// ═════════════════════════════════════════════════════════════════════════════
// Mock implementations — C linkage (AMIO)
//
// These replace the real AMIO library symbols at link time. The test
// executable links dagr (containing transform_node.o with unresolved
// AMIO references) and these mock definitions satisfy those references.
// ═════════════════════════════════════════════════════════════════════════════

extern "C" {

amio_status_t amio_read(amio_dataset_handle /*dataset*/,
                        const char* /*var_name*/,
                        int64_t /*timestep*/,
                        const amio_bbox_t* /*bbox*/,
                        amio_view_handle* out_view)
{
    int call_idx = g_mock.amio_read_call_count++;

    if (call_idx == 0) {
        // First call = left bookend fetch
        g_mock.call_sequence.push_back(MockCall::amio_read_left);
        if (g_mock.amio_read_left_rc != AMIO_OK) {
            *out_view = nullptr;
            return g_mock.amio_read_left_rc;
        }
        *out_view = reinterpret_cast<amio_view_handle>(&g_mock.left_view_sentinel);
    } else {
        // Second call = right bookend fetch
        g_mock.call_sequence.push_back(MockCall::amio_read_right);
        if (g_mock.amio_read_right_rc != AMIO_OK) {
            *out_view = nullptr;
            return g_mock.amio_read_right_rc;
        }
        *out_view = reinterpret_cast<amio_view_handle>(&g_mock.right_view_sentinel);
    }
    return AMIO_OK;
}

amio_status_t amio_view_data(amio_view_handle view,
                             const void** out_data,
                             size_t* out_size)
{
    int call_idx = g_mock.amio_view_data_call_count++;

    if (call_idx == 0) {
        // First view_data call = left bookend
        g_mock.call_sequence.push_back(MockCall::amio_view_data_left);
        if (g_mock.amio_view_data_left_rc != AMIO_OK) {
            return g_mock.amio_view_data_left_rc;
        }
        *out_data = static_cast<const void*>(g_mock.left_data);
        *out_size = MockState::kFieldSize * sizeof(double);
    } else {
        // Second view_data call = right bookend
        g_mock.call_sequence.push_back(MockCall::amio_view_data_right);
        if (g_mock.amio_view_data_right_rc != AMIO_OK) {
            return g_mock.amio_view_data_right_rc;
        }
        *out_data = static_cast<const void*>(g_mock.right_data);
        *out_size = MockState::kFieldSize * sizeof(double);
    }
    return AMIO_OK;
}

const char* amio_strerror(int err)
{
    // Simple mock — just return a fixed string for any error
    static const char* msg = "mock AMIO error";
    (void)err;
    return msg;
}

} // extern "C"

// ═════════════════════════════════════════════════════════════════════════════
// Mock implementation — C++ linkage (BLEND)
// ═════════════════════════════════════════════════════════════════════════════

namespace blend {

void dispatch_blend(const double* left_ptr,
                    const double* right_ptr,
                    double* out_ptr,
                    std::size_t count,
                    double alpha,
                    int profile_tag)
{
    g_mock.call_sequence.push_back(MockCall::blend_dispatch);

    // Capture arguments for verification
    g_mock.blend_left_ptr   = left_ptr;
    g_mock.blend_right_ptr  = right_ptr;
    g_mock.blend_out_ptr    = out_ptr;
    g_mock.blend_count      = count;
    g_mock.blend_alpha      = alpha;
    g_mock.blend_profile_tag = profile_tag;

    if (g_mock.blend_should_throw) {
        throw std::runtime_error("BLEND dispatch failure (mock)");
    }
}

} // namespace blend

// ═════════════════════════════════════════════════════════════════════════════
// Test fixture
// ═════════════════════════════════════════════════════════════════════════════

class TransformNodeTest : public ::testing::Test {
protected:
    void SetUp() override {
        g_mock.reset();
    }

    /// Create a Stream_Descriptor with the given temporal profile.
    dagr::Stream_Descriptor make_stream(dagr::Temporal_Profile profile) {
        return dagr::Stream_Descriptor{
            .name = "test_stream",
            .temporal_profile = profile,
            .oob_policy = dagr::OutOfBounds_Policy::clamp,
            .dataset_path = "/data/test.zarr",
            .snapshot_interval = tick::Duration{3600'000'000'000LL}  // 1 hour
        };
    }

    /// Create a valid AliasedWindow with the given alpha.
    tick::AliasedWindow make_aliased(double alpha) {
        return tick::AliasedWindow{
            tick::Time_Window{tick::Time_Point{0}, tick::Time_Point{100}},
            alpha
        };
    }

    /// Invoke dispatch_temporal_bookend with default/configurable parameters.
    void invoke_dispatch(dagr::Temporal_Profile profile = dagr::Temporal_Profile::linear,
                         double alpha = 0.5) {
        auto stream = make_stream(profile);
        auto aliased = make_aliased(alpha);

        amio_dataset_handle dataset = reinterpret_cast<amio_dataset_handle>(0xDEAD);
        const char* var_name = "temperature";
        std::int64_t t_left = 0;
        std::int64_t t_right = 1;

        double output[MockState::kFieldSize] = {};
        dagr::detail::Completion_Token token{42, 1000};

        dagr::detail::dispatch_temporal_bookend(
            stream, aliased, dataset, var_name,
            t_left, t_right,
            output, MockState::kFieldSize,
            token);
    }
};

// ═════════════════════════════════════════════════════════════════════════════
// Test: Call sequence ordering (Req 4.1, 4.2, 4.3, 4.5)
// ═════════════════════════════════════════════════════════════════════════════

TEST_F(TransformNodeTest, CallSequence_LeftBeforeRight_QueryAfterFetch_RouteAfterQuery) {
    invoke_dispatch();

    // Verify the expected 5-call sequence:
    // 1. amio_read (left)
    // 2. amio_view_data (left)
    // 3. amio_read (right)
    // 4. amio_view_data (right)
    // 5. blend_dispatch
    ASSERT_EQ(g_mock.call_sequence.size(), 5u);

    // Left fetch happens first (amio_read + amio_view_data)
    EXPECT_EQ(g_mock.call_sequence[0], MockCall::amio_read_left);
    EXPECT_EQ(g_mock.call_sequence[1], MockCall::amio_view_data_left);

    // Right fetch happens second
    EXPECT_EQ(g_mock.call_sequence[2], MockCall::amio_read_right);
    EXPECT_EQ(g_mock.call_sequence[3], MockCall::amio_view_data_right);

    // BLEND dispatch happens last (after both fetches and alpha extraction)
    EXPECT_EQ(g_mock.call_sequence[4], MockCall::blend_dispatch);
}

TEST_F(TransformNodeTest, CallSequence_LeftFetchAlwaysPrecedesRightFetch) {
    invoke_dispatch();

    // Find indices of left vs right amio_read calls
    int left_read_idx = -1;
    int right_read_idx = -1;
    for (int i = 0; i < static_cast<int>(g_mock.call_sequence.size()); ++i) {
        if (g_mock.call_sequence[i] == MockCall::amio_read_left) left_read_idx = i;
        if (g_mock.call_sequence[i] == MockCall::amio_read_right) right_read_idx = i;
    }

    ASSERT_NE(left_read_idx, -1);
    ASSERT_NE(right_read_idx, -1);
    EXPECT_LT(left_read_idx, right_read_idx)
        << "Left bookend fetch must precede right bookend fetch (Req 4.1)";
}

TEST_F(TransformNodeTest, CallSequence_BlendDispatchIsLast) {
    invoke_dispatch();

    ASSERT_FALSE(g_mock.call_sequence.empty());
    EXPECT_EQ(g_mock.call_sequence.back(), MockCall::blend_dispatch)
        << "BLEND dispatch must be the final operation (Req 4.3)";
}

// ═════════════════════════════════════════════════════════════════════════════
// Test: Temporal_Profile::linear routes to linear-blend kernel (Req 4.4)
// ═════════════════════════════════════════════════════════════════════════════

TEST_F(TransformNodeTest, LinearProfile_RoutesToLinearBlendKernel) {
    invoke_dispatch(dagr::Temporal_Profile::linear);

    // profile_tag 0 = linear-blend kernel
    EXPECT_EQ(g_mock.blend_profile_tag, 0)
        << "Temporal_Profile::linear must route to profile_tag 0 (linear-blend)";
}

// ═════════════════════════════════════════════════════════════════════════════
// Test: Temporal_Profile::step routes to step-select kernel (Req 4.4)
// ═════════════════════════════════════════════════════════════════════════════

TEST_F(TransformNodeTest, StepProfile_RoutesToStepSelectKernel) {
    invoke_dispatch(dagr::Temporal_Profile::step);

    // profile_tag 1 = step-select kernel
    EXPECT_EQ(g_mock.blend_profile_tag, 1)
        << "Temporal_Profile::step must route to profile_tag 1 (step-select)";
}

// ═════════════════════════════════════════════════════════════════════════════
// Test: AMIO fetch failure propagates error (Req 4.8)
// ═════════════════════════════════════════════════════════════════════════════

TEST_F(TransformNodeTest, AmioReadLeftFailure_PropagatesError) {
    g_mock.amio_read_left_rc = AMIO_ERR_BACKEND_FAILURE;

    EXPECT_THROW(invoke_dispatch(), std::runtime_error)
        << "AMIO left fetch failure must propagate as std::runtime_error (Req 4.8)";

    // BLEND should never be called
    for (const auto& call : g_mock.call_sequence) {
        EXPECT_NE(call, MockCall::blend_dispatch)
            << "BLEND must not be dispatched when AMIO fetch fails";
    }
}

TEST_F(TransformNodeTest, AmioReadRightFailure_PropagatesError) {
    g_mock.amio_read_right_rc = AMIO_ERR_TIMEOUT;

    EXPECT_THROW(invoke_dispatch(), std::runtime_error)
        << "AMIO right fetch failure must propagate as std::runtime_error (Req 4.8)";

    // Left read should have succeeded, right failed
    EXPECT_GE(g_mock.amio_read_call_count, 2);

    // BLEND should never be called
    for (const auto& call : g_mock.call_sequence) {
        EXPECT_NE(call, MockCall::blend_dispatch);
    }
}

TEST_F(TransformNodeTest, AmioViewDataLeftFailure_PropagatesError) {
    g_mock.amio_view_data_left_rc = AMIO_ERR_INVALID_HANDLE;

    EXPECT_THROW(invoke_dispatch(), std::runtime_error)
        << "AMIO view_data failure must propagate as std::runtime_error (Req 4.8)";
}

TEST_F(TransformNodeTest, AmioViewDataRightFailure_PropagatesError) {
    g_mock.amio_view_data_right_rc = AMIO_ERR_INVALID_HANDLE;

    EXPECT_THROW(invoke_dispatch(), std::runtime_error)
        << "AMIO right view_data failure must propagate as std::runtime_error (Req 4.8)";
}

// ═════════════════════════════════════════════════════════════════════════════
// Test: TICK query failure propagates error (Req 4.9)
//
// Note: In the current implementation, the AliasedWindow is passed in
// pre-computed (alpha already extracted by the caller). The TICK query
// failure would manifest as an exception before dispatch_temporal_bookend
// is called. We verify that if a malformed AliasedWindow is provided,
// the function still operates correctly (passing alpha through without
// modification). True TICK failure testing is at the integration level.
// For this unit test, we verify the alpha path is pass-through only.
// ═════════════════════════════════════════════════════════════════════════════

TEST_F(TransformNodeTest, AlphaIsPassedThroughWithoutModification) {
    // Use a distinctive alpha value
    const double test_alpha = 0.73;
    invoke_dispatch(dagr::Temporal_Profile::linear, test_alpha);

    // BLEND must receive the exact same alpha — no arithmetic modification
    EXPECT_EQ(g_mock.blend_alpha, test_alpha)
        << "Alpha must be passed through to BLEND without modification (Req 4.2, 7.3, 7.5)";
}

TEST_F(TransformNodeTest, AlphaZero_PassedThrough) {
    invoke_dispatch(dagr::Temporal_Profile::linear, 0.0);
    EXPECT_EQ(g_mock.blend_alpha, 0.0);
}

TEST_F(TransformNodeTest, AlphaOne_PassedThrough) {
    invoke_dispatch(dagr::Temporal_Profile::step, 1.0);
    EXPECT_EQ(g_mock.blend_alpha, 1.0);
}

// ═════════════════════════════════════════════════════════════════════════════
// Test: BLEND execution failure propagates error (Req 4.10)
// ═════════════════════════════════════════════════════════════════════════════

TEST_F(TransformNodeTest, BlendFailure_PropagatesError) {
    g_mock.blend_should_throw = true;

    EXPECT_THROW(invoke_dispatch(), std::runtime_error)
        << "BLEND execution failure must propagate as std::runtime_error (Req 4.10)";
}

TEST_F(TransformNodeTest, BlendFailure_AmioCallsStillComplete) {
    g_mock.blend_should_throw = true;

    try {
        invoke_dispatch();
        FAIL() << "Expected std::runtime_error from BLEND";
    } catch (const std::runtime_error&) {
        // AMIO reads should have completed before BLEND failure
        EXPECT_EQ(g_mock.amio_read_call_count, 2)
            << "Both AMIO reads must complete before BLEND dispatch";
        EXPECT_EQ(g_mock.amio_view_data_call_count, 2)
            << "Both AMIO view_data calls must complete before BLEND dispatch";
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Test: Zero-computation guarantee (Req 4.5, 7.3, 7.4, 7.5)
//
// DAGR must not perform arithmetic on alpha or FieldView data. It only
// routes pointers. We verify by checking that:
// 1. Alpha is forwarded to BLEND exactly as received (bitwise identical)
// 2. BLEND receives pointers to the original AMIO buffers (not copies
//    with modified values)
// 3. No element-wise loops over field data (verified by checking that
//    output buffer is untouched by DAGR — only BLEND writes to it)
// ═════════════════════════════════════════════════════════════════════════════

TEST_F(TransformNodeTest, ZeroComputation_AlphaNotModified) {
    // Use a precise alpha that would be changed by any arithmetic
    const double precise_alpha = 0.123456789012345;
    invoke_dispatch(dagr::Temporal_Profile::linear, precise_alpha);

    // Bitwise comparison — DAGR must not touch alpha at all
    double received = g_mock.blend_alpha;
    EXPECT_EQ(std::memcmp(&received, &precise_alpha, sizeof(double)), 0)
        << "Alpha must be bitwise identical — DAGR performs zero arithmetic (Req 7.3, 7.5)";
}

TEST_F(TransformNodeTest, ZeroComputation_FieldDataPointersRoutedDirectly) {
    invoke_dispatch();

    // BLEND must receive the exact pointers from AMIO's buffers
    EXPECT_EQ(g_mock.blend_left_ptr, g_mock.left_data)
        << "Left FieldView pointer must be routed directly from AMIO to BLEND (Req 4.5)";
    EXPECT_EQ(g_mock.blend_right_ptr, g_mock.right_data)
        << "Right FieldView pointer must be routed directly from AMIO to BLEND (Req 4.5)";
}

TEST_F(TransformNodeTest, ZeroComputation_FieldDataValuesUnmodifiedByDAGR) {
    // Set known values in the AMIO buffers
    g_mock.left_data[0] = 42.0;
    g_mock.left_data[1] = 99.0;
    g_mock.right_data[0] = -1.5;
    g_mock.right_data[1] = 3.14;

    invoke_dispatch();

    // After dispatch, AMIO buffer values must be unchanged (DAGR didn't touch them)
    EXPECT_EQ(g_mock.left_data[0], 42.0);
    EXPECT_EQ(g_mock.left_data[1], 99.0);
    EXPECT_EQ(g_mock.right_data[0], -1.5);
    EXPECT_EQ(g_mock.right_data[1], 3.14);
}

TEST_F(TransformNodeTest, ZeroComputation_OutputBufferUntouchedByDAGR) {
    // Pre-fill output with sentinel values
    double output[MockState::kFieldSize];
    std::memset(output, 0xAB, sizeof(output));

    // Save a copy to compare against after the mock BLEND (which doesn't write)
    double expected[MockState::kFieldSize];
    std::memcpy(expected, output, sizeof(output));

    auto stream = make_stream(dagr::Temporal_Profile::linear);
    auto aliased = make_aliased(0.5);
    amio_dataset_handle dataset = reinterpret_cast<amio_dataset_handle>(0xDEAD);
    dagr::detail::Completion_Token token{0, 0};

    dagr::detail::dispatch_temporal_bookend(
        stream, aliased, dataset, "var",
        0, 1, output, MockState::kFieldSize, token);

    // DAGR must not write to the output buffer — only BLEND does (and our mock doesn't)
    EXPECT_EQ(std::memcmp(output, expected, sizeof(output)), 0)
        << "DAGR must not write to output buffer — only BLEND may (Req 7.4)";
}

TEST_F(TransformNodeTest, ZeroComputation_ElementCountPassedThrough) {
    const std::size_t expected_count = MockState::kFieldSize;
    invoke_dispatch();

    EXPECT_EQ(g_mock.blend_count, expected_count)
        << "Element count must be passed through to BLEND without modification";
}

// ═════════════════════════════════════════════════════════════════════════════
// Test: Both profiles produce correct routing for various alpha values
// ═════════════════════════════════════════════════════════════════════════════

TEST_F(TransformNodeTest, LinearProfile_WithBoundaryAlpha) {
    invoke_dispatch(dagr::Temporal_Profile::linear, 0.0);
    EXPECT_EQ(g_mock.blend_profile_tag, 0);
    EXPECT_EQ(g_mock.blend_alpha, 0.0);
}

TEST_F(TransformNodeTest, StepProfile_WithBoundaryAlpha) {
    invoke_dispatch(dagr::Temporal_Profile::step, 0.5);
    EXPECT_EQ(g_mock.blend_profile_tag, 1);
    EXPECT_EQ(g_mock.blend_alpha, 0.5);
}
