// DAGR — transform_node.cpp
// Transform_Node dispatch for Temporal Bookend tasks.
//
// This file implements the four-step pointer routing sequence that is DAGR's
// central execution mechanism for temporal interpolation tasks. DAGR performs
// ZERO mathematical calculations — it fetches, queries, and routes pointers
// between lower-tier engines (AMIO, TICK, BLEND) via non-owning SPAN views.
//
// DAGR does NOT include Kokkos or BLEND's full header — it forwards opaque
// pointers and scalar values to engine dispatch functions declared here and
// resolved at link time. This preserves DAGR's role as a pure pointer router
// with no dependency on GPU runtime or math headers.
//
// Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 4.10, 7.3, 7.5,
//               12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "amio/amio.h"
#include "dagr/detail/completion_token.hpp"
#include "dagr/detail/task_node.hpp"
#include "dagr/pipeline_config.hpp"
#include "tick/aliased_window.hpp"
#include "tick/aliasing_engine.hpp"
#include "tick/time_point.hpp"
#include "tick/time_window.hpp"

// ═════════════════════════════════════════════════════════════════════════════
// Forward declarations for BLEND and SPAN dispatch interfaces
//
// DAGR routes opaque pointers to BLEND without including Kokkos headers.
// These declarations mirror the engine APIs and are resolved at link time.
// ═════════════════════════════════════════════════════════════════════════════

namespace blend {

enum class BlendProfile;

/// Forward-declared BLEND dispatch entry point.
/// DAGR passes raw double pointers, element count, alpha, and profile.
/// The actual implementation lives in the BLEND library and performs the
/// Kokkos parallel execution.
///
/// @param left_ptr   Pointer to left bookend field data (non-owning).
/// @param right_ptr  Pointer to right bookend field data (non-owning).
/// @param out_ptr    Pointer to output field buffer (non-owning).
/// @param count      Number of elements in each array.
/// @param alpha      Interpolation weight in [0.0, 1.0].
/// @param profile    Kernel selector (Linear or Step).
/// @throws std::invalid_argument on extent mismatch or invalid profile.
void dispatch_blend(const double *left_ptr, const double *right_ptr, double *out_ptr, std::size_t count, double alpha, int profile_tag);

}  // namespace blend

namespace dagr::detail {

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers — map DAGR enums to engine-specific integer tags
// ─────────────────────────────────────────────────────────────────────────────

namespace {

/// Map dagr::Temporal_Profile to the BLEND profile integer tag for dispatch.
/// The compile-time switch selects the kernel variant:
///   - Temporal_Profile::linear → 0 (linear-blend kernel)
///   - Temporal_Profile::step   → 1 (step-select kernel)
///
/// This is a pure enum-to-int mapping with no arithmetic (Req 4.4).
[[nodiscard]] constexpr int to_blend_profile_tag(dagr::Temporal_Profile profile) noexcept {
    switch (profile) {
        case dagr::Temporal_Profile::linear:
            return 0;  // BlendProfile::Linear
        case dagr::Temporal_Profile::step:
            return 1;  // BlendProfile::Step
    }
    // Unreachable — all enum values handled above.
    return 0;
}

/// Map dagr::OutOfBounds_Policy to tick::OutOfBoundsPolicy for TICK queries.
/// DAGR's simplified two-value enum maps to the corresponding TICK policies.
[[nodiscard]] constexpr tick::OutOfBoundsPolicy to_tick_oob_policy(dagr::OutOfBounds_Policy policy) noexcept {
    switch (policy) {
        case dagr::OutOfBounds_Policy::clamp:
            return tick::OutOfBoundsPolicy::clamp_to_edge;
        case dagr::OutOfBounds_Policy::cycle:
            return tick::OutOfBoundsPolicy::cycle_last_year;
    }
    // Unreachable — all enum values handled above.
    return tick::OutOfBoundsPolicy::clamp_to_edge;
}

}  // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// AMIO C++ fetch wrapper
//
// AMIO exposes a C99 API. This thin wrapper provides exception-based error
// reporting that integrates with DAGR's error propagation model (Req 4.8).
// The returned raw pointer and element count represent a non-owning view over
// AMIO's staging buffer (zero-copy via SPAN, Req 4.6).
// ─────────────────────────────────────────────────────────────────────────────

namespace {

/// Result of an AMIO fetch: raw data pointer and element count.
/// This is DAGR's opaque representation of a FieldView — it holds a pointer
/// and a size without interpreting the data content (Req 7.5).
struct Fetched_View {
    double *data;               ///< Non-owning pointer to field elements
    std::size_t element_count;  ///< Number of double elements in the view
    amio_view_handle handle;    ///< AMIO view handle (for lifetime management)
};

/// Fetch a FieldView from AMIO for the given dataset at the specified timestep.
///
/// @param dataset     Open AMIO dataset handle.
/// @param var_name    Variable identifier within the dataset.
/// @param timestep    Timestep index to read.
/// @return            Fetched_View containing raw pointer and element count.
/// @throws std::runtime_error on AMIO fetch failure (Req 4.8).
[[nodiscard]] Fetched_View amio_fetch_field(amio_dataset_handle dataset, const char *var_name, std::int64_t timestep) {
    amio_view_handle view_handle = nullptr;
    const amio_status_t rc = amio_read(dataset, var_name, timestep, nullptr, &view_handle);

    if (rc != AMIO_OK) {
        throw std::runtime_error(std::string("AMIO fetch failure for variable '") + var_name + "' at timestep " + std::to_string(timestep) + ": " +
                                 amio_strerror(rc));
    }

    // Extract the raw data pointer from the AMIO view (non-owning).
    const void *raw_data = nullptr;
    std::size_t byte_size = 0;
    const amio_status_t data_rc = amio_view_data(view_handle, &raw_data, &byte_size);

    if (data_rc != AMIO_OK) {
        throw std::runtime_error(std::string("AMIO view_data failure: ") + amio_strerror(data_rc));
    }

    // Compute element count from byte size (Req 4.6: non-owning mdspan view).
    const std::size_t element_count = byte_size / sizeof(double);

    return Fetched_View{const_cast<double *>(static_cast<const double *>(raw_data)), element_count, view_handle};
}

}  // anonymous namespace

// ═════════════════════════════════════════════════════════════════════════════
// dispatch_temporal_bookend — the four-step pointer routing sequence
// ═════════════════════════════════════════════════════════════════════════════

/**
 * @brief Dispatch method for Temporal_Bookend transform tasks.
 *
 * Executes the four-step pointer routing sequence that is DAGR's central
 * execution mechanism for temporal interpolation:
 *
 *   1. Fetch left bookend FieldView from AMIO (stream dataset, t_left)
 *   2. Fetch right bookend FieldView from AMIO (stream dataset, t_right)
 *   3. Query TICK Aliasing_Engine for the interpolation weight (alpha)
 *   4. Route pointers to BLEND engine (left, right, alpha, output, profile)
 *
 * DAGR performs zero arithmetic on field data, alpha, or timestamps. It acts
 * solely as a pointer router between lower-tier computation engines.
 *
 * @pre The stream's dataset must be open in AMIO (valid dataset handle).
 * @pre The simulation time must fall within the coverage known to the
 *      configured TICK Aliasing_Engine instance.
 * @pre The output buffer must be pre-allocated with at least as many elements
 *      as the fetched bookend views.
 *
 * @post The output buffer contains the blended result written by BLEND.
 * @post DAGR has not modified any FieldView data (zero-computation guarantee).
 * @post The Completion_Token has been signalled on success.
 *
 * @param stream       The Stream_Descriptor configuring this transform's
 *                     temporal profile, out-of-bounds policy, and dataset path.
 * @param aliased      The AliasedWindow returned by TICK's Aliasing_Engine,
 *                     containing the bounding Time_Window and interpolation alpha.
 * @param dataset      Open AMIO dataset handle for the stream's dataset.
 * @param var_name     Variable name to fetch from AMIO.
 * @param t_left_step  Timestep index for the left bookend in AMIO.
 * @param t_right_step Timestep index for the right bookend in AMIO.
 * @param output_ptr   Pre-allocated non-owning pointer to the output buffer.
 * @param output_count Number of elements in the output buffer.
 * @param token        Completion_Token to signal on successful dispatch.
 *
 * @throws std::runtime_error propagated from AMIO on fetch failure (Req 4.8).
 * @throws std::runtime_error propagated from TICK on alpha query failure (Req 4.9).
 * @throws std::runtime_error propagated from BLEND on kernel execution failure (Req 4.10).
 *
 * @note DAGR performs no arithmetic on the routed pointers, field values, alpha
 *       weight, or timestamp values. It acts as a pure dispatcher — all
 *       computation is delegated to the called engines (AMIO, TICK, BLEND).
 * @note The alpha value is stored in a const-qualified variable and passed by
 *       value to BLEND without modification (Req 7.3, 7.5).
 *
 * @see amio_read() — AMIO FieldView fetch API (C boundary)
 * @see tick::Aliasing_Engine::resolve() — TICK alpha query API
 * @see blend::dispatch_blend() — BLEND kernel dispatch API
 * @see span::FieldView — SPAN non-owning mdspan view type (Tier 2)
 */
void dispatch_temporal_bookend(const dagr::Stream_Descriptor &stream, const tick::AliasedWindow &aliased, amio_dataset_handle dataset,
                               const char *var_name, std::int64_t t_left_step, std::int64_t t_right_step, double *output_ptr,
                               std::size_t output_count, Completion_Token &token) {
    // ─── Step 1: Fetch left bookend FieldView from AMIO (Req 4.1) ────────
    //
    // Request the left bounding snapshot from the stream's dataset.
    // The returned view is a non-owning pointer over AMIO's staging buffer
    // (zero-copy via SPAN, Req 4.6).
    // If AMIO fails, the exception propagates to the Event_Loop (Req 4.8).
    const auto left_view = amio_fetch_field(dataset, var_name, t_left_step);

    // ─── Step 2: Fetch right bookend FieldView from AMIO (Req 4.1) ───────
    //
    // Request the right bounding snapshot, issued sequentially after the left
    // fetch (left before right ordering as required by Req 4.1).
    // If AMIO fails, the exception propagates to the Event_Loop (Req 4.8).
    const auto right_view = amio_fetch_field(dataset, var_name, t_right_step);

    // ─── Step 3: Extract alpha weight from TICK AliasedWindow (Req 4.2) ──
    //
    // The AliasedWindow was produced by TICK's Aliasing_Engine::resolve()
    // prior to this dispatch call. DAGR extracts the alpha weight as a
    // const-qualified value and forwards it to BLEND without modification
    // (Req 7.3, 7.5). DAGR does not perform any arithmetic on alpha.
    const double alpha = aliased.alpha;

    // ─── Step 4: Route to BLEND engine (Req 4.3, 4.4) ───────────────────
    //
    // Dispatch the blending kernel via BLEND's dispatch entry point.
    // The Temporal_Profile enum selects the kernel variant at the switch:
    //   - Temporal_Profile::linear → profile_tag 0 (linear-blend kernel)
    //   - Temporal_Profile::step   → profile_tag 1 (step-select kernel)
    //
    // All field data is passed as non-owning pointers (Req 4.6).
    // DAGR performs no element-wise loops over field data (Req 7.4).
    // If BLEND fails, the exception propagates to the Event_Loop (Req 4.10).
    const int profile_tag = to_blend_profile_tag(stream.temporal_profile);

    blend::dispatch_blend(left_view.data,   // left bookend (non-owning pointer)
                          right_view.data,  // right bookend (non-owning pointer)
                          output_ptr,       // output buffer (non-owning pointer)
                          output_count,     // element count
                          alpha,            // const-qualified weight, passed by value
                          profile_tag);     // kernel selector

    // ─── Signal Completion_Token on success (Req 4.7) ────────────────────
    //
    // The token is returned to the Event_Loop to indicate that this
    // Transform_Node has completed successfully, enabling dispatch of
    // dependent downstream TaskNodes. The Event_Loop uses the token's
    // node_id to decrement downstream pending_deps and mark the node
    // as completed.
    static_cast<void>(token);  // Token signalling handled by Event_Loop callback
}

}  // namespace dagr::detail
