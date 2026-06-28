// ─── Property-Based Tests: Coherency State Machine ──────────────────────────
// Feature: span-field-view, Property 4: Coherency State Machine Invariant
//
// For any sequence of mark_host_dirty(), mark_device_dirty(), sync_for_device(),
// and sync_for_host() operations applied to a FieldView, the coherency state
// after each operation SHALL satisfy:
// (a) mark_host_dirty() always produces HOST_DIRTY
// (b) mark_device_dirty() always produces DEVICE_DIRTY
// (c) sync_for_device() from HOST_DIRTY produces HOST_CLEAN
// (d) sync_for_host() from DEVICE_DIRTY produces HOST_CLEAN
// (e) all other sync calls are no-ops (state unchanged)
// (f) at most one of host or device is dirty at any point in the sequence
//
// **Validates: Requirements 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.11**
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <span/field_view.hpp>
#include <vector>

namespace {

enum class Op { MarkHostDirty, MarkDeviceDirty, SyncForDevice, SyncForHost };

rc::Gen<Op> genOp() {
    return rc::gen::element(Op::MarkHostDirty, Op::MarkDeviceDirty, Op::SyncForDevice, Op::SyncForHost);
}

// Model: compute expected state after applying operation to current state
span::CoherencyState model_transition(span::CoherencyState current, Op op) {
    switch (op) {
        case Op::MarkHostDirty:
            return span::CoherencyState::HOST_DIRTY;
        case Op::MarkDeviceDirty:
            return span::CoherencyState::DEVICE_DIRTY;
        case Op::SyncForDevice:
            return (current == span::CoherencyState::HOST_DIRTY) ? span::CoherencyState::HOST_CLEAN : current;
        case Op::SyncForHost:
            return (current == span::CoherencyState::DEVICE_DIRTY) ? span::CoherencyState::HOST_CLEAN : current;
    }
    return current;
}

// Invariant (f): at most one of host or device is dirty
// Since CoherencyState is a single enum value, it can never simultaneously be
// HOST_DIRTY and DEVICE_DIRTY — but we verify the model and implementation agree.
bool invariant_at_most_one_dirty(span::CoherencyState state) {
    // The state is a single enum value, so it is inherently impossible to be
    // both HOST_DIRTY and DEVICE_DIRTY. We verify it is one of the valid values.
    return state == span::CoherencyState::HOST_CLEAN || state == span::CoherencyState::HOST_DIRTY || state == span::CoherencyState::DEVICE_DIRTY;
}

}  // namespace

// ─── Dual-Pointer: Full State Machine Sequence ──────────────────────────────

RC_GTEST_PROP(CoherencyStateMachine, DualPointerSequence, ()) {
    // Setup: dual-pointer FieldView (host + device) so syncs actually transition
    constexpr std::size_t N = 100;
    std::vector<double> host_buf(N, 0.0);
    std::vector<double> dev_buf(N, 0.0);
    std::array<std::size_t, 1> exts{N};
    span::FieldView<double, 1> fv(host_buf.data(), dev_buf.data(), exts);

    // Generate random operation sequence of length [10, 100]
    const auto seq_len = *rc::gen::inRange<std::size_t>(10, 101);
    const auto ops = *rc::gen::container<std::vector<Op>>(seq_len, genOp());

    auto expected_state = span::CoherencyState::HOST_CLEAN;

    for (auto op : ops) {
        // Apply operation to the real FieldView
        switch (op) {
            case Op::MarkHostDirty:
                fv.mark_host_dirty();
                break;
            case Op::MarkDeviceDirty:
                fv.mark_device_dirty();
                break;
            case Op::SyncForDevice:
                fv.sync_for_device();
                break;
            case Op::SyncForHost:
                fv.sync_for_host();
                break;
        }

        // Compute expected state via the model
        expected_state = model_transition(expected_state, op);

        // Assert implementation matches model
        RC_ASSERT(fv.coherency_state() == expected_state);

        // Assert invariant (f): at most one dirty flag
        RC_ASSERT(invariant_at_most_one_dirty(fv.coherency_state()));
    }
}
