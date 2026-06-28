// ─── Property-Based Tests: Triple-Buffer Pointer Rotation ───────────────────
// Feature: span-field-view, Property 6: Triple-Buffer Pointer Rotation
//
// For any sequence of swap_buffers_for_amio() and release_io() operations on a
// TripleBuffer initialized with three distinct pointers {A, B, C}:
// (a) each successful swap rotates the tracks as Write→Read→IO→Write
// (b) no buffer content is copied (pointer addresses at original memory locations unchanged)
// (c) is_io_locked() returns true after a successful swap and false after release_io()
// (d) a swap while IO is locked returns false with all three pointer tracks unchanged
//
// **Validates: Requirements 5.6, 5.7, 5.10, 6.2, 6.3, 8.5**
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <array>
#include <cstddef>
#include <span/triple_buffer.hpp>
#include <vector>

namespace {

// Size of each buffer track for testing
constexpr std::size_t BUF_ELEMENTS = 16;

// Helper struct tracking expected pointer positions through rotations
struct TripleBufferModel {
    double *write;
    double *read;
    double *io;
    bool locked;

    // Apply a swap according to the spec rotation model
    bool swap() {
        if (locked) return false;
        // Rotation: Write→Read→IO→Write
        double *old_write = write;
        double *old_read = read;
        double *old_io = io;
        read = old_write;  // old Write → new Read
        io = old_read;     // old Read  → new IO
        write = old_io;    // old IO    → new Write
        locked = true;
        return true;
    }

    void release() {
        locked = false;
    }
};

// Operations for random sequence generation
enum class Op : int { Swap, Release };

}  // namespace

// ─── Single Rotation ─────────────────────────────────────────────────────────
// Initialize with {A, B, C}, swap, verify new Write=C, Read=A, IO=B, locked=true

RC_GTEST_PROP(TripleBufferRotation, SingleRotation, ()) {
    std::vector<double> bufA(BUF_ELEMENTS, 1.0);
    std::vector<double> bufB(BUF_ELEMENTS, 2.0);
    std::vector<double> bufC(BUF_ELEMENTS, 3.0);

    double *A = bufA.data();
    double *B = bufB.data();
    double *C = bufC.data();

    span::TripleBuffer<double> tb(A, B, C, BUF_ELEMENTS);

    // Preconditions: initial state
    RC_ASSERT(tb.write_ptr() == A);
    RC_ASSERT(tb.read_ptr() == B);
    RC_ASSERT(tb.io_ptr() == C);
    RC_ASSERT(!tb.is_io_locked());

    // Perform swap
    bool result = tb.swap_buffers_for_amio();
    RC_ASSERT(result == true);

    // Postconditions after rotation: Write→Read→IO→Write
    // Old Write(A) → new Read
    // Old Read(B)  → new IO
    // Old IO(C)    → new Write
    RC_ASSERT(tb.write_ptr() == C);
    RC_ASSERT(tb.read_ptr() == A);
    RC_ASSERT(tb.io_ptr() == B);
    RC_ASSERT(tb.is_io_locked());

    // Verify no data was copied — original buffer contents unchanged
    RC_ASSERT(bufA[0] == 1.0);
    RC_ASSERT(bufB[0] == 2.0);
    RC_ASSERT(bufC[0] == 3.0);
}

// ─── Locked Swap Returns False ───────────────────────────────────────────────
// After swap (locked), another swap returns false, pointers unchanged

RC_GTEST_PROP(TripleBufferRotation, LockedSwapReturnsFalse, ()) {
    std::vector<double> bufA(BUF_ELEMENTS, 1.0);
    std::vector<double> bufB(BUF_ELEMENTS, 2.0);
    std::vector<double> bufC(BUF_ELEMENTS, 3.0);

    span::TripleBuffer<double> tb(bufA.data(), bufB.data(), bufC.data(), BUF_ELEMENTS);

    // First swap succeeds
    RC_ASSERT(tb.swap_buffers_for_amio() == true);
    RC_ASSERT(tb.is_io_locked());

    // Capture pointer state after first swap
    double *w_after = tb.write_ptr();
    double *r_after = tb.read_ptr();
    double *io_after = tb.io_ptr();

    // Second swap while locked returns false
    bool result = tb.swap_buffers_for_amio();
    RC_ASSERT(result == false);

    // All pointers unchanged
    RC_ASSERT(tb.write_ptr() == w_after);
    RC_ASSERT(tb.read_ptr() == r_after);
    RC_ASSERT(tb.io_ptr() == io_after);

    // Still locked
    RC_ASSERT(tb.is_io_locked());
}

// ─── Release Then Swap ───────────────────────────────────────────────────────
// After release_io(), swap succeeds again with correct rotation

RC_GTEST_PROP(TripleBufferRotation, ReleaseThenSwap, ()) {
    std::vector<double> bufA(BUF_ELEMENTS, 1.0);
    std::vector<double> bufB(BUF_ELEMENTS, 2.0);
    std::vector<double> bufC(BUF_ELEMENTS, 3.0);

    double *A = bufA.data();
    double *B = bufB.data();
    double *C = bufC.data();

    span::TripleBuffer<double> tb(A, B, C, BUF_ELEMENTS);

    // First swap: Write=A,Read=B,IO=C → Write=C,Read=A,IO=B (locked)
    RC_ASSERT(tb.swap_buffers_for_amio() == true);
    RC_ASSERT(tb.is_io_locked());
    RC_ASSERT(tb.write_ptr() == C);
    RC_ASSERT(tb.read_ptr() == A);
    RC_ASSERT(tb.io_ptr() == B);

    // Release
    tb.release_io();
    RC_ASSERT(!tb.is_io_locked());

    // Pointers unchanged after release (only lock flag changes)
    RC_ASSERT(tb.write_ptr() == C);
    RC_ASSERT(tb.read_ptr() == A);
    RC_ASSERT(tb.io_ptr() == B);

    // Second swap: Write=C,Read=A,IO=B → Write=B,Read=C,IO=A (locked)
    RC_ASSERT(tb.swap_buffers_for_amio() == true);
    RC_ASSERT(tb.is_io_locked());
    RC_ASSERT(tb.write_ptr() == B);
    RC_ASSERT(tb.read_ptr() == C);
    RC_ASSERT(tb.io_ptr() == A);
}

// ─── Random Operation Sequences ─────────────────────────────────────────────
// Generate random sequences of swap/release, maintain a model of expected
// pointers, verify implementation matches at every step.

RC_GTEST_PROP(TripleBufferRotation, RandomOperationSequences, ()) {
    std::vector<double> bufA(BUF_ELEMENTS, 1.0);
    std::vector<double> bufB(BUF_ELEMENTS, 2.0);
    std::vector<double> bufC(BUF_ELEMENTS, 3.0);

    double *A = bufA.data();
    double *B = bufB.data();
    double *C = bufC.data();

    span::TripleBuffer<double> tb(A, B, C, BUF_ELEMENTS);

    // Reference model tracking expected state
    TripleBufferModel model{A, B, C, false};

    // Generate random operation sequence (length 1..50)
    const auto len = *rc::gen::inRange<std::size_t>(1, 51);
    const auto ops = *rc::gen::container<std::vector<Op>>(len, rc::gen::element(Op::Swap, Op::Release));

    for (const auto &op : ops) {
        switch (op) {
            case Op::Swap: {
                bool model_result = model.swap();
                bool impl_result = tb.swap_buffers_for_amio();
                RC_ASSERT(impl_result == model_result);
                break;
            }
            case Op::Release: {
                model.release();
                tb.release_io();
                break;
            }
        }

        // After each operation, verify implementation matches model
        RC_ASSERT(tb.write_ptr() == model.write);
        RC_ASSERT(tb.read_ptr() == model.read);
        RC_ASSERT(tb.io_ptr() == model.io);
        RC_ASSERT(tb.is_io_locked() == model.locked);
    }

    // Verify zero-copy: original buffer data unchanged throughout
    RC_ASSERT(bufA[0] == 1.0);
    RC_ASSERT(bufB[0] == 2.0);
    RC_ASSERT(bufC[0] == 3.0);
}
