// ─── Property-Based Tests: TripleBuffer Construction Validation ──────────────
// Feature: span-field-view, Property 9: TripleBuffer Construction Validation
//
// For any input to the TripleBuffer constructor where at least one buffer pointer
// is null or n_elements is zero, construction SHALL fail with an error indication.
// For any valid input (three distinct non-null pointers, n_elements > 0),
// construction SHALL succeed with write_ptr(), read_ptr(), and io_ptr() each
// returning distinct non-null values.
//
// **Validates: Requirements 5.2, 5.3, 5.4, 5.5, 5.11**
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <cstddef>
#include <span/triple_buffer.hpp>
#include <stdexcept>
#include <vector>

namespace {

// Helper: generate a random buffer size in [1, 1000]
rc::Gen<std::size_t> genBufferSize() {
    return rc::gen::inRange<std::size_t>(1, 1001);
}

}  // namespace

// ─── Valid Construction ──────────────────────────────────────────────────────
// For any three distinct heap buffers with random size in [1, 1000],
// constructing a TripleBuffer SHALL succeed and the accessors shall return
// distinct non-null pointers with size() matching n_elements and
// is_io_locked() returning false.

RC_GTEST_PROP(TripleBufferConstruction, ValidInputsSucceed, ()) {
    const auto n_elements = *genBufferSize();

    // Allocate three distinct buffers on the heap
    std::vector<double> buf_write(n_elements);
    std::vector<double> buf_read(n_elements);
    std::vector<double> buf_io(n_elements);

    // Pointers are guaranteed distinct (separate heap allocations)
    span::TripleBuffer<double> tb(buf_write.data(), buf_read.data(), buf_io.data(), n_elements);

    // write_ptr(), read_ptr(), io_ptr() are non-null
    RC_ASSERT(tb.write_ptr() != nullptr);
    RC_ASSERT(tb.read_ptr() != nullptr);
    RC_ASSERT(tb.io_ptr() != nullptr);

    // All three pointers are distinct
    RC_ASSERT(tb.write_ptr() != tb.read_ptr());
    RC_ASSERT(tb.write_ptr() != tb.io_ptr());
    RC_ASSERT(tb.read_ptr() != tb.io_ptr());

    // Pointers match the original buffers
    RC_ASSERT(tb.write_ptr() == buf_write.data());
    RC_ASSERT(tb.read_ptr() == buf_read.data());
    RC_ASSERT(tb.io_ptr() == buf_io.data());

    // size() matches n_elements
    RC_ASSERT(tb.size() == n_elements);

    // is_io_locked() starts false
    RC_ASSERT(!tb.is_io_locked());
}

// ─── Invalid Construction: Null Pointers and Zero Elements ───────────────────
// For any input where at least one buffer pointer is null, n_elements is zero,
// or pointers are not distinct, construction SHALL throw std::invalid_argument.

RC_GTEST_PROP(TripleBufferConstruction, NullWritePtrThrows, ()) {
    const auto n_elements = *genBufferSize();
    std::vector<double> buf_read(n_elements);
    std::vector<double> buf_io(n_elements);

    RC_ASSERT_THROWS_AS(span::TripleBuffer<double>(nullptr, buf_read.data(), buf_io.data(), n_elements), std::invalid_argument);
}

RC_GTEST_PROP(TripleBufferConstruction, NullReadPtrThrows, ()) {
    const auto n_elements = *genBufferSize();
    std::vector<double> buf_write(n_elements);
    std::vector<double> buf_io(n_elements);

    RC_ASSERT_THROWS_AS(span::TripleBuffer<double>(buf_write.data(), nullptr, buf_io.data(), n_elements), std::invalid_argument);
}

RC_GTEST_PROP(TripleBufferConstruction, NullIoPtrThrows, ()) {
    const auto n_elements = *genBufferSize();
    std::vector<double> buf_write(n_elements);
    std::vector<double> buf_read(n_elements);

    RC_ASSERT_THROWS_AS(span::TripleBuffer<double>(buf_write.data(), buf_read.data(), nullptr, n_elements), std::invalid_argument);
}

RC_GTEST_PROP(TripleBufferConstruction, ZeroElementsThrows, ()) {
    std::vector<double> buf_write(1);
    std::vector<double> buf_read(1);
    std::vector<double> buf_io(1);

    RC_ASSERT_THROWS_AS(span::TripleBuffer<double>(buf_write.data(), buf_read.data(), buf_io.data(), 0), std::invalid_argument);
}

RC_GTEST_PROP(TripleBufferConstruction, DuplicatePointersThrow, ()) {
    const auto n_elements = *genBufferSize();
    std::vector<double> buf_a(n_elements);
    std::vector<double> buf_b(n_elements);

    // write == read
    RC_ASSERT_THROWS_AS(span::TripleBuffer<double>(buf_a.data(), buf_a.data(), buf_b.data(), n_elements), std::invalid_argument);

    // write == io
    RC_ASSERT_THROWS_AS(span::TripleBuffer<double>(buf_a.data(), buf_b.data(), buf_a.data(), n_elements), std::invalid_argument);

    // read == io
    RC_ASSERT_THROWS_AS(span::TripleBuffer<double>(buf_b.data(), buf_a.data(), buf_a.data(), n_elements), std::invalid_argument);
}
