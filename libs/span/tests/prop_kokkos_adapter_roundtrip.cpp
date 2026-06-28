// ─── Property-Based Tests: Kokkos Adapter Round-Trip ────────────────────────
// Feature: span-field-view, Property 7: Kokkos Adapter Round-Trip
//
// For any valid FieldView<T, Rank> with a non-null host pointer, calling
// to_kokkos_view() and then from_kokkos_view() on the result SHALL produce a
// FieldView whose host_data() is identical to the original's host_data().
// Neither adapter call SHALL modify the coherency state or invoke
// Kokkos::deep_copy.
//
// **Validates: Requirements 9.1, 9.2, 9.3, 9.5**
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <array>
#include <cstddef>
#include <span/field_view.hpp>
#include <stdexcept>
#include <vector>

namespace {

// ─── Kokkos Initialization ───────────────────────────────────────────────────

class KokkosEnvironment : public ::testing::Environment {
   public:
    void SetUp() override {
        if (!Kokkos::is_initialized()) {
            Kokkos::initialize();
        }
    }
    void TearDown() override {
        if (Kokkos::is_initialized()) {
            Kokkos::finalize();
        }
    }
};

static auto *const kokkos_env = ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);

}  // namespace

// ─── Rank 1 Pointer Identity Round-Trip ──────────────────────────────────────
// Validates Requirement 9.1:
//   THE FieldView SHALL provide a to_kokkos_view() method that returns a
//   Kokkos::View wrapping the same data pointer and extents.
// Validates Requirement 9.2:
//   THE FieldView SHALL provide a static from_kokkos_view() factory method that
//   constructs a FieldView from an unmanaged Kokkos::View without copying.
// Validates Requirement 9.3:
//   WHEN to_kokkos_view() is called and then from_kokkos_view() is called on
//   the result, THE round-trip SHALL produce a FieldView whose host_data()
//   pointer is identical to the original.

RC_GTEST_PROP(KokkosAdapterRoundTrip, Rank1PointerIdentity, ()) {
    const auto n = *rc::gen::inRange<std::size_t>(1, 1001);
    std::vector<double> buf(n);
    std::array<std::size_t, 1> exts{n};
    span::FieldView<double, 1> fv(buf.data(), exts);

    // to_kokkos_view() must wrap the same pointer and size
    auto kv = fv.to_kokkos_view();
    RC_ASSERT(kv.data() == buf.data());
    RC_ASSERT(kv.extent(0) == n);

    // from_kokkos_view() must reconstruct a FieldView with identical pointer
    auto fv2 = span::FieldView<double, 1>::from_kokkos_view(kv);
    RC_ASSERT(fv2.host_data() == buf.data());
    RC_ASSERT(fv2.size() == n);

    // Round-trip preserves coherency state at HOST_CLEAN (no modification)
    RC_ASSERT(fv2.coherency_state() == span::CoherencyState::HOST_CLEAN);

    // Original FieldView coherency state is unchanged
    RC_ASSERT(fv.coherency_state() == span::CoherencyState::HOST_CLEAN);
}

// ─── Coherency State Unchanged After Adapters ────────────────────────────────
// Validates Requirement 9.5:
//   THE adapters SHALL NOT invoke Kokkos::deep_copy, allocate device or host
//   memory, or modify the FieldView's coherency state.

RC_GTEST_PROP(KokkosAdapterRoundTrip, CoherencyStateUnchanged, ()) {
    const auto n = *rc::gen::inRange<std::size_t>(1, 1001);
    std::vector<double> buf(n);
    std::array<std::size_t, 1> exts{n};
    span::FieldView<double, 1> fv(buf.data(), exts);

    // Mark the FieldView dirty to test that adapters do NOT reset state
    fv.mark_host_dirty();
    RC_ASSERT(fv.coherency_state() == span::CoherencyState::HOST_DIRTY);

    // to_kokkos_view() must not alter coherency state
    auto kv = fv.to_kokkos_view();
    RC_ASSERT(fv.coherency_state() == span::CoherencyState::HOST_DIRTY);

    // from_kokkos_view() produces a new FieldView with fresh HOST_CLEAN state
    // (the original's state remains untouched)
    auto fv2 = span::FieldView<double, 1>::from_kokkos_view(kv);
    RC_ASSERT(fv2.coherency_state() == span::CoherencyState::HOST_CLEAN);

    // Original still HOST_DIRTY — adapters did not modify it
    RC_ASSERT(fv.coherency_state() == span::CoherencyState::HOST_DIRTY);
}

// ─── to_kokkos_view() Throws on Invalid FieldView ────────────────────────────
// Validates Requirement 9.6:
//   IF to_kokkos_view() is called on a FieldView whose valid() method returns
//   false (null pointer), THEN THE method SHALL throw std::invalid_argument.

RC_GTEST_PROP(KokkosAdapterRoundTrip, ThrowsOnInvalidFieldView, ()) {
    // Default-constructed FieldView is invalid
    span::FieldView<double, 1> fv;
    RC_ASSERT(!fv.valid());

    bool threw = false;
    try {
        [[maybe_unused]] auto kv = fv.to_kokkos_view();
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    RC_ASSERT(threw);
}

// ─── from_kokkos_view() Throws on Null Data Kokkos View ─────────────────────
// Validates Requirement 9.7:
//   IF from_kokkos_view() is called with a Kokkos::View whose data() pointer
//   is null, THEN THE factory method SHALL throw std::invalid_argument.

RC_GTEST_PROP(KokkosAdapterRoundTrip, FromKokkosViewThrowsOnNull, ()) {
    // Construct a Kokkos::View with null data pointer
    using kokkos_view_t = Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
    kokkos_view_t null_view(nullptr, 0);

    bool threw = false;
    try {
        [[maybe_unused]] auto fv = span::FieldView<double, 1>::from_kokkos_view(null_view);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    RC_ASSERT(threw);
}
