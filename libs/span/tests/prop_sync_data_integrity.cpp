// ─── Property-Based Tests: Sync Data Integrity ─────────────────────────────
// Feature: span-field-view, Property 5: Sync Data Integrity
//
// For any FieldView with both host and device pointers and for any array of
// values written to the host buffer, calling mark_host_dirty() then
// sync_for_device() SHALL result in the device buffer containing element-wise
// identical values. Symmetrically, for any values written to the device buffer,
// calling mark_device_dirty() then sync_for_host() SHALL result in the host
// buffer containing identical values.
//
// **Validates: Requirements 4.7, 4.8, 3.6, 3.8**
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <span/field_view.hpp>
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

// ─── Host-to-Device Sync ─────────────────────────────────────────────────────
// Validates Requirement 4.7:
//   WHEN sync_for_device() is called and the coherency state is HOST_DIRTY,
//   THE FieldView SHALL copy total_size elements from the host pointer to the
//   device pointer using Kokkos::deep_copy.
// Validates Requirement 3.6:
//   WHEN sync_for_device() is called and the CoherencyState is HOST_DIRTY,
//   THE FieldView SHALL execute a Kokkos::deep_copy from host to device and
//   transition the state to HOST_CLEAN.

RC_GTEST_PROP(SyncDataIntegrity, HostToDevice, ()) {
    const auto n = *rc::gen::inRange<std::size_t>(1, 1001);
    std::vector<double> host_buf(n);
    std::vector<double> dev_buf(n, 0.0);  // initially zeros

    // Fill host with random values
    for (auto &v : host_buf) {
        v = *rc::gen::arbitrary<double>();
    }

    std::array<std::size_t, 1> exts{n};
    span::FieldView<double, 1> fv(host_buf.data(), dev_buf.data(), exts);

    fv.mark_host_dirty();
    fv.sync_for_device();

    // Verify device buffer now matches host element-wise
    for (std::size_t i = 0; i < n; ++i) {
        RC_ASSERT(dev_buf[i] == host_buf[i]);
    }
    // State must transition to HOST_CLEAN after successful sync
    RC_ASSERT(fv.coherency_state() == span::CoherencyState::HOST_CLEAN);
}

// ─── Device-to-Host Sync ─────────────────────────────────────────────────────
// Validates Requirement 4.8:
//   WHEN sync_for_host() is called and the coherency state is DEVICE_DIRTY,
//   THE FieldView SHALL copy total_size elements from the device pointer to the
//   host pointer using Kokkos::deep_copy.
// Validates Requirement 3.8:
//   WHEN sync_for_host() is called and the CoherencyState is DEVICE_DIRTY,
//   THE FieldView SHALL execute a Kokkos::deep_copy from device to host and
//   transition the state to HOST_CLEAN.

RC_GTEST_PROP(SyncDataIntegrity, DeviceToHost, ()) {
    const auto n = *rc::gen::inRange<std::size_t>(1, 1001);
    std::vector<double> host_buf(n, 0.0);  // initially zeros
    std::vector<double> dev_buf(n);

    // Fill device with random values
    for (auto &v : dev_buf) {
        v = *rc::gen::arbitrary<double>();
    }

    std::array<std::size_t, 1> exts{n};
    span::FieldView<double, 1> fv(host_buf.data(), dev_buf.data(), exts);

    fv.mark_device_dirty();
    fv.sync_for_host();

    // Verify host buffer now matches device element-wise
    for (std::size_t i = 0; i < n; ++i) {
        RC_ASSERT(host_buf[i] == dev_buf[i]);
    }
    // State must transition to HOST_CLEAN after successful sync
    RC_ASSERT(fv.coherency_state() == span::CoherencyState::HOST_CLEAN);
}
