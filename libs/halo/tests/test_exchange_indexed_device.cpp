// ─── Indexed Halo Exchange Device-View Round-Trip Test (single-process MPI) ──
// Feature: cpp-dycore-halo-exchange
// Task 4.2: device-view round-trip integration test
//
// Exercises halo::exchange_indexed against a field view that lives in the
// default execution space's memory space (Kokkos::DefaultExecutionSpace::
// memory_space). This is the same view configuration the dycore uses for its
// prognostic/tendency fields, and it drives the memory-space dispatch inside
// exchange_indexed:
//
//   * When the view is device-resident and GPU-aware MPI is NOT available
//     (the compile-time detail::requires_staging_v path), the exchange stages
//     gathered data through host mirrors, communicates on the host, and copies
//     the scattered results back to device memory (Requirement 3.3).
//   * When the view is device-resident and a GPU-aware MPI environment IS
//     available, the exchange communicates the device-resident buffers directly
//     without a host copy (Requirement 3.4). That assertion is guarded and
//     skipped gracefully when Environment::is_gpu_aware_mpi() is false (e.g. on
//     a host-only build).
//   * When the view is host-resident (this container is host-only, so the
//     default execution space is typically Serial/OpenMP and its memory space
//     is HostSpace), the exchange communicates directly from host buffers and
//     the gather/scatter kernels run in the view's native execution space
//     (Requirements 4.1, 3.5). The staging code path is still meaningfully
//     exercised: on a device build the SAME test body drives the host-staged
//     branch, so this single test validates the round-trip for whichever path
//     the build selects.
//
// Regardless of which communication path the build selects, the round-trip
// invariant is identical: after the exchange each receive (halo) index holds
// exactly the source-owned value that resided at its paired send index; for
// rank-2 fields every vertical level is transferred. Owned elements and halo
// elements that are not receive targets are left unchanged.
//
// Runs single-process with real MPI using an in-process paired self-send index
// model (the running rank is its own sole send/recv neighbor), matching the
// design's in-process paired send/recv index model used by
// prop_exchange_indexed_roundtrip.cpp, so no mpirun launch is required.
//
//   Requirements: 3.3, 3.4, 4.1
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <mpi.h>

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <span>
#include <vector>

#include "halo/communicator.hpp"
#include "halo/detail/memory_traits.hpp"
#include "halo/environment.hpp"
#include "halo/exchange_indexed.hpp"
#include "halo/indexed_halo_plan.hpp"

namespace {

// ─── MPI rank global ─────────────────────────────────────────────────────────
int g_rank = 0;

/// Distinct value used to pre-fill halo (receive) slots so that halo elements
/// that are not receive targets can be confirmed unchanged after the exchange.
constexpr double kSentinel = -777.0;

/// Distinct owned value for a given local owned index on this rank. Distinct
/// across owned indices so a delivered halo value uniquely identifies its
/// source owned index.
double owned_value(std::size_t owned_index) { return 1000.0 * static_cast<double>(g_rank + 1) + static_cast<double>(owned_index); }

/// Per-(index, level) owned value for rank-2 fields; distinct per (index, level).
double owned_value_r2(std::size_t owned_index, std::size_t level) { return owned_value(owned_index) * 10.0 + static_cast<double>(level); }

// ─── Fixed synthetic paired send/recv topology ──────────────────────────────
// Owned indices occupy [0, kOwned); halo indices occupy [kOwned, kElements).
// The k-th send index is paired with the k-th receive index. Receive indices
// are distinct so every halo cell is written exactly once. Halo slot 9
// (kElements - 1) is deliberately NOT a receive target so the "unchanged"
// invariant on excluded halo slots is testable.
constexpr std::size_t kOwned = 6;
constexpr std::size_t kHalo = 4;
constexpr std::size_t kElements = kOwned + kHalo;  // 10

/// Owned indices to gather (position-paired with recv_indices()).
std::vector<std::size_t> send_indices() { return {0, 2, 5}; }
/// Distinct halo indices to fill (position-paired). Slot 9 stays at the sentinel.
std::vector<std::size_t> recv_indices() { return {6, 7, 8}; }

/// Build a single-layer self-send plan for the fixed paired indices. The whole
/// exchange happens on halo layer 0, and the exchange selects that single layer.
halo::Indexed_Halo_Plan make_self_plan(const halo::Communicator &comm) {
    halo::Indexed_Neighbor send_nbr{g_rank, {send_indices()}};
    halo::Indexed_Neighbor recv_nbr{g_rank, {recv_indices()}};
    return halo::Indexed_Halo_Plan(comm, halo::Element_Kind::cell, {send_nbr}, {recv_nbr});
}

/// Report (once) which communication path this build/runtime exercises so the
/// test log records whether the host-staged branch or the direct branch ran.
template <typename ViewType>
void record_path() {
    constexpr bool ct_requires_staging = halo::detail::requires_staging_v<ViewType>;
    const bool gpu_aware = halo::Environment::is_gpu_aware_mpi();
    const bool staged = ct_requires_staging && !gpu_aware;

    ::testing::Test::RecordProperty("view_requires_staging", ct_requires_staging ? "true" : "false");
    ::testing::Test::RecordProperty("gpu_aware_mpi", gpu_aware ? "true" : "false");
    ::testing::Test::RecordProperty("path", staged ? "host-staged" : (ct_requires_staging ? "gpu-aware-direct" : "host-direct"));
}

}  // namespace

// ─── Rank-1 device-view round-trip ───────────────────────────────────────────
// Fills halo indices from the paired source-owned values through whichever
// communication path the build selects (host-staged on a device build without
// GPU-aware MPI; host-direct on this host-only container).
//
// Requirements: 3.3, 3.4, 4.1

TEST(IndexedExchangeDeviceView, Rank1RoundTripFillsHalo) {
    using MemSpace = Kokkos::DefaultExecutionSpace::memory_space;
    using ViewR1 = Kokkos::View<double *, MemSpace>;

    record_path<ViewR1>();

    halo::Communicator comm(MPI_COMM_WORLD);
    halo::Indexed_Halo_Plan plan = make_self_plan(comm);

    const std::vector<std::size_t> send_idx = send_indices();
    const std::vector<std::size_t> recv_idx = recv_indices();
    const int layer0[] = {0};

    // Allocate the field in the default execution space's memory space and seed
    // it through a host mirror (works whether MemSpace is host or device).
    ViewR1 field("field_dev_r1", kElements);
    auto host = Kokkos::create_mirror_view(field);
    for (std::size_t i = 0; i < kElements; ++i) {
        host(i) = kSentinel;
    }
    for (std::size_t si = 0; si < kOwned; ++si) {
        host(si) = owned_value(si);
    }
    Kokkos::deep_copy(field, host);

    halo::exchange_indexed(plan, field, std::span<const int>(layer0, 1));

    // Copy results back to host for verification.
    auto result = Kokkos::create_mirror_view(field);
    Kokkos::deep_copy(result, field);

    // Each receive index holds the paired send index's source-owned value.
    for (std::size_t k = 0; k < recv_idx.size(); ++k) {
        EXPECT_DOUBLE_EQ(result(recv_idx[k]), owned_value(send_idx[k])) << "halo slot " << recv_idx[k] << " (pair " << k << ")";
    }

    // Owned elements are never written by the exchange.
    for (std::size_t si = 0; si < kOwned; ++si) {
        EXPECT_DOUBLE_EQ(result(si), owned_value(si)) << "owned cell " << si << " modified";
    }

    // Halo cells that were not receive targets keep the sentinel.
    std::vector<char> is_target(kElements, 0);
    for (std::size_t idx : recv_idx) {
        is_target[idx] = 1;
    }
    for (std::size_t i = kOwned; i < kElements; ++i) {
        if (!is_target[i]) {
            EXPECT_DOUBLE_EQ(result(i), kSentinel) << "non-target halo slot " << i << " modified";
        }
    }
}

// ─── Rank-2 device-view round-trip (nVertLevels x nElements, LayoutLeft) ──────
// Every vertical level of each receive column is transferred from the paired
// send column, through whichever communication path the build selects.
//
// Requirements: 3.3, 3.4, 4.1

TEST(IndexedExchangeDeviceView, Rank2RoundTripTransfersAllLevels) {
    using MemSpace = Kokkos::DefaultExecutionSpace::memory_space;
    using ViewR2 = Kokkos::View<double **, Kokkos::LayoutLeft, MemSpace>;

    record_path<ViewR2>();

    constexpr std::size_t nlev = 5;  // nVertLevels

    halo::Communicator comm(MPI_COMM_WORLD);
    halo::Indexed_Halo_Plan plan = make_self_plan(comm);

    const std::vector<std::size_t> send_idx = send_indices();
    const std::vector<std::size_t> recv_idx = recv_indices();
    const int layer0[] = {0};

    // Rank-2 view: (nVertLevels x nElements) in LayoutLeft, matching the dycore.
    ViewR2 field("field_dev_r2", nlev, kElements);
    auto host = Kokkos::create_mirror_view(field);
    for (std::size_t j = 0; j < kElements; ++j) {
        for (std::size_t lev = 0; lev < nlev; ++lev) {
            host(lev, j) = kSentinel;
        }
    }
    for (std::size_t si = 0; si < kOwned; ++si) {
        for (std::size_t lev = 0; lev < nlev; ++lev) {
            host(lev, si) = owned_value_r2(si, lev);
        }
    }
    Kokkos::deep_copy(field, host);

    halo::exchange_indexed(plan, field, std::span<const int>(layer0, 1));

    auto result = Kokkos::create_mirror_view(field);
    Kokkos::deep_copy(result, field);

    // Each receive column holds the paired send column's values, all levels.
    for (std::size_t k = 0; k < recv_idx.size(); ++k) {
        const std::size_t src = send_idx[k];
        const std::size_t dst = recv_idx[k];
        for (std::size_t lev = 0; lev < nlev; ++lev) {
            EXPECT_DOUBLE_EQ(result(lev, dst), owned_value_r2(src, lev)) << "halo col " << dst << " lev " << lev << " (pair " << k << ")";
        }
    }

    // Owned columns are never written by the exchange.
    for (std::size_t si = 0; si < kOwned; ++si) {
        for (std::size_t lev = 0; lev < nlev; ++lev) {
            EXPECT_DOUBLE_EQ(result(lev, si), owned_value_r2(si, lev)) << "owned col " << si << " lev " << lev << " modified";
        }
    }

    // Halo columns that were not receive targets keep the sentinel at all levels.
    std::vector<char> is_target(kElements, 0);
    for (std::size_t idx : recv_idx) {
        is_target[idx] = 1;
    }
    for (std::size_t j = kOwned; j < kElements; ++j) {
        if (!is_target[j]) {
            for (std::size_t lev = 0; lev < nlev; ++lev) {
                EXPECT_DOUBLE_EQ(result(lev, j), kSentinel) << "non-target halo col " << j << " lev " << lev << " modified";
            }
        }
    }
}

// ─── GPU-aware direct path (Requirement 3.4) ─────────────────────────────────
// The direct device-to-device communication path is only meaningful when the
// field view is device-resident AND a GPU-aware MPI environment is available.
// On a host-only build (this container) or when the runtime reports no GPU-aware
// MPI, skip gracefully. When it IS available, re-run the round-trip and assert
// no host staging occurred (the compile-time trait must have disabled staging).
//
// Requirements: 3.4

TEST(IndexedExchangeDeviceView, GpuAwareDirectPathWhenAvailable) {
    using MemSpace = Kokkos::DefaultExecutionSpace::memory_space;
    using ViewR1 = Kokkos::View<double *, MemSpace>;

    constexpr bool ct_requires_staging = halo::detail::requires_staging_v<ViewR1>;

    if (!halo::Environment::is_gpu_aware_mpi()) {
        GTEST_SKIP() << "GPU-aware MPI not available (host-only build or runtime probe false); "
                        "direct device-to-device path not exercised.";
    }
    if (!halo::detail::is_device_space_v<MemSpace>) {
        GTEST_SKIP() << "Default execution space is host-resident; direct device path not applicable.";
    }

    // On a GPU-aware build the compile-time staging trait must be disabled so the
    // exchange passes device pointers to MPI directly (no host copy).
    EXPECT_FALSE(ct_requires_staging) << "GPU-aware MPI build must not require host staging for device views";

    record_path<ViewR1>();

    halo::Communicator comm(MPI_COMM_WORLD);
    halo::Indexed_Halo_Plan plan = make_self_plan(comm);

    const std::vector<std::size_t> send_idx = send_indices();
    const std::vector<std::size_t> recv_idx = recv_indices();
    const int layer0[] = {0};

    ViewR1 field("field_dev_gpu_r1", kElements);
    auto host = Kokkos::create_mirror_view(field);
    for (std::size_t i = 0; i < kElements; ++i) {
        host(i) = kSentinel;
    }
    for (std::size_t si = 0; si < kOwned; ++si) {
        host(si) = owned_value(si);
    }
    Kokkos::deep_copy(field, host);

    halo::exchange_indexed(plan, field, std::span<const int>(layer0, 1));

    auto result = Kokkos::create_mirror_view(field);
    Kokkos::deep_copy(result, field);

    for (std::size_t k = 0; k < recv_idx.size(); ++k) {
        EXPECT_DOUBLE_EQ(result(recv_idx[k]), owned_value(send_idx[k])) << "halo slot " << recv_idx[k] << " (pair " << k << ")";
    }
}

// ─── Global MPI + Kokkos + HALO environment ─────────────────────────────────

namespace {

class HaloMpiEnvironment : public ::testing::Environment {
   public:
    void SetUp() override {
        int provided = 0;
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
        Kokkos::initialize();
        halo::Environment::initialize();

        MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    }

    void TearDown() override {
        Kokkos::finalize();
        MPI_Finalize();
    }
};

}  // namespace

// Register the environment (gtest_main provides main()).
static ::testing::Environment *const halo_mpi_env = ::testing::AddGlobalTestEnvironment(new HaloMpiEnvironment);
