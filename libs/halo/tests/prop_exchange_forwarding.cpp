// --- Property-Based Tests: C-Interop Exchange Forwarding ---------------------
// Feature: helm-halo-microlibrary, Property 25: Exchange Forwarding Preserves Pointer and Size
//
// Uses RapidCheck to verify the extern "C" interop layer
// (src/fortran/halo_c_interop.cpp) forwards a raw Fortran array pointer to the
// HALO exchange functions WITHOUT copying or relocating the data:
//
//   halo_exchange_blocking_c(plan, data, num_elements, element_size) and
//   halo_exchange_async_c(plan, data, num_elements, element_size, handle_out)
//   each construct a non-owning Kokkos::View<char*, HostSpace, Unmanaged> over
//   the raw `data` pointer spanning exactly num_elements * element_size bytes,
//   then forward to halo::exchange_blocking / halo::exchange_async.
//
// The property verified here (observed through the MPI interposition spy, which
// records the buffer argument of every MPI_Irecv / MPI_Isend):
//
//   For any (data pointer, num_elements, element_size) whose byte length is
//   consistent with the plan's send/recv geometry, EVERY buffer pointer that
//   reaches MPI points INTO the provided data buffer, i.e.
//       base <= buf < base + num_elements * element_size,
//   and NO copy/relocation occurs (a HostSpace unmanaged view means MPI sees
//   the exact Fortran pointer). The send region begins at EXACTLY `data`
//   (the first MPI_Isend buffer == base, offset 0).
//
// The non-negotiable invariants verified here:
//   - Pointer containment: every Irecv/Isend buffer lies within the view bytes.
//   - Pointer identity / size preservation: the send region starts at offset 0
//     (== data), confirming the unmanaged char view spans the raw pointer with
//     no staging copy interposed.
//   - No C++ exception ever escapes across the language boundary.
//
// These run single-rank with the MPI interposition spy (MPI_Comm_size mock
// returns 4), so no mpirun is required.
//
// **Validates: Requirements 14.5, 14.6**
// -----------------------------------------------------------------------------

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include "halo/communicator.hpp"
#include "handle_registry.hpp"
#include "mpi_interposition.hpp"

// --- extern "C" interop forward declarations ---------------------------------
// halo_c_interop.cpp defines these without a public header, so we declare the
// signatures here (matching the definitions exactly) to call across the boundary.

extern "C" {
int halo_init_c(int mpi_comm_int, int* comm_handle_out);
int halo_comm_create_c(int parent_handle, int color, int key,
                       int* child_handle_out);
int halo_plan_create_c(int comm_handle,
                       const int* send_ranks, const int* send_counts, int num_send,
                       const int* recv_ranks, const int* recv_counts, int num_recv,
                       int* plan_handle_out);
int halo_exchange_blocking_c(int plan_handle, void* data,
                             int num_elements, int element_size);
int halo_exchange_async_c(int plan_handle, void* data,
                          int num_elements, int element_size,
                          int* handle_out);
int halo_wait_c(int handle);
int halo_test_c(int handle, int* complete_out);
int halo_destroy_plan_c(int plan_handle);
int halo_destroy_comm_c(int comm_handle);
}

namespace {

// --- Error code mirror -------------------------------------------------------
// The Halo_Error enum lives in an anonymous namespace inside halo_c_interop.cpp
// and is not exported, so we mirror the contract values here for assertions.
constexpr int HALO_SUCCESS         = 0;
constexpr int HALO_ERR_INVALID_ARG = 1;
constexpr int HALO_ERR_BAD_HANDLE  = 4;

/// The mock MPI_Comm_size always returns 4, so valid ranks are [0, 4).
constexpr int MOCK_COMM_SIZE = 4;

// --- Valid communicator handle (registered once, intentionally leaked) -------
// halo_plan_create_c needs a live Communicator referenced by an opaque handle.
// We heap-allocate a Communicator wrapping MPI_COMM_WORLD and register it in the
// process-global Handle_Registry. We never call halo_destroy_comm_c on it (which
// would `delete` the pointer), so the handle stays valid for every iteration and
// there is no double-free. The single leak is harmless for a test process.
int valid_comm_handle() {
    static int handle = [] {
        auto* comm = new halo::Communicator(MPI_COMM_WORLD);
        return halo::fortran::Handle_Registry::instance()
            .register_handle(static_cast<void*>(comm));
    }();
    return handle;
}

/// Generate a NON-EMPTY valid neighbor set: a random subset of unique ranks in
/// [0, MOCK_COMM_SIZE) with positive byte counts. Returns parallel (ranks,
/// counts). The byte counts double as element counts because the interop layer
/// builds a char-typed view (1 byte per element), so a neighbor's "count" is a
/// span of that many bytes inside the data buffer.
std::pair<std::vector<int>, std::vector<int>> gen_nonempty_neighbors() {
    int n = *rc::gen::inRange(1, MOCK_COMM_SIZE + 1);  // 1..4 neighbors
    std::vector<int> available{0, 1, 2, 3};
    std::vector<int> ranks;
    std::vector<int> counts;
    ranks.reserve(static_cast<std::size_t>(n));
    counts.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        int idx = *rc::gen::inRange(0, static_cast<int>(available.size()));
        ranks.push_back(available[static_cast<std::size_t>(idx)]);
        available.erase(available.begin() + idx);
        counts.push_back(*rc::gen::inRange(1, 65));  // 1..64 bytes per neighbor
    }
    return {ranks, counts};
}

/// Sum a count vector as bytes.
std::size_t sum_bytes(const std::vector<int>& counts) {
    std::size_t total = 0;
    for (int c : counts) total += static_cast<std::size_t>(c);
    return total;
}

}  // anonymous namespace

// --- Property 25 (a): Blocking forwarding preserves pointer and size ---------
// Feature: helm-halo-microlibrary, Property 25: Exchange Forwarding Preserves Pointer and Size
//
// For any data buffer + (num_elements, element_size) whose total byte length is
// large enough for the plan's send + recv regions, halo_exchange_blocking_c
// SHALL forward the raw pointer such that EVERY MPI_Irecv / MPI_Isend buffer
// recorded by the spy lies within [base, base + num_elements * element_size),
// and the first send buffer begins at EXACTLY `data` (offset 0). The call SHALL
// return HALO_SUCCESS and SHALL NOT throw across the boundary.
//
// **Validates: Requirements 14.5, 14.6**

RC_GTEST_PROP(InteropProperty25, BlockingForwardingPreservesPointerAndSize, ()) {
    auto& spy = halo::testing::MPI_Spy::instance();
    spy.reset();

    const int comm = valid_comm_handle();

    auto [send_ranks, send_counts] = gen_nonempty_neighbors();
    auto [recv_ranks, recv_counts] = gen_nonempty_neighbors();

    // The exchange reads a contiguous send region [0, total_send) followed by a
    // recv region [total_send, total_send + total_recv) -- all measured in bytes
    // because the interop view is char-typed. The data buffer must be at least
    // this many bytes for every MPI buffer pointer to fall inside it.
    const std::size_t total_send = sum_bytes(send_counts);
    const std::size_t total_recv = sum_bytes(recv_counts);
    const std::size_t needed_bytes = total_send + total_recv;

    // Choose (element_size, num_elements) so num_elements*element_size covers
    // the needed bytes, with a random surplus to exercise oversized buffers.
    const int element_size = *rc::gen::inRange(1, 9);  // 1..8 bytes per element
    const std::size_t min_elems =
        (needed_bytes + static_cast<std::size_t>(element_size) - 1) /
        static_cast<std::size_t>(element_size);
    const std::size_t extra_elems =
        static_cast<std::size_t>(*rc::gen::inRange(0, 17));
    const int num_elements = static_cast<int>(min_elems + extra_elems);

    const std::size_t total_bytes =
        static_cast<std::size_t>(num_elements) *
        static_cast<std::size_t>(element_size);

    // The real host buffer the Fortran pointer would reference.
    std::vector<char> buffer(total_bytes);
    void* base = static_cast<void*>(buffer.data());

    // Create the plan through the interop layer (returns an int plan handle).
    int plan_handle = -1;
    int create_code = halo_plan_create_c(
        comm,
        send_ranks.data(), send_counts.data(),
        static_cast<int>(send_ranks.size()),
        recv_ranks.data(), recv_counts.data(),
        static_cast<int>(recv_ranks.size()),
        &plan_handle);
    RC_ASSERT(create_code == HALO_SUCCESS);

    // Only the exchange's MPI traffic should be observed.
    spy.reset();

    // --- Exercise the forwarding under test ---
    bool threw = false;
    int exchange_code = HALO_ERR_INVALID_ARG;
    try {
        exchange_code =
            halo_exchange_blocking_c(plan_handle, base, num_elements, element_size);
    } catch (...) {
        threw = true;
    }

    // Collect the buffer pointers that reached MPI, preserving call order.
    auto calls = spy.calls_copy();
    std::vector<void*> irecv_bufs;
    std::vector<void*> isend_bufs;
    for (auto const& rec : calls) {
        if (rec.type == halo::testing::MPI_Call_Record::Type::Irecv) {
            irecv_bufs.push_back(rec.handle);
        } else if (rec.type == halo::testing::MPI_Call_Record::Type::Isend) {
            isend_bufs.push_back(rec.handle);
        }
    }

    const std::uintptr_t base_addr = reinterpret_cast<std::uintptr_t>(base);
    const std::uintptr_t end_addr  = base_addr + total_bytes;

    // Compute containment as a plain boolean so RC_ASSERT compares values.
    bool all_within = true;
    for (void* b : irecv_bufs) {
        const std::uintptr_t a = reinterpret_cast<std::uintptr_t>(b);
        if (!(a >= base_addr && a < end_addr)) all_within = false;
    }
    for (void* b : isend_bufs) {
        const std::uintptr_t a = reinterpret_cast<std::uintptr_t>(b);
        if (!(a >= base_addr && a < end_addr)) all_within = false;
    }

    const std::uintptr_t first_send_addr =
        isend_bufs.empty() ? 0u : reinterpret_cast<std::uintptr_t>(isend_bufs.front());

    // (1) No C++ exception escaped the boundary; forwarding succeeded.
    RC_ASSERT(!threw);
    RC_ASSERT(exchange_code == HALO_SUCCESS);

    // (2) The exchange posted one Irecv per recv-neighbor and one Isend per
    //     send-neighbor (no fan-out, no relocation into extra staging buffers).
    RC_ASSERT(irecv_bufs.size() == recv_ranks.size());
    RC_ASSERT(isend_bufs.size() == send_ranks.size());

    // (3) Every MPI buffer points INTO the provided data buffer.
    RC_ASSERT(all_within);

    // (4) Pointer identity: the send region begins at EXACTLY `data` (offset 0),
    //     proving the unmanaged char view forwards the raw pointer unchanged.
    RC_ASSERT(!isend_bufs.empty());
    RC_ASSERT(first_send_addr == base_addr);

    // Clean up the plan handle to avoid leaks.
    int destroy_code = halo_destroy_plan_c(plan_handle);
    RC_ASSERT(destroy_code == HALO_SUCCESS);
}

// --- Property 25 (b): Async forwarding preserves pointer and size ------------
// Feature: helm-halo-microlibrary, Property 25: Exchange Forwarding Preserves Pointer and Size
//
// Identical contract for halo_exchange_async_c: every MPI_Irecv / MPI_Isend
// buffer lies within [base, base + num_elements * element_size), the send
// region begins at EXACTLY `data` (offset 0), the call returns HALO_SUCCESS and
// a usable handle, and nothing throws across the boundary. The async operation
// is then completed via halo_wait_c so no requests dangle.
//
// **Validates: Requirements 14.5, 14.6**

RC_GTEST_PROP(InteropProperty25, AsyncForwardingPreservesPointerAndSize, ()) {
    auto& spy = halo::testing::MPI_Spy::instance();
    spy.reset();

    const int comm = valid_comm_handle();

    auto [send_ranks, send_counts] = gen_nonempty_neighbors();
    auto [recv_ranks, recv_counts] = gen_nonempty_neighbors();

    const std::size_t total_send = sum_bytes(send_counts);
    const std::size_t total_recv = sum_bytes(recv_counts);
    const std::size_t needed_bytes = total_send + total_recv;

    const int element_size = *rc::gen::inRange(1, 9);  // 1..8 bytes per element
    const std::size_t min_elems =
        (needed_bytes + static_cast<std::size_t>(element_size) - 1) /
        static_cast<std::size_t>(element_size);
    const std::size_t extra_elems =
        static_cast<std::size_t>(*rc::gen::inRange(0, 17));
    const int num_elements = static_cast<int>(min_elems + extra_elems);

    const std::size_t total_bytes =
        static_cast<std::size_t>(num_elements) *
        static_cast<std::size_t>(element_size);

    std::vector<char> buffer(total_bytes);
    void* base = static_cast<void*>(buffer.data());

    int plan_handle = -1;
    int create_code = halo_plan_create_c(
        comm,
        send_ranks.data(), send_counts.data(),
        static_cast<int>(send_ranks.size()),
        recv_ranks.data(), recv_counts.data(),
        static_cast<int>(recv_ranks.size()),
        &plan_handle);
    RC_ASSERT(create_code == HALO_SUCCESS);

    spy.reset();

    // --- Exercise the async forwarding under test ---
    bool threw = false;
    int exchange_code = HALO_ERR_INVALID_ARG;
    int async_handle = -1;
    try {
        exchange_code = halo_exchange_async_c(
            plan_handle, base, num_elements, element_size, &async_handle);
    } catch (...) {
        threw = true;
    }

    auto calls = spy.calls_copy();
    std::vector<void*> irecv_bufs;
    std::vector<void*> isend_bufs;
    for (auto const& rec : calls) {
        if (rec.type == halo::testing::MPI_Call_Record::Type::Irecv) {
            irecv_bufs.push_back(rec.handle);
        } else if (rec.type == halo::testing::MPI_Call_Record::Type::Isend) {
            isend_bufs.push_back(rec.handle);
        }
    }

    const std::uintptr_t base_addr = reinterpret_cast<std::uintptr_t>(base);
    const std::uintptr_t end_addr  = base_addr + total_bytes;

    bool all_within = true;
    for (void* b : irecv_bufs) {
        const std::uintptr_t a = reinterpret_cast<std::uintptr_t>(b);
        if (!(a >= base_addr && a < end_addr)) all_within = false;
    }
    for (void* b : isend_bufs) {
        const std::uintptr_t a = reinterpret_cast<std::uintptr_t>(b);
        if (!(a >= base_addr && a < end_addr)) all_within = false;
    }

    const std::uintptr_t first_send_addr =
        isend_bufs.empty() ? 0u : reinterpret_cast<std::uintptr_t>(isend_bufs.front());

    const bool handle_is_usable = (async_handle > 0);

    // (1) No C++ exception escaped the boundary; forwarding succeeded with a
    //     usable async handle.
    RC_ASSERT(!threw);
    RC_ASSERT(exchange_code == HALO_SUCCESS);
    RC_ASSERT(handle_is_usable);

    // (2) One Irecv per recv-neighbor, one Isend per send-neighbor.
    RC_ASSERT(irecv_bufs.size() == recv_ranks.size());
    RC_ASSERT(isend_bufs.size() == send_ranks.size());

    // (3) Every MPI buffer points INTO the provided data buffer.
    RC_ASSERT(all_within);

    // (4) Pointer identity: send region begins at EXACTLY `data` (offset 0).
    RC_ASSERT(!isend_bufs.empty());
    RC_ASSERT(first_send_addr == base_addr);

    // Drive the async operation to completion so no MPI requests dangle, then
    // destroy the plan handle. (The interop layer exposes no destroy for the
    // Halo_Handle itself; the heap object is intentionally left -- a harmless
    // test-process leak, consistent with valid_comm_handle() above.)
    int wait_code = halo_wait_c(async_handle);
    RC_ASSERT(wait_code == HALO_SUCCESS);

    int destroy_code = halo_destroy_plan_c(plan_handle);
    RC_ASSERT(destroy_code == HALO_SUCCESS);
}

// --- Kokkos Initialization ---------------------------------------------------
// The interop translation unit references Kokkos; initialize it once for the
// whole test binary so any code path that touches a Kokkos::View is safe.

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

static auto* const kokkos_env =
    ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
