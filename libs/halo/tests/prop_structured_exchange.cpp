// ─── Property-Based Tests: Structured Halo Exchange ─────────────────────────
// Feature: halo-production-hardening
//
// Uses RapidCheck with real MPI (mpirun -np 4) to verify structured exchange
// properties with randomized inputs.
//
// Property 1: Pack then unpack is identity (for any rank-2 subview shape)
//   Validates: Requirements 2.1
//
// Property 2: Structured exchange round-trip preserves halo data
//   (random grid sizes, random halo widths)
//   Validates: Requirements 1.1
//
// Property 3: Persistent start/wait delivers same data as non-persistent exchange
//   Validates: Requirements 7.1
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <mpi.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

#include "halo/communicator.hpp"
#include "halo/detail/pack_unpack.hpp"
#include "halo/environment.hpp"
#include "halo/exchange.hpp"
#include "halo/exchange_structured.hpp"
#include "halo/halo_plan.hpp"
#include "halo/persistent_halo_handle.hpp"
#include "halo/structured_halo_plan.hpp"

namespace {

// ─── Globals for MPI rank / size ────────────────────────────────────────────
int g_rank = 0;
int g_size = 0;

// ─── Property 1: Pack then Unpack is Identity ───────────────────────────────
// Feature: halo-production-hardening
// Property 1: Pack then unpack is identity (for any subview shape)
//
// For any 2D view with random extents (n0 in [2,32], n1 in [2,32]), filling
// with arbitrary double values, packing into a flat buffer and unpacking into
// a fresh view of the same shape yields element-wise equality.
//
// **Validates: Requirements 2.1**

RC_GTEST_PROP(StructuredExchangeProps, PackUnpackIsIdentity, ()) {
    // Generate random 2D view dimensions [2, 32]
    const auto n0 = static_cast<std::size_t>(*rc::gen::inRange(2, 33));
    const auto n1 = static_cast<std::size_t>(*rc::gen::inRange(2, 33));

    // Create source view and fill with deterministic pattern
    Kokkos::View<double **, Kokkos::LayoutRight, Kokkos::HostSpace> src("src", n0, n1);

    for (std::size_t i = 0; i < n0; ++i) {
        for (std::size_t j = 0; j < n1; ++j) {
            src(i, j) = static_cast<double>(i * n1 + j) + 0.5;
        }
    }

    // Pack into flat buffer
    const std::size_t total = n0 * n1;
    Kokkos::View<double *, Kokkos::HostSpace> buffer("buffer", total);
    halo::detail::pack(src, buffer);

    // Unpack into a fresh destination view
    Kokkos::View<double **, Kokkos::LayoutRight, Kokkos::HostSpace> dst("dst", n0, n1);
    halo::detail::unpack(buffer, dst);

    // Verify element-wise equality
    for (std::size_t i = 0; i < n0; ++i) {
        for (std::size_t j = 0; j < n1; ++j) {
            RC_ASSERT(dst(i, j) == src(i, j));
        }
    }
}

// ─── Property 1b: Pack/Unpack identity for LayoutLeft ───────────────────────
// Same as Property 1 but with Kokkos::LayoutLeft (column-major) views.
//
// **Validates: Requirements 2.1**

RC_GTEST_PROP(StructuredExchangeProps, PackUnpackIdentityLayoutLeft, ()) {
    const auto n0 = static_cast<std::size_t>(*rc::gen::inRange(2, 33));
    const auto n1 = static_cast<std::size_t>(*rc::gen::inRange(2, 33));

    Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> src("src_ll", n0, n1);

    for (std::size_t i = 0; i < n0; ++i) {
        for (std::size_t j = 0; j < n1; ++j) {
            src(i, j) = static_cast<double>(i * 100 + j) + 0.25;
        }
    }

    const std::size_t total = n0 * n1;
    Kokkos::View<double *, Kokkos::HostSpace> buffer("buffer_ll", total);
    halo::detail::pack(src, buffer);

    Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> dst("dst_ll", n0, n1);
    halo::detail::unpack(buffer, dst);

    for (std::size_t i = 0; i < n0; ++i) {
        for (std::size_t j = 0; j < n1; ++j) {
            RC_ASSERT(dst(i, j) == src(i, j));
        }
    }
}

// ─── Property 1c: Pack/Unpack identity for strided subview ──────────────────
// Extract a subview from a larger 2D view (strided), pack/unpack, verify.
//
// **Validates: Requirements 2.1**

RC_GTEST_PROP(StructuredExchangeProps, PackUnpackIdentityStrided, ()) {
    // Outer view dimensions [6, 32] to allow meaningful subviews
    const auto outer_n0 = static_cast<std::size_t>(*rc::gen::inRange(6, 33));
    const auto outer_n1 = static_cast<std::size_t>(*rc::gen::inRange(6, 33));

    // Subview start/end: ensure at least 2 elements per dimension
    const auto start0 = static_cast<std::size_t>(*rc::gen::inRange(0, static_cast<int>(outer_n0 - 2)));
    const auto end0 = static_cast<std::size_t>(*rc::gen::inRange(static_cast<int>(start0 + 2), static_cast<int>(outer_n0) + 1));
    const auto start1 = static_cast<std::size_t>(*rc::gen::inRange(0, static_cast<int>(outer_n1 - 2)));
    const auto end1 = static_cast<std::size_t>(*rc::gen::inRange(static_cast<int>(start1 + 2), static_cast<int>(outer_n1) + 1));

    Kokkos::View<double **, Kokkos::LayoutRight, Kokkos::HostSpace> outer("outer", outer_n0, outer_n1);

    // Fill the whole outer view
    for (std::size_t i = 0; i < outer_n0; ++i) {
        for (std::size_t j = 0; j < outer_n1; ++j) {
            outer(i, j) = static_cast<double>(i * outer_n1 + j);
        }
    }

    // Extract strided subview
    auto sub = Kokkos::subview(outer, Kokkos::make_pair(start0, end0), Kokkos::make_pair(start1, end1));

    const std::size_t sub_n0 = end0 - start0;
    const std::size_t sub_n1 = end1 - start1;
    const std::size_t total = sub_n0 * sub_n1;

    // Pack the subview
    Kokkos::View<double *, Kokkos::HostSpace> buffer("buf_strided", total);
    halo::detail::pack(sub, buffer);

    // Unpack into a fresh view of matching shape
    Kokkos::View<double **, Kokkos::LayoutRight, Kokkos::HostSpace> dst("dst_strided", sub_n0, sub_n1);
    halo::detail::unpack(buffer, dst);

    // Verify
    for (std::size_t i = 0; i < sub_n0; ++i) {
        for (std::size_t j = 0; j < sub_n1; ++j) {
            RC_ASSERT(dst(i, j) == sub(i, j));
        }
    }
}

// ─── Property 2: Structured Exchange Round-Trip Preserves Halo Data ─────────
// Feature: halo-production-hardening
// Property 2: Structured exchange round-trip preserves halo data
//
// On 4 ranks in a periodic ring along d0, for random grid interior size
// [4, 24] and random halo width [1, 4] (with interior >= 2*halo), fill
// interior with rank-encoded data, perform structured exchange, and verify
// that halo zones contain the correct neighbor's interior boundary data.
//
// **Validates: Requirements 1.1**

RC_GTEST_PROP(StructuredExchangeProps, RoundTripPreservesHaloData, ()) {
    RC_PRE(g_size == 4);

    // Generate random parameters on rank 0 and broadcast to all ranks
    // so every rank uses the same grid geometry.
    int params[3] = {0, 0, 0};  // halo_w, interior, interior1
    if (g_rank == 0) {
        params[0] = *rc::gen::inRange(1, 5);
        int interior_min = 2 * params[0];
        params[1] = *rc::gen::inRange(interior_min, 25);
        params[2] = *rc::gen::inRange(interior_min, 25);
    }
    MPI_Bcast(params, 3, MPI_INT, 0, MPI_COMM_WORLD);

    const int halo_w = params[0];
    const int interior = params[1];
    const int interior1_val = params[2];
    const auto halo = static_cast<std::size_t>(halo_w);
    const auto total_dim = static_cast<std::size_t>(interior) + 2 * halo;
    const auto total_dim1 = static_cast<std::size_t>(interior1_val) + 2 * halo;

    // Periodic ring topology along d0
    int west = (g_rank - 1 + g_size) % g_size;
    int east = (g_rank + 1) % g_size;
    std::array<int, 4> neighbors = {west, east, -1, -1};

    std::array<std::size_t, 2> extents = {total_dim, total_dim1};
    std::array<std::size_t, 2> halo_widths = {halo, halo};

    halo::Communicator comm(MPI_COMM_WORLD);
    halo::Structured_Halo_Plan<2> plan(extents, neighbors, halo_widths, comm);

    // Create and fill view
    Kokkos::View<double **, Kokkos::LayoutRight> view("prop_field", total_dim, total_dim1);
    auto h_view = Kokkos::create_mirror_view(view);

    const double sentinel = -999.0;
    Kokkos::deep_copy(h_view, sentinel);

    // Fill interior with rank-encoded data
    for (std::size_t i = halo; i < halo + static_cast<std::size_t>(interior); ++i) {
        for (std::size_t j = halo; j < halo + static_cast<std::size_t>(interior1_val); ++j) {
            std::size_t local_idx = (i - halo) * static_cast<std::size_t>(interior1_val) + (j - halo);
            h_view(i, j) = static_cast<double>(g_rank) * 1000000.0 + static_cast<double>(local_idx);
        }
    }
    Kokkos::deep_copy(view, h_view);

    // Exchange
    halo::exchange_structured_blocking(plan, view);

    // Verify halos
    auto h_result = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);

    // West halo: dim0 [0, halo), dim1 [halo, halo+interior1_val)
    // This was received from the west neighbor's send region: dim0 [halo, 2*halo)
    // West neighbor's data at (halo + offset, j) = west * 1e6 + offset * interior1_val + (j - halo)
    // But the send region for face high-d0 is [total_dim - 2*halo, total_dim - halo)
    // which corresponds to the west neighbor's interior boundary.
    for (std::size_t i = 0; i < halo; ++i) {
        for (std::size_t j = halo; j < halo + static_cast<std::size_t>(interior1_val); ++j) {
            // West neighbor sends from face 1 (high-d0): dim0 [total_dim-2*halo, total_dim-halo)
            // That's interior indices [interior - halo + i] in the west neighbor's grid
            std::size_t west_interior_row = static_cast<std::size_t>(interior) - halo + i;
            std::size_t west_local_idx = west_interior_row * static_cast<std::size_t>(interior1_val) + (j - halo);
            double expected = static_cast<double>(west) * 1000000.0 + static_cast<double>(west_local_idx);
            RC_ASSERT(h_result(i, j) == expected);
        }
    }

    // East halo: dim0 [total_dim - halo, total_dim), dim1 [halo, halo+interior1_val)
    // Received from east neighbor's face 0 (low-d0) send region: dim0 [halo, 2*halo)
    for (std::size_t i = 0; i < halo; ++i) {
        for (std::size_t j = halo; j < halo + static_cast<std::size_t>(interior1_val); ++j) {
            std::size_t east_interior_row = i;  // rows 0..halo-1 of east's interior
            std::size_t east_local_idx = east_interior_row * static_cast<std::size_t>(interior1_val) + (j - halo);
            double expected = static_cast<double>(east) * 1000000.0 + static_cast<double>(east_local_idx);
            RC_ASSERT(h_result(total_dim - halo + i, j) == expected);
        }
    }

    // South/North halos should remain sentinel (no neighbors)
    for (std::size_t i = halo; i < halo + static_cast<std::size_t>(interior); ++i) {
        for (std::size_t j = 0; j < halo; ++j) {
            RC_ASSERT(h_result(i, j) == sentinel);
        }
        for (std::size_t j = total_dim1 - halo; j < total_dim1; ++j) {
            RC_ASSERT(h_result(i, j) == sentinel);
        }
    }
}

// ─── Property 3: Persistent Start/Wait Delivers Same Data ───────────────────
// Feature: halo-production-hardening
// Property 3: Persistent start/wait delivers same data as non-persistent exchange
//
// On 4 ranks in a ring topology, for a random buffer size [4, 64], performing
// a non-persistent exchange_blocking and a persistent start/wait cycle on
// identically initialized buffers SHALL produce identical receive-region data.
//
// **Validates: Requirements 7.1**

RC_GTEST_PROP(StructuredExchangeProps, PersistentEqualsNonPersistent, ()) {
    RC_PRE(g_size == 4);

    // Generate random element count on rank 0 and broadcast to all ranks
    int count_int = 0;
    if (g_rank == 0) {
        count_int = *rc::gen::inRange(4, 65);
    }
    MPI_Bcast(&count_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    const auto count = static_cast<std::size_t>(count_int);

    // Ring topology: send to right, recv from left
    int send_rank = (g_rank + 1) % g_size;
    int recv_rank = (g_rank - 1 + g_size) % g_size;

    halo::Communicator comm(MPI_COMM_WORLD);

    std::vector<halo::Neighbor_Info> send_info{{send_rank, count}};
    std::vector<halo::Neighbor_Info> recv_info{{recv_rank, count}};
    halo::Halo_Plan plan(comm, send_info, recv_info);

    // Layout: [send_region(count) | recv_region(count)]
    const std::size_t total = 2 * count;

    // ─── Non-persistent exchange ────────────────────────────────────────────
    Kokkos::View<double *, Kokkos::HostSpace> field_np("np_field", total);
    // Fill send region with rank-encoded data
    for (std::size_t j = 0; j < count; ++j) {
        field_np(j) = static_cast<double>(g_rank) * 1000.0 + static_cast<double>(j);
    }
    // Clear recv region
    for (std::size_t j = count; j < total; ++j) {
        field_np(j) = -1.0;
    }

    halo::exchange_blocking(plan, field_np);

    // ─── Persistent exchange ────────────────────────────────────────────────
    Kokkos::View<double *, Kokkos::HostSpace> field_p("p_field", total);
    // Fill identically
    for (std::size_t j = 0; j < count; ++j) {
        field_p(j) = static_cast<double>(g_rank) * 1000.0 + static_cast<double>(j);
    }
    for (std::size_t j = count; j < total; ++j) {
        field_p(j) = -1.0;
    }

    halo::Persistent_Halo_Handle handle(plan, field_p);
    handle.start();
    handle.wait();

    // ─── Compare receive regions ────────────────────────────────────────────
    for (std::size_t j = count; j < total; ++j) {
        RC_ASSERT(field_p(j) == field_np(j));
    }
}

// ─── Global MPI + Kokkos + HALO environment ─────────────────────────────────

class HaloMpiEnvironment : public ::testing::Environment {
   public:
    void SetUp() override {
        int provided = 0;
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
        Kokkos::initialize();
        halo::Environment::initialize();

        MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &g_size);
    }

    void TearDown() override {
        Kokkos::finalize();
        MPI_Finalize();
    }
};

}  // namespace

// Register the environment (gtest_main provides main()).
static ::testing::Environment *const halo_mpi_env = ::testing::AddGlobalTestEnvironment(new HaloMpiEnvironment);
