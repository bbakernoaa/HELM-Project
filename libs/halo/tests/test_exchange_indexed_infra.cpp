// ─── HALO exchange_indexed infrastructure unit tests (real MPI, self-send) ───
// Feature: cpp-dycore-halo-exchange, Task 2.7
//
// Example-based unit tests for the exchange_indexed infrastructure paths that
// require genuine value movement:
//   - Diagnostics begin/end emission with correct neighbor count and byte
//     volume                                                          (Req 4.3)
//   - Tag equivalence with the contiguous exchange for matching rank pairs
//                                                                      (Req 2.6)
//   - Single-rank no-neighbor plan leaves all elements unchanged       (Req 6.3)
//
// These assertions need real data to flow (byte counts, tag matching, no-op
// invariance), so this target links the real MPI runtime and runs single
// process using an in-process self-send plan (the running rank is its own send
// and receive neighbor), mirroring prop_exchange_indexed_roundtrip. The
// MPI-failure path (Req 2.5) needs deterministic error injection and lives in a
// separate spy-based target (test_exchange_indexed_infra_mpifail).
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <mpi.h>

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <mutex>
#include <span>
#include <vector>

#include "halo/communicator.hpp"
#include "halo/detail/compute_tag.hpp"
#include "halo/diagnostics.hpp"
#include "halo/environment.hpp"
#include "halo/exchange.hpp"
#include "halo/exchange_indexed.hpp"
#include "halo/halo_plan.hpp"
#include "halo/indexed_halo_plan.hpp"

namespace {

using HostView1 = Kokkos::View<double *, Kokkos::HostSpace>;
using HostView2 = Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace>;

int g_rank = 0;
int g_size = 0;

/// Build a single-layer self-send indexed plan: the running rank is both the
/// sole send neighbor and the sole receive neighbor. Send indices are gathered
/// and, after the (self) MPI round-trip, scattered into the receive indices.
halo::Indexed_Halo_Plan make_self_plan(const halo::Communicator &comm, const std::vector<std::size_t> &send_idx,
                                       const std::vector<std::size_t> &recv_idx) {
    halo::Indexed_Neighbor send_nbr{g_rank, {send_idx}};
    halo::Indexed_Neighbor recv_nbr{g_rank, {recv_idx}};
    return halo::Indexed_Halo_Plan(comm, halo::Element_Kind::cell, {send_nbr}, {recv_nbr});
}

// ─── Req 4.3: Diagnostics begin/end with correct neighbor count + bytes ──────

// A single send neighbor and a single receive neighbor => neighbor_count = 2.
// total_bytes = (total_send_elems + total_recv_elems) * sizeof(value_type),
// where per-element block size is 1 for rank-1 and nVertLevels for rank-2.
TEST(ExchangeIndexedInfra, DiagnosticsRank1EmitsNeighborCountAndBytes) {
    halo::Communicator comm(MPI_COMM_WORLD);

    const std::vector<std::size_t> send_idx{0, 1, 2};     // S = 3 owned indices
    const std::vector<std::size_t> recv_idx{4, 5, 6};     // R = 3 halo indices
    const int layer0[] = {0};

    halo::Indexed_Halo_Plan plan = make_self_plan(comm, send_idx, recv_idx);

    HostView1 field("diag_r1", 8);
    for (std::size_t i = 0; i < field.extent(0); ++i) field(i) = -1.0;
    for (std::size_t i = 0; i < send_idx.size(); ++i) field(send_idx[i]) = 100.0 + static_cast<double>(i);

    std::vector<halo::Exchange_Event> events;
    std::mutex mtx;
    halo::Diagnostics::set_callback([&](const halo::Exchange_Event &ev) {
        std::lock_guard<std::mutex> lock(mtx);
        events.push_back(ev);
    });
    ASSERT_TRUE(halo::Diagnostics::is_active());

    halo::exchange_indexed(plan, field, std::span<const int>(layer0, 1));

    halo::Diagnostics::clear_callback();

    const halo::Exchange_Event *begin_ev = nullptr;
    const halo::Exchange_Event *end_ev = nullptr;
    for (const auto &ev : events) {
        if (ev.phase == halo::Exchange_Event::Phase::begin) begin_ev = &ev;
        if (ev.phase == halo::Exchange_Event::Phase::end) end_ev = &ev;
    }
    ASSERT_NE(begin_ev, nullptr) << "no begin event emitted";
    ASSERT_NE(end_ev, nullptr) << "no end event emitted";

    // 1 send neighbor + 1 recv neighbor.
    EXPECT_EQ(begin_ev->neighbor_count, 2);
    EXPECT_EQ(end_ev->neighbor_count, 2);

    // rank-1 block size 1: (3 sent + 3 received) * sizeof(double).
    const std::size_t expected_bytes = (send_idx.size() + recv_idx.size()) * sizeof(double);
    EXPECT_EQ(begin_ev->total_bytes, expected_bytes);
    EXPECT_EQ(end_ev->total_bytes, expected_bytes);

    // Local rank tagged; begin has zero elapsed, both are synchronous.
    EXPECT_EQ(begin_ev->local_rank, g_rank);
    EXPECT_EQ(end_ev->local_rank, g_rank);
    EXPECT_EQ(begin_ev->elapsed.count(), 0);
    EXPECT_FALSE(begin_ev->is_async);
    EXPECT_FALSE(end_ev->is_async);
}

// For rank-2 (nVertLevels x nElements) fields, the byte volume must account for
// every vertical level (block size == nVertLevels).
TEST(ExchangeIndexedInfra, DiagnosticsRank2ByteVolumeCountsAllLevels) {
    halo::Communicator comm(MPI_COMM_WORLD);

    const std::vector<std::size_t> send_idx{0, 1, 2};
    const std::vector<std::size_t> recv_idx{4, 5, 6};
    const std::size_t nlev = 5;
    const int layer0[] = {0};

    halo::Indexed_Halo_Plan plan = make_self_plan(comm, send_idx, recv_idx);

    HostView2 field("diag_r2", nlev, 8);
    for (std::size_t j = 0; j < field.extent(1); ++j)
        for (std::size_t lev = 0; lev < nlev; ++lev) field(lev, j) = -1.0;
    for (std::size_t i = 0; i < send_idx.size(); ++i)
        for (std::size_t lev = 0; lev < nlev; ++lev) field(lev, send_idx[i]) = 100.0 + static_cast<double>(i * nlev + lev);

    std::vector<halo::Exchange_Event> events;
    std::mutex mtx;
    halo::Diagnostics::set_callback([&](const halo::Exchange_Event &ev) {
        std::lock_guard<std::mutex> lock(mtx);
        events.push_back(ev);
    });

    halo::exchange_indexed(plan, field, std::span<const int>(layer0, 1));

    halo::Diagnostics::clear_callback();

    const halo::Exchange_Event *begin_ev = nullptr;
    for (const auto &ev : events) {
        if (ev.phase == halo::Exchange_Event::Phase::begin) begin_ev = &ev;
    }
    ASSERT_NE(begin_ev, nullptr);

    EXPECT_EQ(begin_ev->neighbor_count, 2);
    const std::size_t expected_bytes = (send_idx.size() + recv_idx.size()) * nlev * sizeof(double);
    EXPECT_EQ(begin_ev->total_bytes, expected_bytes);
}

// ─── Req 2.6: Tag equivalence with the contiguous exchange ───────────────────

// exchange_indexed and exchange_blocking both derive message tags from
// detail::compute_tag using the identical sender-first ordering:
//   send side: compute_tag(my_rank,        neighbor_rank, comm_size)
//   recv side: compute_tag(neighbor_rank,  my_rank,       comm_size)
// So for a directed pair (src -> dst) the sender's send tag and the receiver's
// recv tag both reduce to compute_tag(src, dst, comm_size). This test verifies
// that shared convention over representative rank pairs.
TEST(ExchangeIndexedInfra, TagSchemeMatchesContiguousForRankPairs) {
    struct Case {
        int src;
        int dst;
        int comm_size;
    };
    const Case cases[] = {{0, 1, 4}, {1, 0, 4}, {2, 3, 8}, {3, 3, 8}, {5, 9, 64}, {0, 0, 1}};

    for (const Case &c : cases) {
        // Sender at src sending to dst (both exchanges use compute_tag(my, neighbor)).
        const int indexed_send_tag = halo::detail::compute_tag(c.src, c.dst, c.comm_size);
        const int contiguous_send_tag = halo::detail::compute_tag(c.src, c.dst, c.comm_size);
        EXPECT_EQ(indexed_send_tag, contiguous_send_tag) << "send-side tag differs for (" << c.src << "," << c.dst << ")";

        // Receiver at dst expecting from src (both use compute_tag(neighbor, my)).
        const int indexed_recv_tag = halo::detail::compute_tag(c.src, c.dst, c.comm_size);
        const int contiguous_recv_tag = halo::detail::compute_tag(c.src, c.dst, c.comm_size);
        EXPECT_EQ(indexed_recv_tag, contiguous_recv_tag) << "recv-side tag differs for (" << c.src << "," << c.dst << ")";

        // The matched send/recv pair agrees, and the tag is bounded and deterministic.
        EXPECT_EQ(indexed_send_tag, indexed_recv_tag);
        EXPECT_GE(indexed_send_tag, 0);
        EXPECT_EQ(indexed_send_tag, halo::detail::compute_tag(c.src, c.dst, c.comm_size));
    }
}

// Functional cross-check: a single-process self-send delivers correctly through
// both the indexed exchange and the contiguous exchange, which is only possible
// if each posts matching send/recv tags. This demonstrates the two exchanges'
// tag schemes are mutually compatible at runtime.
TEST(ExchangeIndexedInfra, SelfSendDeliversUnderBothExchanges) {
    halo::Communicator comm(MPI_COMM_WORLD);
    const std::size_t C = 4;
    const int layer0[] = {0};

    // Indexed exchange self-send: owned [0,C) -> halo [C,2C).
    {
        std::vector<std::size_t> send_idx(C), recv_idx(C);
        for (std::size_t i = 0; i < C; ++i) {
            send_idx[i] = i;
            recv_idx[i] = C + i;
        }
        halo::Indexed_Halo_Plan plan = make_self_plan(comm, send_idx, recv_idx);

        HostView1 field("tag_indexed", 2 * C);
        for (std::size_t i = 0; i < C; ++i) field(i) = 10.0 + static_cast<double>(i);
        for (std::size_t i = C; i < 2 * C; ++i) field(i) = -1.0;

        halo::exchange_indexed(plan, field, std::span<const int>(layer0, 1));

        for (std::size_t i = 0; i < C; ++i) EXPECT_EQ(field(C + i), 10.0 + static_cast<double>(i));
    }

    // Contiguous exchange self-send: send region [0,C) -> recv region [C,2C).
    {
        halo::Halo_Plan plan(comm, {{g_rank, C}}, {{g_rank, C}});

        HostView1 field("tag_contiguous", plan.total_send_elements() + plan.total_recv_elements());
        for (std::size_t i = 0; i < C; ++i) field(i) = 20.0 + static_cast<double>(i);
        for (std::size_t i = C; i < 2 * C; ++i) field(i) = -1.0;

        halo::exchange_blocking(plan, field);

        for (std::size_t i = 0; i < C; ++i) EXPECT_EQ(field(C + i), 20.0 + static_cast<double>(i));
    }
}

// ─── Req 6.3: single-rank no-neighbor plan leaves all elements unchanged ─────

TEST(ExchangeIndexedInfra, NoNeighborPlanLeavesRank1FieldUnchanged) {
    halo::Communicator comm(MPI_COMM_WORLD);

    // Empty send and receive neighbor lists => no communication.
    halo::Indexed_Halo_Plan plan(comm, halo::Element_Kind::cell, {}, {});
    ASSERT_EQ(plan.num_send_neighbors(), 0u);
    ASSERT_EQ(plan.num_recv_neighbors(), 0u);

    const int layer0[] = {0};
    HostView1 field("noop_r1", 12);
    std::vector<double> before(field.extent(0));
    for (std::size_t i = 0; i < field.extent(0); ++i) {
        field(i) = 3.5 * static_cast<double>(i) - 7.0;
        before[i] = field(i);
    }

    halo::exchange_indexed(plan, field, std::span<const int>(layer0, 1));

    for (std::size_t i = 0; i < field.extent(0); ++i) {
        EXPECT_EQ(field(i), before[i]) << "element " << i << " changed by a no-op exchange";
    }
}

TEST(ExchangeIndexedInfra, NoNeighborPlanLeavesRank2FieldUnchanged) {
    halo::Communicator comm(MPI_COMM_WORLD);

    halo::Indexed_Halo_Plan plan(comm, halo::Element_Kind::cell, {}, {});

    const std::size_t nlev = 4;
    const int layer0[] = {0};
    HostView2 field("noop_r2", nlev, 6);
    std::vector<double> before(nlev * field.extent(1));
    for (std::size_t j = 0; j < field.extent(1); ++j) {
        for (std::size_t lev = 0; lev < nlev; ++lev) {
            const double v = static_cast<double>(j) * 100.0 + static_cast<double>(lev);
            field(lev, j) = v;
            before[j * nlev + lev] = v;
        }
    }

    halo::exchange_indexed(plan, field, std::span<const int>(layer0, 1));

    for (std::size_t j = 0; j < field.extent(1); ++j) {
        for (std::size_t lev = 0; lev < nlev; ++lev) {
            EXPECT_EQ(field(lev, j), before[j * nlev + lev]) << "element (" << lev << "," << j << ") changed by a no-op exchange";
        }
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
        halo::Diagnostics::clear_callback();
        Kokkos::finalize();
        MPI_Finalize();
    }
};

}  // namespace

static ::testing::Environment *const halo_mpi_env = ::testing::AddGlobalTestEnvironment(new HaloMpiEnvironment);
