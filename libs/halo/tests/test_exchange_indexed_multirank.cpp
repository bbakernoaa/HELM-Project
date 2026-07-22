// ─── Indexed Halo Exchange Multi-Rank Integration Test (real MPI, np=4) ──────
// Feature: cpp-dycore-halo-exchange
// Task 4.1: multi-rank MPI integration test
//
// Exercises halo::exchange_indexed on a small synthetic unstructured partition
// distributed across the real MPI runtime (mpirun -np 4, meaningful for 2-4
// ranks). Every rank owns a contiguous block of cells in [0, n_owned) and holds
// a distinct block of halo (ghost) cells in [n_owned, n_owned + n_halo). Owned
// values are seeded as an analytic function of (rank, index); after the
// exchange every halo cell must hold exactly the source-owned value that its
// paired send index carried on the neighbor that owns it. This validates
// Property 3 (round-trip fills halo with source-owned values) over the genuine
// gather / MPI Irecv+Isend+Waitall / scatter path rather than the in-process
// self-send model used by prop_exchange_indexed_roundtrip.
//
// Synthetic topology (deterministic, no broadcast needed):
//   - Neighbors of a rank are every OTHER rank, in ascending rank order.
//   - send_count(a, b) = 1 + ((a + 2*b) % 3)  (an integer in [1, 3]) is the
//     number of owned cells rank a sends to rank b. Because both endpoints can
//     evaluate this function, rank r knows it receives send_count(q, r) cells
//     from neighbor q without any metadata exchange.
//   - Rank r gathers owned indices [0, send_count(r, q)) for neighbor q.
//   - Rank r fills a contiguous, disjoint halo block per neighbor (assigned in
//     ascending neighbor-rank order), so halo slots are written exactly once.
//   - The k-th cell received from q corresponds to q's owned index k, so its
//     expected value is owned_value(q, k) (rank-1) or owned_value_r2(q, k, lev)
//     (rank-2, every vertical level).
//
// Covers rank-1 (nElements) and rank-2 (nVertLevels x nElements, LayoutLeft)
// fields over the real MPI path.
//
//   Requirements: 2.2, 3.3, 3.4, 3.5, 4.1, 4.2, 4.3, 4.4
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <mpi.h>

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <span>
#include <vector>

#include "halo/communicator.hpp"
#include "halo/environment.hpp"
#include "halo/exchange_indexed.hpp"
#include "halo/indexed_halo_plan.hpp"

namespace {

// ─── Synthetic partition constants ──────────────────────────────────────────

/// Number of owned cells per rank. Must be >= the maximum send count (3) so the
/// owned-index range [0, send_count) is always in bounds.
constexpr std::size_t kOwned = 8;

/// Number of vertical levels for the rank-2 field (nVertLevels).
constexpr std::size_t kLevels = 5;

/// Distinct value used to pre-fill halo slots before the exchange so untouched
/// slots (should be none in this topology) would be detectable.
constexpr double kSentinel = -777.0;

// ─── Analytic seeding functions ─────────────────────────────────────────────

/// Distinct owned value for (rank, owned_index). Distinct across ranks and
/// indices so a delivered halo value uniquely identifies its source.
double owned_value(int rank, std::size_t owned_index) {
    return 1000.0 * static_cast<double>(rank + 1) + static_cast<double>(owned_index);
}

/// Per-(rank, index, level) owned value for rank-2 fields.
double owned_value_r2(int rank, std::size_t owned_index, std::size_t level) {
    return owned_value(rank, owned_index) * 100.0 + static_cast<double>(level);
}

/// Deterministic number of cells rank `a` sends to rank `b` (in [1, 3]).
/// Evaluable by both endpoints, so no metadata exchange is required.
int send_count(int a, int b) { return 1 + ((a + 2 * b) % 3); }

// ─── Per-rank plan construction from the synthetic topology ──────────────────

struct Local_Plan_Data {
    std::vector<halo::Indexed_Neighbor> send_neighbors;  // ascending neighbor rank
    std::vector<halo::Indexed_Neighbor> recv_neighbors;  // ascending neighbor rank
    // For each recv neighbor entry (parallel to recv_neighbors), the neighbor
    // rank and the number of cells received, so the checker can recompute the
    // expected source-owned values.
    std::vector<int> recv_neighbor_rank;
    std::vector<int> recv_neighbor_count;
    std::size_t n_owned = kOwned;
    std::size_t n_halo = 0;
    std::size_t n_elements = 0;
};

/// Build this rank's send/recv neighbor lists (single halo layer, layer 0).
Local_Plan_Data build_local_plan(int rank, int size) {
    Local_Plan_Data data;

    std::size_t halo_offset = 0;
    for (int q = 0; q < size; ++q) {
        if (q == rank) {
            continue;
        }

        // Send to q: gather owned indices [0, send_count(rank, q)).
        const int sc = send_count(rank, q);
        std::vector<std::size_t> send_idx;
        send_idx.reserve(static_cast<std::size_t>(sc));
        for (int k = 0; k < sc; ++k) {
            send_idx.push_back(static_cast<std::size_t>(k));
        }
        data.send_neighbors.push_back(halo::Indexed_Neighbor{q, {std::move(send_idx)}});

        // Receive from q: fill a distinct contiguous halo block. The count is
        // send_count(q, rank) because that is how many q gathers for us.
        const int rc = send_count(q, rank);
        std::vector<std::size_t> recv_idx;
        recv_idx.reserve(static_cast<std::size_t>(rc));
        for (int k = 0; k < rc; ++k) {
            recv_idx.push_back(data.n_owned + halo_offset + static_cast<std::size_t>(k));
        }
        halo_offset += static_cast<std::size_t>(rc);
        data.recv_neighbors.push_back(halo::Indexed_Neighbor{q, {std::move(recv_idx)}});
        data.recv_neighbor_rank.push_back(q);
        data.recv_neighbor_count.push_back(rc);
    }

    data.n_halo = halo_offset;
    data.n_elements = data.n_owned + data.n_halo;
    return data;
}

// ─── Test fixture ─────────────────────────────────────────────────────────────

class IndexedExchangeMultiRankTest : public ::testing::Test {
   protected:
    void SetUp() override {
        comm_ = std::make_unique<halo::Communicator>(MPI_COMM_WORLD);
        rank_ = comm_->rank();
        size_ = comm_->size();
    }

    std::unique_ptr<halo::Communicator> comm_;
    int rank_ = 0;
    int size_ = 0;
};

}  // namespace

// ─── Rank-1 multi-rank round-trip ────────────────────────────────────────────
// Requirements: 2.2, 3.3, 3.5, 4.1, 4.2, 4.3, 4.4

TEST_F(IndexedExchangeMultiRankTest, Rank1RoundTripFillsHalo) {
    if (size_ < 2) {
        GTEST_SKIP() << "Multi-rank exchange requires at least 2 ranks";
    }

    const Local_Plan_Data data = build_local_plan(rank_, size_);
    const int layer0[] = {0};

    halo::Indexed_Halo_Plan plan(*comm_, halo::Element_Kind::cell, data.send_neighbors, data.recv_neighbors);

    // Seed owned slots with analytic values; halo slots with the sentinel.
    Kokkos::View<double *, Kokkos::HostSpace> field("field_mr_r1", data.n_elements);
    for (std::size_t i = 0; i < data.n_elements; ++i) {
        field(i) = kSentinel;
    }
    for (std::size_t si = 0; si < data.n_owned; ++si) {
        field(si) = owned_value(rank_, si);
    }

    halo::exchange_indexed(plan, field, std::span<const int>(layer0, 1));

    // Verify each halo cell against the analytic source value on the owning
    // neighbor. Walk the recv neighbors in the same order the plan was built so
    // the contiguous halo blocks line up.
    std::size_t halo_pos = data.n_owned;
    for (std::size_t n = 0; n < data.recv_neighbor_rank.size(); ++n) {
        const int q = data.recv_neighbor_rank[n];
        const int count = data.recv_neighbor_count[n];
        for (int k = 0; k < count; ++k) {
            const double expected = owned_value(q, static_cast<std::size_t>(k));
            EXPECT_DOUBLE_EQ(field(halo_pos), expected)
                << "rank " << rank_ << " halo slot " << halo_pos << " from neighbor " << q << " (k=" << k << ")";
            ++halo_pos;
        }
    }
    EXPECT_EQ(halo_pos, data.n_elements) << "every halo slot must be a receive target on rank " << rank_;

    // Owned cells are never written by the exchange.
    for (std::size_t si = 0; si < data.n_owned; ++si) {
        EXPECT_DOUBLE_EQ(field(si), owned_value(rank_, si)) << "owned cell " << si << " modified on rank " << rank_;
    }
}

// ─── Rank-2 multi-rank round-trip (nVertLevels x nElements, LayoutLeft) ───────
// Requirements: 2.2, 3.2, 3.3, 3.5, 4.1, 4.2, 4.3, 4.4

TEST_F(IndexedExchangeMultiRankTest, Rank2RoundTripTransfersAllLevels) {
    if (size_ < 2) {
        GTEST_SKIP() << "Multi-rank exchange requires at least 2 ranks";
    }

    const Local_Plan_Data data = build_local_plan(rank_, size_);
    const int layer0[] = {0};

    halo::Indexed_Halo_Plan plan(*comm_, halo::Element_Kind::cell, data.send_neighbors, data.recv_neighbors);

    // Rank-2 field: (nVertLevels x nElements) in LayoutLeft, matching the dycore.
    Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> field("field_mr_r2", kLevels, data.n_elements);
    for (std::size_t j = 0; j < data.n_elements; ++j) {
        for (std::size_t lev = 0; lev < kLevels; ++lev) {
            field(lev, j) = kSentinel;
        }
    }
    for (std::size_t si = 0; si < data.n_owned; ++si) {
        for (std::size_t lev = 0; lev < kLevels; ++lev) {
            field(lev, si) = owned_value_r2(rank_, si, lev);
        }
    }

    halo::exchange_indexed(plan, field, std::span<const int>(layer0, 1));

    // Verify each halo column against the analytic source value, all levels.
    std::size_t halo_pos = data.n_owned;
    for (std::size_t n = 0; n < data.recv_neighbor_rank.size(); ++n) {
        const int q = data.recv_neighbor_rank[n];
        const int count = data.recv_neighbor_count[n];
        for (int k = 0; k < count; ++k) {
            for (std::size_t lev = 0; lev < kLevels; ++lev) {
                const double expected = owned_value_r2(q, static_cast<std::size_t>(k), lev);
                EXPECT_DOUBLE_EQ(field(lev, halo_pos), expected)
                    << "rank " << rank_ << " halo col " << halo_pos << " lev " << lev << " from neighbor " << q << " (k=" << k << ")";
            }
            ++halo_pos;
        }
    }
    EXPECT_EQ(halo_pos, data.n_elements) << "every halo column must be a receive target on rank " << rank_;

    // Owned columns are never written by the exchange.
    for (std::size_t si = 0; si < data.n_owned; ++si) {
        for (std::size_t lev = 0; lev < kLevels; ++lev) {
            EXPECT_DOUBLE_EQ(field(lev, si), owned_value_r2(rank_, si, lev)) << "owned col " << si << " lev " << lev << " modified on rank " << rank_;
        }
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
    }

    void TearDown() override {
        Kokkos::finalize();
        MPI_Finalize();
    }
};

}  // namespace

// Register the environment (gtest_main provides main()).
static ::testing::Environment *const halo_mpi_env = ::testing::AddGlobalTestEnvironment(new HaloMpiEnvironment);
