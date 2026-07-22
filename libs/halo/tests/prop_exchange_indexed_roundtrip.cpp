// ─── Property-Based Tests: halo::exchange_indexed round-trip ────────────────
// Feature: cpp-dycore-halo-exchange, Property 3
//
// Uses RapidCheck with real MPI to verify the indexed gather/scatter exchange
// round-trip over randomized inputs. The test runs single-process with an
// in-process paired send/recv index model: the plan's sole send and receive
// neighbor is the running rank itself (self-send), so the owned values gathered
// from the send indices are communicated (through the real MPI self-send path
// inside exchange_indexed) and scattered back into the paired receive (halo)
// indices on the same rank. The k-th send index is paired with the k-th receive
// index, so each halo cell must end up holding the source-owned value of its
// paired send index. No mpirun launch is required.
//
// Property 3: Indexed exchange round-trip fills halo with source-owned values
//   For any field view (rank-1 indexed by element, or rank-2 nVertLevels x
//   nElements in LayoutLeft), and for any paired send/receive index lists
//   between a sending and receiving side, after an indexed exchange each receive
//   index holds exactly the value(s) that resided at the paired send index on
//   the sending side; for rank-2 fields every vertical level is transferred.
//
//   Validates: Requirements 2.1, 2.3, 3.1, 3.2, 3.5
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <mpi.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <numeric>
#include <span>
#include <vector>

#include "halo/communicator.hpp"
#include "halo/environment.hpp"
#include "halo/exchange_indexed.hpp"
#include "halo/indexed_halo_plan.hpp"

namespace {

// ─── Globals for MPI rank / size ────────────────────────────────────────────
int g_rank = 0;
int g_size = 0;

// A distinct value used to pre-fill halo (receive) slots so that halo elements
// that are *not* receive targets can be confirmed unchanged after the exchange.
constexpr double SENTINEL = -777.0;

/// Distinct owned value for a given local owned index on this rank. Distinct
/// across owned indices so a delivered halo value uniquely identifies its
/// source owned index.
double owned_value(std::size_t owned_index) {
    return 1000.0 * static_cast<double>(g_rank + 1) + static_cast<double>(owned_index);
}

/// Per-(index, level) owned value for rank-2 fields; distinct per (index, level).
double owned_value_r2(std::size_t owned_index, std::size_t level) {
    return owned_value(owned_index) * 10.0 + static_cast<double>(level);
}

/// A randomly generated paired send/recv index configuration for one property
/// iteration. Owned indices occupy [0, n_owned); halo indices occupy
/// [n_owned, n_owned + n_halo). The k-th send index is paired with the k-th
/// receive index. Send indices may repeat (an owned value may fan out to
/// several halo cells); receive indices are distinct so each halo cell is
/// written exactly once and the round-trip is well defined.
struct Paired_Indices {
    std::vector<std::size_t> send_idx;  // owned indices to gather (position-paired)
    std::vector<std::size_t> recv_idx;  // distinct halo indices to fill (position-paired)
    std::size_t n_owned = 0;
    std::size_t n_halo = 0;
    std::size_t n_elements = 0;
};

Paired_Indices generate_paired_indices() {
    Paired_Indices pi;

    const int n_owned = *rc::gen::inRange(1, 41);
    const int n_halo = *rc::gen::inRange(1, 41);
    const int n_pairs = *rc::gen::inRange(1, n_halo + 1);

    // Distinct receive indices drawn from the halo region.
    std::vector<std::size_t> halo_slots(static_cast<std::size_t>(n_halo));
    std::iota(halo_slots.begin(), halo_slots.end(), static_cast<std::size_t>(n_owned));

    pi.recv_idx.reserve(static_cast<std::size_t>(n_pairs));
    for (int i = 0; i < n_pairs; ++i) {
        const int pick = *rc::gen::inRange(0, static_cast<int>(halo_slots.size()));
        pi.recv_idx.push_back(halo_slots[static_cast<std::size_t>(pick)]);
        halo_slots.erase(halo_slots.begin() + pick);
    }

    // Send indices from the owned region; repeats allowed.
    pi.send_idx.reserve(static_cast<std::size_t>(n_pairs));
    for (int i = 0; i < n_pairs; ++i) {
        pi.send_idx.push_back(static_cast<std::size_t>(*rc::gen::inRange(0, n_owned)));
    }

    pi.n_owned = static_cast<std::size_t>(n_owned);
    pi.n_halo = static_cast<std::size_t>(n_halo);
    pi.n_elements = pi.n_owned + pi.n_halo;
    return pi;
}

/// Build a single-layer self-send plan for the given paired indices. The whole
/// exchange happens on halo layer 0, and the exchange selects that single layer.
halo::Indexed_Halo_Plan make_self_plan(const halo::Communicator &comm, const Paired_Indices &pi) {
    halo::Indexed_Neighbor send_nbr{g_rank, {pi.send_idx}};
    halo::Indexed_Neighbor recv_nbr{g_rank, {pi.recv_idx}};
    return halo::Indexed_Halo_Plan(comm, halo::Element_Kind::cell, {send_nbr}, {recv_nbr});
}

}  // namespace

// ─── Property 3 (rank-1): round-trip fills halo with source-owned values ─────
// Feature: cpp-dycore-halo-exchange, Property 3
//
// **Validates: Requirements 2.1, 2.3, 3.1, 3.5**

RC_GTEST_PROP(ExchangeIndexedProperty3, Rank1RoundTripFillsHalo, ()) {
    const Paired_Indices pi = generate_paired_indices();
    const int layer0[] = {0};

    halo::Communicator comm(MPI_COMM_WORLD);
    halo::Indexed_Halo_Plan plan = make_self_plan(comm, pi);

    // Seed: halo slots to SENTINEL, owned slots to distinct known values.
    Kokkos::View<double *, Kokkos::HostSpace> field("field_r1", pi.n_elements);
    for (std::size_t i = 0; i < pi.n_elements; ++i) {
        field(i) = SENTINEL;
    }
    for (std::size_t si = 0; si < pi.n_owned; ++si) {
        field(si) = owned_value(si);
    }

    halo::exchange_indexed(plan, field, std::span<const int>(layer0, 1));

    // Each receive index holds the paired send index's source-owned value.
    for (std::size_t k = 0; k < pi.recv_idx.size(); ++k) {
        RC_ASSERT(field(pi.recv_idx[k]) == owned_value(pi.send_idx[k]));
    }

    // Owned elements are never written by the exchange.
    for (std::size_t si = 0; si < pi.n_owned; ++si) {
        RC_ASSERT(field(si) == owned_value(si));
    }

    // Halo cells that were not receive targets keep the sentinel.
    std::vector<char> is_target(pi.n_elements, 0);
    for (std::size_t idx : pi.recv_idx) {
        is_target[idx] = 1;
    }
    for (std::size_t i = pi.n_owned; i < pi.n_elements; ++i) {
        if (!is_target[i]) {
            RC_ASSERT(field(i) == SENTINEL);
        }
    }
}

// ─── Property 3 (rank-2): every vertical level is transferred ────────────────
// Feature: cpp-dycore-halo-exchange, Property 3
//
// Rank-2 field of shape (nVertLevels x nElements) in LayoutLeft: after the
// round-trip each receive column holds, for every vertical level, the paired
// send column's value; halo columns that are not receive targets are unchanged.
//
// **Validates: Requirements 2.1, 2.3, 3.2, 3.5**

RC_GTEST_PROP(ExchangeIndexedProperty3, Rank2RoundTripTransfersAllLevels, ()) {
    const Paired_Indices pi = generate_paired_indices();
    const auto nlev = static_cast<std::size_t>(*rc::gen::inRange(1, 12));
    const int layer0[] = {0};

    halo::Communicator comm(MPI_COMM_WORLD);
    halo::Indexed_Halo_Plan plan = make_self_plan(comm, pi);

    // Rank-2 view: (nVertLevels x nElements) in LayoutLeft, matching the dycore.
    Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> field("field_r2", nlev, pi.n_elements);
    for (std::size_t j = 0; j < pi.n_elements; ++j) {
        for (std::size_t lev = 0; lev < nlev; ++lev) {
            field(lev, j) = SENTINEL;
        }
    }
    for (std::size_t si = 0; si < pi.n_owned; ++si) {
        for (std::size_t lev = 0; lev < nlev; ++lev) {
            field(lev, si) = owned_value_r2(si, lev);
        }
    }

    halo::exchange_indexed(plan, field, std::span<const int>(layer0, 1));

    // Each receive column holds the paired send column's values, all levels.
    for (std::size_t k = 0; k < pi.recv_idx.size(); ++k) {
        const std::size_t src = pi.send_idx[k];
        const std::size_t dst = pi.recv_idx[k];
        for (std::size_t lev = 0; lev < nlev; ++lev) {
            RC_ASSERT(field(lev, dst) == owned_value_r2(src, lev));
        }
    }

    // Owned columns are never written by the exchange.
    for (std::size_t si = 0; si < pi.n_owned; ++si) {
        for (std::size_t lev = 0; lev < nlev; ++lev) {
            RC_ASSERT(field(lev, si) == owned_value_r2(si, lev));
        }
    }

    // Halo columns that were not receive targets keep the sentinel at all levels.
    std::vector<char> is_target(pi.n_elements, 0);
    for (std::size_t idx : pi.recv_idx) {
        is_target[idx] = 1;
    }
    for (std::size_t j = pi.n_owned; j < pi.n_elements; ++j) {
        if (!is_target[j]) {
            for (std::size_t lev = 0; lev < nlev; ++lev) {
                RC_ASSERT(field(lev, j) == SENTINEL);
            }
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
