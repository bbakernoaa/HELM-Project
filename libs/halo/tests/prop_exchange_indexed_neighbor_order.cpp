// ─── Property-Based Tests: halo::exchange_indexed neighbor-order independence ─
// Feature: cpp-dycore-halo-exchange, Property 8
//
// Uses RapidCheck with real MPI (mpirun -np 4) to verify that the indexed
// gather/scatter exchange produces identical field-view results regardless of
// the order in which the plan lists its neighbors (and, consequently, the order
// in which sends/recvs are posted).
//
// Property 8 fundamentally requires MORE THAN ONE neighbor to be meaningful:
// permuting a one-element neighbor list is a no-op. A single-process plan can
// only ever have one neighbor per direction (the running rank itself), because
// Indexed_Halo_Plan rejects duplicate ranks within a send or receive list. We
// therefore exercise the property over the real multi-rank MPI runtime (np = 4,
// like prop_structured_exchange), where each rank has several distinct
// neighbors and a genuine permutation of the neighbor order can be applied.
//
// Strategy: every rank builds a canonical plan whose send/recv neighbor lists
// are the other ranks in ascending order, and a "permuted" plan with exactly the
// same per-neighbor index lists but with the neighbor entries reordered by a
// randomly generated permutation. Because receives are scattered by index into
// disjoint halo slots, the final field view must be bitwise identical for the
// two orderings.
//
// Property 8: Neighbor-order independence
//   For any field view and Indexed_Halo_Plan, performing the indexed exchange
//   with the plan's neighbors in any permutation of order produces identical
//   field-view values after completion.
//
//   Validates: Requirements 12.1
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

// A distinct value used to pre-fill halo (receive) slots. Halo cells that end
// up receiving data overwrite it; either way both orderings must agree.
constexpr double SENTINEL = -777.0;

/// Distinct owned value for a given local owned index on this rank.
double owned_value(std::size_t owned_index) {
    return 1000.0 * static_cast<double>(g_rank + 1) + static_cast<double>(owned_index);
}

/// Per-(index, level) owned value for rank-2 fields; distinct per (index, level).
double owned_value_r2(std::size_t owned_index, std::size_t level) {
    return owned_value(owned_index) * 10.0 + static_cast<double>(level);
}

/// Geometry shared by all ranks for one property iteration. Generated on rank 0
/// with RapidCheck, then broadcast so every rank agrees on the message sizes and
/// the neighbor-order permutation.
struct Topology {
    int n_owned = 0;
    // counts[a*size + b] == number of elements rank a sends to rank b
    //                    == number of elements rank b receives from a.
    std::vector<int> counts;
    // A permutation of [0, size-1): positional reordering applied identically to
    // every rank's (length size-1) neighbor list to produce the permuted plan.
    std::vector<int> perm;
};

/// Generate the shared topology on rank 0 and broadcast it to all ranks.
Topology generate_topology() {
    Topology topo;
    const int size = g_size;

    int n_owned = 0;
    std::vector<int> counts(static_cast<std::size_t>(size * size), 0);
    std::vector<int> perm(static_cast<std::size_t>(size - 1));

    if (g_rank == 0) {
        n_owned = *rc::gen::inRange(4, 33);
        const int max_count = *rc::gen::inRange(0, 5);
        for (int a = 0; a < size; ++a) {
            for (int b = 0; b < size; ++b) {
                if (a != b) {
                    counts[static_cast<std::size_t>(a * size + b)] = *rc::gen::inRange(0, max_count + 1);
                }
            }
        }
        // Build a permutation of [0, size-1).
        std::iota(perm.begin(), perm.end(), 0);
        for (int i = static_cast<int>(perm.size()) - 1; i > 0; --i) {
            const int j = *rc::gen::inRange(0, i + 1);
            std::swap(perm[static_cast<std::size_t>(i)], perm[static_cast<std::size_t>(j)]);
        }
    }

    MPI_Bcast(&n_owned, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(counts.data(), size * size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(perm.data(), size - 1, MPI_INT, 0, MPI_COMM_WORLD);

    topo.n_owned = n_owned;
    topo.counts = std::move(counts);
    topo.perm = std::move(perm);
    return topo;
}

/// The per-rank plan layout derived from the shared topology: the canonical
/// (ascending-rank) send/recv neighbor lists, plus the total halo size so the
/// caller can allocate the [owned | halo] field view.
struct Local_Plan_Data {
    std::vector<halo::Indexed_Neighbor> send_neighbors;  // ascending neighbor rank
    std::vector<halo::Indexed_Neighbor> recv_neighbors;  // ascending neighbor rank
    std::size_t n_owned = 0;
    std::size_t n_halo = 0;
    std::size_t n_elements = 0;
};

/// Build this rank's canonical neighbor lists from the shared topology.
///
/// Send indices gather owned elements (indices modulo n_owned). Receive indices
/// occupy a distinct contiguous halo block per neighbor, assigned in ascending
/// neighbor-rank order, so the blocks are disjoint no matter how the neighbor
/// entries are later permuted. All lists use a single halo layer (layer 0).
Local_Plan_Data build_local_plan(const Topology &topo) {
    Local_Plan_Data data;
    const int size = g_size;
    const auto n_owned = static_cast<std::size_t>(topo.n_owned);
    data.n_owned = n_owned;

    std::size_t halo_offset = 0;
    for (int q = 0; q < size; ++q) {
        if (q == g_rank) {
            continue;
        }

        // Send to q: gather this many owned values (may be 0).
        const int send_count = topo.counts[static_cast<std::size_t>(g_rank * size + q)];
        std::vector<std::size_t> send_idx;
        send_idx.reserve(static_cast<std::size_t>(send_count));
        for (int k = 0; k < send_count; ++k) {
            send_idx.push_back(static_cast<std::size_t>(k) % n_owned);
        }
        data.send_neighbors.push_back(halo::Indexed_Neighbor{q, {std::move(send_idx)}});

        // Receive from q: fill this many distinct halo slots (may be 0).
        const int recv_count = topo.counts[static_cast<std::size_t>(q * size + g_rank)];
        std::vector<std::size_t> recv_idx;
        recv_idx.reserve(static_cast<std::size_t>(recv_count));
        for (int k = 0; k < recv_count; ++k) {
            recv_idx.push_back(n_owned + halo_offset + static_cast<std::size_t>(k));
        }
        halo_offset += static_cast<std::size_t>(recv_count);
        data.recv_neighbors.push_back(halo::Indexed_Neighbor{q, {std::move(recv_idx)}});
    }

    data.n_halo = halo_offset;
    data.n_elements = data.n_owned + data.n_halo;
    return data;
}

/// Apply a positional permutation to a neighbor list. The neighbor entries (and
/// their per-layer index lists) are unchanged; only their order in the list
/// differs.
std::vector<halo::Indexed_Neighbor> permute_neighbors(const std::vector<halo::Indexed_Neighbor> &neighbors,
                                                      const std::vector<int> &perm) {
    std::vector<halo::Indexed_Neighbor> out;
    out.reserve(neighbors.size());
    for (int p : perm) {
        out.push_back(neighbors[static_cast<std::size_t>(p)]);
    }
    return out;
}

}  // namespace

// ─── Property 8 (rank-1): neighbor-order independence ────────────────────────
// Feature: cpp-dycore-halo-exchange, Property 8
//
// **Validates: Requirements 12.1**

RC_GTEST_PROP(ExchangeIndexedProperty8, Rank1NeighborOrderIndependent, ()) {
    if (g_size < 2) {
        return;  // Property is only meaningful with more than one neighbor.
    }

    const Topology topo = generate_topology();
    const Local_Plan_Data data = build_local_plan(topo);
    const int layer0[] = {0};

    halo::Communicator comm(MPI_COMM_WORLD);

    // Canonical (ascending neighbor rank) plan.
    halo::Indexed_Halo_Plan plan_ref(comm, halo::Element_Kind::cell, data.send_neighbors, data.recv_neighbors);
    // Permuted-neighbor plan: identical index lists, reordered neighbor entries.
    halo::Indexed_Halo_Plan plan_perm(comm, halo::Element_Kind::cell,
                                      permute_neighbors(data.send_neighbors, topo.perm),
                                      permute_neighbors(data.recv_neighbors, topo.perm));

    // Two identically initialized field views: owned slots hold distinct known
    // values, halo slots hold the sentinel.
    auto seed_field = [&](Kokkos::View<double *, Kokkos::HostSpace> &field) {
        for (std::size_t i = 0; i < data.n_elements; ++i) {
            field(i) = SENTINEL;
        }
        for (std::size_t si = 0; si < data.n_owned; ++si) {
            field(si) = owned_value(si);
        }
    };

    Kokkos::View<double *, Kokkos::HostSpace> field_ref("field_ref_r1", data.n_elements);
    Kokkos::View<double *, Kokkos::HostSpace> field_perm("field_perm_r1", data.n_elements);
    seed_field(field_ref);
    seed_field(field_perm);

    halo::exchange_indexed(plan_ref, field_ref, std::span<const int>(layer0, 1));
    halo::exchange_indexed(plan_perm, field_perm, std::span<const int>(layer0, 1));

    // Neighbor-order independence: every element agrees between the two orderings.
    for (std::size_t i = 0; i < data.n_elements; ++i) {
        RC_ASSERT(field_ref(i) == field_perm(i));
    }
}

// ─── Property 8 (rank-2): neighbor-order independence, all levels ────────────
// Feature: cpp-dycore-halo-exchange, Property 8
//
// Rank-2 field of shape (nVertLevels x nElements) in LayoutLeft: permuting the
// neighbor order must leave every vertical level of every element unchanged
// relative to the canonical order.
//
// **Validates: Requirements 12.1**

RC_GTEST_PROP(ExchangeIndexedProperty8, Rank2NeighborOrderIndependent, ()) {
    if (g_size < 2) {
        return;  // Property is only meaningful with more than one neighbor.
    }

    const Topology topo = generate_topology();
    const Local_Plan_Data data = build_local_plan(topo);
    const int layer0[] = {0};

    // Random number of vertical levels, agreed across ranks.
    int nlev_int = 0;
    if (g_rank == 0) {
        nlev_int = *rc::gen::inRange(1, 12);
    }
    MPI_Bcast(&nlev_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
    const auto nlev = static_cast<std::size_t>(nlev_int);

    halo::Communicator comm(MPI_COMM_WORLD);

    halo::Indexed_Halo_Plan plan_ref(comm, halo::Element_Kind::cell, data.send_neighbors, data.recv_neighbors);
    halo::Indexed_Halo_Plan plan_perm(comm, halo::Element_Kind::cell,
                                      permute_neighbors(data.send_neighbors, topo.perm),
                                      permute_neighbors(data.recv_neighbors, topo.perm));

    auto seed_field = [&](Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> &field) {
        for (std::size_t j = 0; j < data.n_elements; ++j) {
            for (std::size_t lev = 0; lev < nlev; ++lev) {
                field(lev, j) = SENTINEL;
            }
        }
        for (std::size_t si = 0; si < data.n_owned; ++si) {
            for (std::size_t lev = 0; lev < nlev; ++lev) {
                field(lev, si) = owned_value_r2(si, lev);
            }
        }
    };

    Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> field_ref("field_ref_r2", nlev, data.n_elements);
    Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> field_perm("field_perm_r2", nlev, data.n_elements);
    seed_field(field_ref);
    seed_field(field_perm);

    halo::exchange_indexed(plan_ref, field_ref, std::span<const int>(layer0, 1));
    halo::exchange_indexed(plan_perm, field_perm, std::span<const int>(layer0, 1));

    for (std::size_t j = 0; j < data.n_elements; ++j) {
        for (std::size_t lev = 0; lev < nlev; ++lev) {
            RC_ASSERT(field_ref(lev, j) == field_perm(lev, j));
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
