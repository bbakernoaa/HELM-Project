// ─── Property-Based Tests: halo::exchange_indexed ───────────────────────────
// Feature: cpp-dycore-halo-exchange, Property 5
//
// Uses RapidCheck with real MPI to verify indexed gather/scatter exchange
// properties over randomized inputs. Each MPI rank runs an independent
// single-process (self-send) exchange: the plan's send and receive neighbor is
// the rank itself, so owned values gathered from the send indices are delivered
// back into the receive (halo) indices on the same rank. This validates the
// pure gather/scatter/layer-subset logic deterministically without cross-rank
// coordination.
//
// Property 5: Layer-subset exactness
//   For any Indexed_Halo_Plan and for any subset of halo layers, an indexed
//   exchange over that subset writes exactly the receive indices belonging to
//   the selected layers and leaves every halo element that belongs only to
//   excluded layers unchanged; the selected-layer receive indices receive the
//   correct source-owned values.
//
//   Validates: Requirements 5.2, 5.3
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <mpi.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <vector>

#include "halo/communicator.hpp"
#include "halo/environment.hpp"
#include "halo/exchange_indexed.hpp"
#include "halo/indexed_halo_plan.hpp"

namespace {

// ─── Globals for MPI rank / size ────────────────────────────────────────────
int g_rank = 0;
int g_size = 0;

// A distinct value used to pre-fill halo (receive) slots so that "unchanged"
// halo elements can be detected reliably after the exchange.
constexpr double SENTINEL = -777.0;

/// Distinct owned value for a given local owned index on this rank.
/// Distinct across owned indices so a delivered halo value uniquely identifies
/// its source owned index.
double owned_value(std::size_t owned_index) {
    return 1000.0 * static_cast<double>(g_rank + 1) + static_cast<double>(owned_index);
}

/// Per-(index, level) owned value for rank-2 fields.
double owned_value_r2(std::size_t owned_index, std::size_t level) {
    return owned_value(owned_index) * 10.0 + static_cast<double>(level);
}

/// The randomly generated topology for one property iteration.
struct Layered_Indices {
    std::vector<std::vector<std::size_t>> send_layers;  // send_layers[l] = owned indices
    std::vector<std::vector<std::size_t>> recv_layers;  // recv_layers[l] = halo indices
    std::vector<int> subset;                            // selected 0-based layer indices
    std::vector<bool> selected;                         // selected[l] == true if l in subset
    std::size_t total_owned = 0;                        // count of owned (send) indices
    std::size_t n_elements = 0;                         // field view element extent
};

/// Generate a random layered send/recv index structure with globally disjoint
/// owned indices (`[0, total_owned)`) and halo indices
/// (`[total_owned, 2*total_owned)`), plus a random layer subset. Per-layer send
/// and receive counts are equal so the self-send gather/scatter aligns
/// position-by-position within each layer.
Layered_Indices generate_layered_indices() {
    Layered_Indices li;

    const int num_layers = *rc::gen::inRange(1, 6);
    li.send_layers.resize(static_cast<std::size_t>(num_layers));
    li.recv_layers.resize(static_cast<std::size_t>(num_layers));
    li.selected.resize(static_cast<std::size_t>(num_layers), false);

    // First pass: choose per-layer counts and assign disjoint owned indices.
    std::vector<int> counts(static_cast<std::size_t>(num_layers));
    std::size_t total_owned = 0;
    for (int l = 0; l < num_layers; ++l) {
        counts[static_cast<std::size_t>(l)] = *rc::gen::inRange(0, 6);
        total_owned += static_cast<std::size_t>(counts[static_cast<std::size_t>(l)]);
    }

    // Owned indices occupy [0, total_owned); halo indices occupy
    // [total_owned, 2*total_owned). Both are globally disjoint across layers.
    std::size_t owned_cursor = 0;
    std::size_t halo_cursor = total_owned;
    for (int l = 0; l < num_layers; ++l) {
        const int n = counts[static_cast<std::size_t>(l)];
        auto &sl = li.send_layers[static_cast<std::size_t>(l)];
        auto &rl = li.recv_layers[static_cast<std::size_t>(l)];
        sl.reserve(static_cast<std::size_t>(n));
        rl.reserve(static_cast<std::size_t>(n));
        for (int k = 0; k < n; ++k) {
            sl.push_back(owned_cursor++);
            rl.push_back(halo_cursor++);
        }

        // Randomly select this layer for the subset.
        const bool pick = *rc::gen::arbitrary<bool>();
        li.selected[static_cast<std::size_t>(l)] = pick;
        if (pick) {
            li.subset.push_back(l);
        }
    }

    li.total_owned = total_owned;
    // Ensure a non-zero extent so the Kokkos view is always allocatable.
    li.n_elements = (2 * total_owned == 0) ? std::size_t{1} : 2 * total_owned;
    return li;
}

}  // namespace

// ─── Property 5: Layer-subset exactness (rank-1 field) ──────────────────────
// Feature: cpp-dycore-halo-exchange, Property 5
//
// **Validates: Requirements 5.2, 5.3**

RC_GTEST_PROP(ExchangeIndexedProperty5, LayerSubsetExactnessRank1, ()) {
    const Layered_Indices li = generate_layered_indices();

    halo::Communicator comm(MPI_COMM_WORLD);

    // Self-send plan: this rank both sends and receives from itself.
    halo::Indexed_Neighbor send_nbr{g_rank, li.send_layers};
    halo::Indexed_Neighbor recv_nbr{g_rank, li.recv_layers};
    halo::Indexed_Halo_Plan plan(comm, halo::Element_Kind::cell, {send_nbr}, {recv_nbr});

    // Seed the field: halo slots to SENTINEL, owned slots to distinct values.
    Kokkos::View<double *, Kokkos::HostSpace> field("field_r1", li.n_elements);
    for (std::size_t i = 0; i < li.n_elements; ++i) {
        field(i) = SENTINEL;
    }
    for (std::size_t si = 0; si < li.total_owned; ++si) {
        field(si) = owned_value(si);
    }

    // Exchange only the selected layer subset.
    halo::exchange_indexed(plan, field, std::span<const int>(li.subset));

    const std::size_t num_layers = li.send_layers.size();
    for (std::size_t l = 0; l < num_layers; ++l) {
        const auto &sl = li.send_layers[l];
        const auto &rl = li.recv_layers[l];
        if (li.selected[l]) {
            // Selected layers: each recv index holds its paired owned value.
            for (std::size_t k = 0; k < rl.size(); ++k) {
                RC_ASSERT(field(rl[k]) == owned_value(sl[k]));
            }
        } else {
            // Excluded layers: recv (halo) indices remain untouched.
            for (std::size_t k = 0; k < rl.size(); ++k) {
                RC_ASSERT(field(rl[k]) == SENTINEL);
            }
        }
    }

    // Owned elements are never written by the exchange.
    for (std::size_t si = 0; si < li.total_owned; ++si) {
        RC_ASSERT(field(si) == owned_value(si));
    }
}

// ─── Property 5: Layer-subset exactness (rank-2 field) ──────────────────────
// Feature: cpp-dycore-halo-exchange, Property 5
//
// Same property for a rank-2 (nVertLevels x nElements, LayoutLeft) field: every
// vertical level of each selected element is transferred, and excluded-layer
// halo columns remain unchanged.
//
// **Validates: Requirements 5.2, 5.3**

RC_GTEST_PROP(ExchangeIndexedProperty5, LayerSubsetExactnessRank2, ()) {
    const Layered_Indices li = generate_layered_indices();
    const auto nlev = static_cast<std::size_t>(*rc::gen::inRange(1, 5));

    halo::Communicator comm(MPI_COMM_WORLD);

    halo::Indexed_Neighbor send_nbr{g_rank, li.send_layers};
    halo::Indexed_Neighbor recv_nbr{g_rank, li.recv_layers};
    halo::Indexed_Halo_Plan plan(comm, halo::Element_Kind::cell, {send_nbr}, {recv_nbr});

    // Rank-2 view: (nVertLevels x nElements) in LayoutLeft, matching the dycore.
    Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> field("field_r2", nlev, li.n_elements);
    for (std::size_t i = 0; i < li.n_elements; ++i) {
        for (std::size_t lev = 0; lev < nlev; ++lev) {
            field(lev, i) = SENTINEL;
        }
    }
    for (std::size_t si = 0; si < li.total_owned; ++si) {
        for (std::size_t lev = 0; lev < nlev; ++lev) {
            field(lev, si) = owned_value_r2(si, lev);
        }
    }

    halo::exchange_indexed(plan, field, std::span<const int>(li.subset));

    const std::size_t num_layers = li.send_layers.size();
    for (std::size_t l = 0; l < num_layers; ++l) {
        const auto &sl = li.send_layers[l];
        const auto &rl = li.recv_layers[l];
        if (li.selected[l]) {
            for (std::size_t k = 0; k < rl.size(); ++k) {
                for (std::size_t lev = 0; lev < nlev; ++lev) {
                    RC_ASSERT(field(lev, rl[k]) == owned_value_r2(sl[k], lev));
                }
            }
        } else {
            for (std::size_t k = 0; k < rl.size(); ++k) {
                for (std::size_t lev = 0; lev < nlev; ++lev) {
                    RC_ASSERT(field(lev, rl[k]) == SENTINEL);
                }
            }
        }
    }

    for (std::size_t si = 0; si < li.total_owned; ++si) {
        for (std::size_t lev = 0; lev < nlev; ++lev) {
            RC_ASSERT(field(lev, si) == owned_value_r2(si, lev));
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
