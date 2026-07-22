// --- Property-Based Tests: halo::Indexed_Halo_Plan ---------------------------
// Feature: cpp-dycore-halo-exchange, Property 1
// Feature: cpp-dycore-halo-exchange, Property 2
//
// Uses RapidCheck to verify structural invariants of an Indexed_Halo_Plan.
//
// Property 1: Plan preserves neighbor topology
//   For any set of send and receive neighbors, each with a rank and per-layer
//   local index lists, constructing an Indexed_Halo_Plan and reading back its
//   send and receive neighbor information yields the same ranks, the same
//   per-layer index lists, and the same element kind that were supplied.
//
//   Validates: Requirements 1.1, 1.5, 5.1
//
// Property 2: Plan totals equal the sum of index lists
//   For any Indexed_Halo_Plan, the reported total send index count equals the
//   sum of the lengths of all per-neighbor, per-layer send index lists, and the
//   reported total receive index count equals the sum of the lengths of all
//   per-neighbor, per-layer receive index lists.
//
//   Validates: Requirements 1.3
// -----------------------------------------------------------------------------

#include <gtest/gtest.h>
#include <mpi.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <cstddef>
#include <numeric>
#include <vector>

#include "halo/communicator.hpp"
#include "halo/indexed_halo_plan.hpp"
#include "mpi_interposition.hpp"

// --- RapidCheck Generators ---------------------------------------------------

namespace {

/// The mock MPI_Comm_size always returns 4, so valid ranks are [0, 4).
constexpr int MOCK_COMM_SIZE = 4;

/// Generate a single layer: an ordered list of 0-based local indices.
/// Lists may be empty (Req 1.4 - empty index lists are valid), and are bounded
/// in size to keep test iterations fast.
rc::Gen<std::vector<std::size_t>> genLayer() {
    return rc::gen::exec([]() -> std::vector<std::size_t> {
        int n = *rc::gen::inRange(0, 8);
        std::vector<std::size_t> layer;
        layer.reserve(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            // Indices are arbitrary non-negative local element indices.
            layer.push_back(static_cast<std::size_t>(*rc::gen::inRange(0, 100000)));
        }
        return layer;
    });
}

/// Generate a valid Indexed_Neighbor list: unique ranks in [0, comm_size), each
/// with a per-layer set of index lists (0 to a few layers).
rc::Gen<std::vector<halo::Indexed_Neighbor>> genValidNeighborList() {
    return rc::gen::exec([]() -> std::vector<halo::Indexed_Neighbor> {
        int num_neighbors = *rc::gen::inRange(0, MOCK_COMM_SIZE + 1);

        // Pick num_neighbors unique ranks from [0, MOCK_COMM_SIZE).
        std::vector<int> available(MOCK_COMM_SIZE);
        std::iota(available.begin(), available.end(), 0);

        std::vector<halo::Indexed_Neighbor> neighbors;
        neighbors.reserve(static_cast<std::size_t>(num_neighbors));

        for (int i = 0; i < num_neighbors; ++i) {
            int idx = *rc::gen::inRange(0, static_cast<int>(available.size()));
            int rank = available[static_cast<std::size_t>(idx)];
            available.erase(available.begin() + idx);

            // Generate 0 to a few halo layers for this neighbor.
            int num_layers = *rc::gen::inRange(0, 5);
            std::vector<std::vector<std::size_t>> layers;
            layers.reserve(static_cast<std::size_t>(num_layers));
            for (int l = 0; l < num_layers; ++l) {
                layers.push_back(*genLayer());
            }

            neighbors.push_back(halo::Indexed_Neighbor{rank, std::move(layers)});
        }

        return neighbors;
    });
}

/// Generate a random element kind.
rc::Gen<halo::Element_Kind> genElementKind() {
    return rc::gen::element(halo::Element_Kind::cell, halo::Element_Kind::edge, halo::Element_Kind::vertex,
                            halo::Element_Kind::generic);
}

/// Independently sum the lengths of every per-neighbor, per-layer index list.
/// This mirrors the definition of "total index count" directly from the input
/// data, without relying on the plan's own accessors, so the property compares
/// the plan's reported totals against a from-scratch reference computation.
std::size_t sumAllIndexLengths(const std::vector<halo::Indexed_Neighbor> &neighbors) {
    std::size_t total = 0;
    for (const auto &neighbor : neighbors) {
        for (const auto &layer : neighbor.layers) {
            total += layer.size();
        }
    }
    return total;
}

/// Assert that a read-back neighbor span matches the supplied input exactly:
/// same size, same ranks in order, and same per-layer index lists.
void assertNeighborsMatch(std::span<const halo::Indexed_Neighbor> actual,
                          const std::vector<halo::Indexed_Neighbor> &expected) {
    RC_ASSERT(actual.size() == expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        RC_ASSERT(actual[i].rank == expected[i].rank);
        RC_ASSERT(actual[i].layers.size() == expected[i].layers.size());
        for (std::size_t l = 0; l < expected[i].layers.size(); ++l) {
            RC_ASSERT(actual[i].layers[l] == expected[i].layers[l]);
        }
    }
}

}  // anonymous namespace

// --- Property 1: Plan preserves neighbor topology ----------------------------
// Feature: cpp-dycore-halo-exchange, Property 1
//
// For any set of send and receive neighbors, each with a rank and per-layer
// local index lists, constructing an Indexed_Halo_Plan and reading back its
// send and receive neighbor information yields the same ranks, the same
// per-layer index lists, and the same element kind that were supplied.
//
// **Validates: Requirements 1.1, 1.5, 5.1**

RC_GTEST_PROP(IndexedHaloPlanProperty1, PreservesNeighborTopology, ()) {
    auto &spy = halo::testing::MPI_Spy::instance();
    spy.reset();

    // Generate valid send and receive neighbor lists and an element kind.
    auto send_input = *genValidNeighborList();
    auto recv_input = *genValidNeighborList();
    auto kind = *genElementKind();

    // Create a Communicator (mock MPI_Comm_size returns 4).
    halo::Communicator comm(MPI_COMM_WORLD);

    // Construct the plan with copies of the input.
    halo::Indexed_Halo_Plan plan(comm, kind, send_input, recv_input);

    // Element kind is preserved (Req 1.5).
    RC_ASSERT(plan.element_kind() == kind);

    // Send/recv neighbor ranks and per-layer index lists are preserved
    // (Req 1.1, 5.1).
    assertNeighborsMatch(plan.send_info(), send_input);
    assertNeighborsMatch(plan.recv_info(), recv_input);

    // Neighbor counts are preserved.
    RC_ASSERT(plan.num_send_neighbors() == send_input.size());
    RC_ASSERT(plan.num_recv_neighbors() == recv_input.size());
}

// --- Property 2: Plan totals equal the sum of index lists --------------------
// Feature: cpp-dycore-halo-exchange, Property 2
//
// For any Indexed_Halo_Plan, the reported total send index count equals the sum
// of the lengths of all per-neighbor, per-layer send index lists, and the
// reported total receive index count equals the sum of the lengths of all
// per-neighbor, per-layer receive index lists.
//
// **Validates: Requirements 1.3**

RC_GTEST_PROP(IndexedHaloPlanProperty2, TotalsEqualSumOfIndexLists, ()) {
    auto &spy = halo::testing::MPI_Spy::instance();
    spy.reset();

    // Generate valid send and receive neighbor lists and an element kind.
    auto send_input = *genValidNeighborList();
    auto recv_input = *genValidNeighborList();
    auto kind = *genElementKind();

    // Compute the expected totals directly from the generated input, summing the
    // lengths of every per-neighbor, per-layer index list.
    const std::size_t expected_send_total = sumAllIndexLengths(send_input);
    const std::size_t expected_recv_total = sumAllIndexLengths(recv_input);

    // Create a Communicator (mock MPI_Comm_size returns 4).
    halo::Communicator comm(MPI_COMM_WORLD);

    // Construct the plan with copies of the input.
    halo::Indexed_Halo_Plan plan(comm, kind, send_input, recv_input);

    // The plan's reported totals must equal the independently computed sums
    // (Req 1.3).
    RC_ASSERT(plan.total_send_indices() == expected_send_total);
    RC_ASSERT(plan.total_recv_indices() == expected_recv_total);
}
