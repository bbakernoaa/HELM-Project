// --- Property-Based Tests: halo::Halo_Plan -----------------------------------
// Feature: helm-halo-microlibrary
//
// Uses RapidCheck to verify precomputed halo exchange plan properties including
// construction round-trip, invalid rank rejection, duplicate rank rejection,
// and copy equivalence.
//
// Property 8: Halo_Plan Construction Round-Trip
//   Validates: Requirements 5.1, 5.2, 5.3
//
// Property 9: Halo_Plan Rejects Invalid Ranks
//   Validates: Requirements 5.4
//
// Property 10: Halo_Plan Rejects Duplicate Ranks
//   Validates: Requirements 5.9
//
// Property 11: Halo_Plan Copy Equivalence
//   Validates: Requirements 5.5
// -----------------------------------------------------------------------------

#include <gtest/gtest.h>
#include <mpi.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "halo/communicator.hpp"
#include "halo/halo_plan.hpp"
#include "mpi_interposition.hpp"

// --- RapidCheck Generators ---------------------------------------------------

namespace {

/// The mock MPI_Comm_size always returns 4, so valid ranks are [0, 4).
constexpr int MOCK_COMM_SIZE = 4;

/// Generate a valid Neighbor_Info list: unique ranks in [0, comm_size),
/// random counts > 0. The list length is in [0, comm_size] (at most one entry
/// per rank since duplicates are forbidden).
rc::Gen<std::vector<halo::Neighbor_Info>> genValidNeighborList() {
    return rc::gen::exec([]() -> std::vector<halo::Neighbor_Info> {
        // Generate a random subset of ranks from [0, MOCK_COMM_SIZE)
        int num_neighbors = *rc::gen::inRange(0, MOCK_COMM_SIZE + 1);

        // Create all possible ranks and shuffle to pick a random subset
        std::vector<int> all_ranks(MOCK_COMM_SIZE);
        std::iota(all_ranks.begin(), all_ranks.end(), 0);

        // Pick num_neighbors unique ranks by generating indices
        std::vector<int> available = all_ranks;
        std::vector<int> selected;
        selected.reserve(num_neighbors);

        for (int i = 0; i < num_neighbors; ++i) {
            int idx = *rc::gen::inRange(0, static_cast<int>(available.size()));
            selected.push_back(available[idx]);
            available.erase(available.begin() + idx);
        }

        // Generate random counts for each selected rank
        std::vector<halo::Neighbor_Info> neighbors;
        neighbors.reserve(selected.size());
        for (int rank : selected) {
            // Count in [1, 10000] -- reasonable buffer sizes
            std::size_t count = static_cast<std::size_t>(*rc::gen::inRange(1, 10001));
            neighbors.push_back({rank, count});
        }

        return neighbors;
    });
}

/// Generate a neighbor list that contains at least one duplicate rank.
/// Strategy:
///   1. Generate a base list of 1 to min(3, comm_size) unique valid ranks
///   2. Pick one rank from the list and insert a duplicate at a random position
rc::Gen<std::vector<halo::Neighbor_Info>> genNeighborListWithDuplicate() {
    return rc::gen::exec([]() -> std::vector<halo::Neighbor_Info> {
        // Generate 1 to min(3, MOCK_COMM_SIZE) unique ranks for the base list
        int max_unique = std::min(3, MOCK_COMM_SIZE);
        int num_unique = *rc::gen::inRange(1, max_unique + 1);

        // Generate unique ranks by picking from available pool
        std::vector<int> available(MOCK_COMM_SIZE);
        std::iota(available.begin(), available.end(), 0);

        std::vector<int> selected;
        selected.reserve(num_unique);
        for (int i = 0; i < num_unique; ++i) {
            int idx = *rc::gen::inRange(0, static_cast<int>(available.size()));
            selected.push_back(available[idx]);
            available.erase(available.begin() + idx);
        }

        // Build the base neighbor list with random counts
        std::vector<halo::Neighbor_Info> neighbors;
        neighbors.reserve(static_cast<std::size_t>(num_unique + 1));
        for (int rank : selected) {
            std::size_t count = static_cast<std::size_t>(*rc::gen::inRange(1, 10001));
            neighbors.push_back({rank, count});
        }

        // Pick a rank to duplicate
        int dup_idx = *rc::gen::inRange(0, static_cast<int>(neighbors.size()));
        int dup_rank = neighbors[static_cast<std::size_t>(dup_idx)].rank;

        // Insert the duplicate at a random position (possibly different count)
        std::size_t dup_count = static_cast<std::size_t>(*rc::gen::inRange(1, 10001));
        int insert_pos = *rc::gen::inRange(0, static_cast<int>(neighbors.size()) + 1);
        neighbors.insert(neighbors.begin() + insert_pos, {dup_rank, dup_count});

        return neighbors;
    });
}

}  // anonymous namespace

// --- Property 8: Halo_Plan Construction Round-Trip ---------------------------
// Feature: helm-halo-microlibrary, Property 8: Halo_Plan Construction Round-Trip
//
// For any valid set of send-neighbor and receive-neighbor lists (ranks in
// [0, comm_size) with no duplicates), constructing a Halo_Plan and querying
// send_info() and recv_info() SHALL return neighbor data identical to the input
// in both content and order.
//
// **Validates: Requirements 5.1, 5.2, 5.3**

RC_GTEST_PROP(HaloPlanProperty8, ConstructionRoundTrip, ()) {
    auto &spy = halo::testing::MPI_Spy::instance();
    spy.reset();

    // Generate valid send and receive neighbor lists independently
    auto send_input = *genValidNeighborList();
    auto recv_input = *genValidNeighborList();

    // Create a Communicator (mock MPI_Comm_size returns 4)
    halo::Communicator comm(MPI_COMM_WORLD);

    // Construct the Halo_Plan with copies of the input
    halo::Halo_Plan plan(comm, send_input, recv_input);

    // Verify send_info() matches input in size
    auto send_span = plan.send_info();
    RC_ASSERT(send_span.size() == send_input.size());

    // Verify send_info() matches input in content and order
    for (std::size_t i = 0; i < send_input.size(); ++i) {
        RC_ASSERT(send_span[i].rank == send_input[i].rank);
        RC_ASSERT(send_span[i].count == send_input[i].count);
    }

    // Verify recv_info() matches input in size
    auto recv_span = plan.recv_info();
    RC_ASSERT(recv_span.size() == recv_input.size());

    // Verify recv_info() matches input in content and order
    for (std::size_t i = 0; i < recv_input.size(); ++i) {
        RC_ASSERT(recv_span[i].rank == recv_input[i].rank);
        RC_ASSERT(recv_span[i].count == recv_input[i].count);
    }

    // Verify num_send_neighbors and num_recv_neighbors
    RC_ASSERT(plan.num_send_neighbors() == send_input.size());
    RC_ASSERT(plan.num_recv_neighbors() == recv_input.size());
}

// --- Property 8b: Empty neighbor lists are valid -----------------------------
// Edge case: both send and receive lists can be empty.
// This is a sub-property of Property 8 verifying the empty-list boundary.
//
// **Validates: Requirements 5.1, 5.2, 5.3**

RC_GTEST_PROP(HaloPlanProperty8, EmptyListsRoundTrip, ()) {
    auto &spy = halo::testing::MPI_Spy::instance();
    spy.reset();

    // Generate: one or both lists may be empty
    auto send_input = *genValidNeighborList();
    auto recv_input = *genValidNeighborList();

    // Force at least one to be empty for this sub-property
    bool empty_send = *rc::gen::arbitrary<bool>();
    if (empty_send) {
        send_input.clear();
    } else {
        recv_input.clear();
    }

    halo::Communicator comm(MPI_COMM_WORLD);
    halo::Halo_Plan plan(comm, send_input, recv_input);

    // Verify the empty list returns empty span
    if (empty_send) {
        RC_ASSERT(plan.send_info().empty());
        RC_ASSERT(plan.num_send_neighbors() == 0u);
    } else {
        RC_ASSERT(plan.recv_info().empty());
        RC_ASSERT(plan.num_recv_neighbors() == 0u);
    }

    // Verify the non-empty list still round-trips correctly
    if (!empty_send) {
        auto send_span = plan.send_info();
        RC_ASSERT(send_span.size() == send_input.size());
        for (std::size_t i = 0; i < send_input.size(); ++i) {
            RC_ASSERT(send_span[i].rank == send_input[i].rank);
            RC_ASSERT(send_span[i].count == send_input[i].count);
        }
    } else {
        auto recv_span = plan.recv_info();
        RC_ASSERT(recv_span.size() == recv_input.size());
        for (std::size_t i = 0; i < recv_input.size(); ++i) {
            RC_ASSERT(recv_span[i].rank == recv_input[i].rank);
            RC_ASSERT(recv_span[i].count == recv_input[i].count);
        }
    }
}

// --- Property 9: Halo_Plan Rejects Invalid Ranks -----------------------------
// Feature: helm-halo-microlibrary, Property 9: Halo_Plan Rejects Invalid Ranks
//
// For any neighbor list containing at least one rank that is negative or
// greater than or equal to the communicator size, Halo_Plan construction SHALL
// throw std::invalid_argument.
//
// **Validates: Requirements 5.4**

RC_GTEST_PROP(HaloPlanProperty9, RejectsInvalidRanks, ()) {
    auto &spy = halo::testing::MPI_Spy::instance();
    spy.reset();

    // Generate a valid neighbor list as a base
    auto valid_list = *genValidNeighborList();

    // Decide whether to inject the invalid rank into send or recv list
    bool inject_in_send = *rc::gen::arbitrary<bool>();

    // Generate at least one invalid rank: either negative or >= MOCK_COMM_SIZE
    bool use_negative = *rc::gen::arbitrary<bool>();
    int invalid_rank;
    if (use_negative) {
        // Negative rank: in range [-1000, -1]
        invalid_rank = *rc::gen::inRange(-1000, 0);
    } else {
        // Rank >= comm_size: in range [MOCK_COMM_SIZE, MOCK_COMM_SIZE + 1000]
        invalid_rank = *rc::gen::inRange(MOCK_COMM_SIZE, MOCK_COMM_SIZE + 1000);
    }

    // Generate a random count for the invalid neighbor
    std::size_t count = static_cast<std::size_t>(*rc::gen::inRange(1, 10001));

    // Build the invalid neighbor list by appending the invalid rank entry
    // to a (possibly empty) valid list prefix
    std::vector<halo::Neighbor_Info> invalid_list;
    int prefix_len = *rc::gen::inRange(0, static_cast<int>(valid_list.size()) + 1);
    for (int i = 0; i < prefix_len; ++i) {
        invalid_list.push_back(valid_list[static_cast<std::size_t>(i)]);
    }
    // Append the invalid rank entry
    invalid_list.push_back({invalid_rank, count});

    // Generate a separate valid list for the other direction
    auto other_list = *genValidNeighborList();

    // Create a Communicator (mock MPI_Comm_size returns 4)
    halo::Communicator comm(MPI_COMM_WORLD);

    // Verify that construction throws std::invalid_argument
    bool threw_invalid_argument = false;
    try {
        if (inject_in_send) {
            halo::Halo_Plan plan(comm, invalid_list, other_list);
        } else {
            halo::Halo_Plan plan(comm, other_list, invalid_list);
        }
    } catch (const std::invalid_argument &) {
        threw_invalid_argument = true;
    } catch (...) {
        RC_FAIL("Halo_Plan threw unexpected exception type (expected std::invalid_argument)");
    }

    RC_ASSERT(threw_invalid_argument);
}

// --- Property 10: Halo_Plan Rejects Duplicate Ranks --------------------------
// Feature: helm-halo-microlibrary, Property 10: Halo_Plan Rejects Duplicate Ranks
//
// For any send or receive neighbor list containing at least one rank that
// appears more than once, Halo_Plan construction SHALL throw
// std::invalid_argument indicating the duplicate.
//
// **Validates: Requirements 5.9**

RC_GTEST_PROP(HaloPlanProperty10, DuplicateInSendListThrows, ()) {
    auto &spy = halo::testing::MPI_Spy::instance();
    spy.reset();

    // Generate a send list with at least one duplicate rank
    auto send_input = *genNeighborListWithDuplicate();
    // Generate a valid (no duplicates) receive list
    auto recv_input = *genValidNeighborList();

    halo::Communicator comm(MPI_COMM_WORLD);

    // Construction must throw std::invalid_argument for duplicate send ranks
    bool threw_invalid_arg = false;
    try {
        halo::Halo_Plan plan(comm, send_input, recv_input);
    } catch (const std::invalid_argument &e) {
        threw_invalid_arg = true;
        // Verify the error message mentions "duplicate"
        std::string msg = e.what();
        RC_ASSERT(msg.find("duplicate") != std::string::npos);
    } catch (...) {
        RC_FAIL("Expected std::invalid_argument but got a different exception");
    }

    RC_ASSERT(threw_invalid_arg);
}

RC_GTEST_PROP(HaloPlanProperty10, DuplicateInRecvListThrows, ()) {
    auto &spy = halo::testing::MPI_Spy::instance();
    spy.reset();

    // Generate a valid (no duplicates) send list
    auto send_input = *genValidNeighborList();
    // Generate a receive list with at least one duplicate rank
    auto recv_input = *genNeighborListWithDuplicate();

    halo::Communicator comm(MPI_COMM_WORLD);

    // Construction must throw std::invalid_argument for duplicate recv ranks
    bool threw_invalid_arg = false;
    try {
        halo::Halo_Plan plan(comm, send_input, recv_input);
    } catch (const std::invalid_argument &e) {
        threw_invalid_arg = true;
        // Verify the error message mentions "duplicate"
        std::string msg = e.what();
        RC_ASSERT(msg.find("duplicate") != std::string::npos);
    } catch (...) {
        RC_FAIL("Expected std::invalid_argument but got a different exception");
    }

    RC_ASSERT(threw_invalid_arg);
}

RC_GTEST_PROP(HaloPlanProperty10, DuplicateInBothListsThrows, ()) {
    auto &spy = halo::testing::MPI_Spy::instance();
    spy.reset();

    // Generate both lists with duplicates
    auto send_input = *genNeighborListWithDuplicate();
    auto recv_input = *genNeighborListWithDuplicate();

    halo::Communicator comm(MPI_COMM_WORLD);

    // Construction must throw std::invalid_argument (send list validated first)
    bool threw_invalid_arg = false;
    try {
        halo::Halo_Plan plan(comm, send_input, recv_input);
    } catch (const std::invalid_argument &e) {
        threw_invalid_arg = true;
        // Verify the error message mentions "duplicate"
        std::string msg = e.what();
        RC_ASSERT(msg.find("duplicate") != std::string::npos);
    } catch (...) {
        RC_FAIL("Expected std::invalid_argument but got a different exception");
    }

    RC_ASSERT(threw_invalid_arg);
}

// --- Property 11: Halo_Plan Copy Equivalence ---------------------------------
// Feature: helm-halo-microlibrary, Property 11: Halo_Plan Copy Equivalence
//
// For any valid Halo_Plan, copy-constructing a new plan SHALL produce an object
// where send_info(), recv_info(), num_send_neighbors(), num_recv_neighbors(),
// total_send_elements(), and total_recv_elements() all return values equal to
// the original.
//
// **Validates: Requirements 5.5**

RC_GTEST_PROP(HaloPlanProperty11, CopyEquivalence, ()) {
    auto &spy = halo::testing::MPI_Spy::instance();
    spy.reset();

    // Generate valid send and receive neighbor lists
    auto send_input = *genValidNeighborList();
    auto recv_input = *genValidNeighborList();

    // Create a Communicator (mock MPI_Comm_size returns 4)
    halo::Communicator comm(MPI_COMM_WORLD);

    // Construct the original Halo_Plan
    halo::Halo_Plan original(comm, send_input, recv_input);

    // Copy-construct a new plan from the original
    halo::Halo_Plan copy(original);  // NOLINT(performance-unnecessary-copy-initialization)

    // Verify num_send_neighbors() matches
    RC_ASSERT(copy.num_send_neighbors() == original.num_send_neighbors());

    // Verify num_recv_neighbors() matches
    RC_ASSERT(copy.num_recv_neighbors() == original.num_recv_neighbors());

    // Verify total_send_elements() matches
    RC_ASSERT(copy.total_send_elements() == original.total_send_elements());

    // Verify total_recv_elements() matches
    RC_ASSERT(copy.total_recv_elements() == original.total_recv_elements());

    // Verify send_info() content matches element-by-element
    auto orig_send = original.send_info();
    auto copy_send = copy.send_info();
    RC_ASSERT(copy_send.size() == orig_send.size());
    for (std::size_t i = 0; i < orig_send.size(); ++i) {
        RC_ASSERT(copy_send[i].rank == orig_send[i].rank);
        RC_ASSERT(copy_send[i].count == orig_send[i].count);
    }

    // Verify recv_info() content matches element-by-element
    auto orig_recv = original.recv_info();
    auto copy_recv = copy.recv_info();
    RC_ASSERT(copy_recv.size() == orig_recv.size());
    for (std::size_t i = 0; i < orig_recv.size(); ++i) {
        RC_ASSERT(copy_recv[i].rank == orig_recv[i].rank);
        RC_ASSERT(copy_recv[i].count == orig_recv[i].count);
    }
}
