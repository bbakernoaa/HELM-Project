// --- Property-Based Tests: halo::exchange_indexed (no-op behavior) -----------
// Feature: cpp-dycore-halo-exchange, Property 4
//
// Uses RapidCheck (single-rank, mocked MPI via the interposition spy) to verify
// that an indexed exchange with an empty or neighborless plan performs no work
// and leaves the field view bitwise unchanged. Because a no-op transfers zero
// elements, no real cross-rank communication is required, so the mocked-MPI
// single-process harness validates the property deterministically.
//
// Property 4: Empty or neighborless plan is a no-op
//   For any field view, invoking indexed exchange with a plan that has no send
//   and no receive neighbors leaves every element of the field view bitwise
//   unchanged. The same holds when neighbors are present but every per-layer
//   index list is empty (zero exchanged elements): no halo element is written.
//
//   Validates: Requirements 2.4, 6.3
// -----------------------------------------------------------------------------

#include <gtest/gtest.h>
#include <mpi.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <cstring>
#include <numeric>
#include <span>
#include <vector>

#include "halo/communicator.hpp"
#include "halo/exchange_indexed.hpp"
#include "halo/indexed_halo_plan.hpp"
#include "mpi_interposition.hpp"

namespace {

/// The mock MPI_Comm_size always returns 4, so valid ranks are [0, 4).
constexpr int MOCK_COMM_SIZE = 4;

/// Generate a list of arbitrary double values of the given length. Uses
/// RapidCheck's full double generator (may include zeros, negatives, subnormals,
/// and non-finite values); the no-op property is validated bitwise via memcmp,
/// so any bit pattern is a valid input.
std::vector<double> genValues(std::size_t n) {
    std::vector<double> values;
    values.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        values.push_back(*rc::gen::arbitrary<double>());
    }
    return values;
}

/// Generate a neighbor list whose ranks are unique in [0, MOCK_COMM_SIZE) and
/// whose per-layer index lists are ALL empty. Such a neighbor exchanges zero
/// elements, so the exchange must leave the field view untouched even though the
/// plan reports one or more neighbors.
std::vector<halo::Indexed_Neighbor> genEmptyIndexNeighborList() {
    int num_neighbors = *rc::gen::inRange(0, MOCK_COMM_SIZE + 1);

    std::vector<int> available(MOCK_COMM_SIZE);
    std::iota(available.begin(), available.end(), 0);

    std::vector<halo::Indexed_Neighbor> neighbors;
    neighbors.reserve(static_cast<std::size_t>(num_neighbors));

    for (int i = 0; i < num_neighbors; ++i) {
        int idx = *rc::gen::inRange(0, static_cast<int>(available.size()));
        int rank = available[static_cast<std::size_t>(idx)];
        available.erase(available.begin() + idx);

        // Zero to a few halo layers, every layer an empty index list.
        int num_layers = *rc::gen::inRange(0, 5);
        std::vector<std::vector<std::size_t>> layers(static_cast<std::size_t>(num_layers));
        neighbors.push_back(halo::Indexed_Neighbor{rank, std::move(layers)});
    }

    return neighbors;
}

/// Assert that a field view's raw bytes are identical to a saved snapshot,
/// i.e. the exchange left it bitwise unchanged.
template <typename ViewType>
void assertBitwiseUnchanged(const ViewType &view, const std::vector<double> &snapshot) {
    RC_ASSERT(view.span() == snapshot.size());
    const int cmp = std::memcmp(view.data(), snapshot.data(), snapshot.size() * sizeof(double));
    RC_ASSERT(cmp == 0);
}

}  // anonymous namespace

// --- Property 4: Empty or neighborless plan is a no-op -----------------------
// Feature: cpp-dycore-halo-exchange, Property 4
//
// For any field view, invoking indexed exchange with a plan that has no send and
// no receive neighbors leaves every element of the field view bitwise unchanged.
// The same holds when neighbors are present but every per-layer index list is
// empty (zero exchanged elements).
//
// **Validates: Requirements 2.4, 6.3**

RC_GTEST_PROP(ExchangeIndexedProperty4, EmptyOrNeighborlessPlanIsNoOp, ()) {
    auto &spy = halo::testing::MPI_Spy::instance();
    spy.reset();

    // A Communicator over the mocked world (MPI_Comm_size == 4).
    halo::Communicator comm(MPI_COMM_WORLD);

    // Randomly choose the element kind (irrelevant to the no-op behavior, but
    // exercises all kinds).
    auto kind = *rc::gen::element(halo::Element_Kind::cell, halo::Element_Kind::edge,
                                  halo::Element_Kind::vertex, halo::Element_Kind::generic);

    // Randomly choose between a truly neighborless plan and one whose neighbors
    // all carry empty index lists. Both must be no-ops.
    const bool neighborless = *rc::gen::arbitrary<bool>();
    std::vector<halo::Indexed_Neighbor> send_neighbors;
    std::vector<halo::Indexed_Neighbor> recv_neighbors;
    if (!neighborless) {
        send_neighbors = genEmptyIndexNeighborList();
        recv_neighbors = genEmptyIndexNeighborList();
    }

    halo::Indexed_Halo_Plan plan(comm, kind, send_neighbors, recv_neighbors);

    // A layer subset covering a generous range of layers, so that if any index
    // list were (erroneously) non-empty it would be selected. Order is shuffled
    // to also exercise the subset resolution path.
    const std::vector<int> layer_subset = {2, 0, 4, 1, 3};
    std::span<const int> subset_span(layer_subset);

    // Exercise both a rank-1 (nElements) and a rank-2 (nVertLevels x nElements,
    // LayoutLeft) field view.
    const bool rank2 = *rc::gen::arbitrary<bool>();

    if (!rank2) {
        const auto n = static_cast<std::size_t>(*rc::gen::inRange(1, 257));
        auto values = genValues(n);

        Kokkos::View<double *, Kokkos::HostSpace> view("indexed_r1_field", n);
        for (std::size_t i = 0; i < n; ++i) {
            view(i) = values[i];
        }

        // Snapshot the exact bytes before the exchange.
        std::vector<double> snapshot(view.data(), view.data() + view.span());

        halo::exchange_indexed(plan, view, subset_span);

        assertBitwiseUnchanged(view, snapshot);
    } else {
        const auto nlev = static_cast<std::size_t>(*rc::gen::inRange(1, 33));
        const auto nelem = static_cast<std::size_t>(*rc::gen::inRange(1, 65));
        auto values = genValues(nlev * nelem);

        Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> view("indexed_r2_field", nlev, nelem);
        // LayoutLeft: element columns are contiguous along dim 0; fill directly.
        std::memcpy(view.data(), values.data(), values.size() * sizeof(double));

        std::vector<double> snapshot(view.data(), view.data() + view.span());

        halo::exchange_indexed(plan, view, subset_span);

        assertBitwiseUnchanged(view, snapshot);
    }

    // A neighborless plan must not touch MPI at all (early return); a plan with
    // empty-index neighbors performs zero-element sends/receives but never
    // scatters into the view. In neither case is any halo element written.
    if (neighborless) {
        RC_ASSERT(spy.count_of(halo::testing::MPI_Call_Record::Type::Irecv) == 0);
        RC_ASSERT(spy.count_of(halo::testing::MPI_Call_Record::Type::Isend) == 0);
    }
}

// --- Kokkos Initialization ---------------------------------------------------
// Property tests need Kokkos initialized for View allocation.

namespace {

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

}  // namespace

static auto *const kokkos_env = ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
