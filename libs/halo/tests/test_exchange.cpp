// ─── HALO End-to-End Lifecycle Test (real multi-rank MPI) ───────────────────
// Feature: helm-halo-microlibrary
//
// A genuine end-to-end exercise of the HALO public C++ API against a REAL
// multi-rank MPI job (run via `mpirun -np 4` by CTest). Unlike the single-rank
// property tests that use the MPI interposition spy, this test performs ACTUAL
// halo exchanges between distributed ranks and verifies the received data is
// byte-for-byte correct.
//
// Lifecycle covered:
//   Environment::initialize  ->  Communicator(MPI_COMM_WORLD)
//     ->  Halo_Plan (ring topology, validated neighbors)
//     ->  exchange_blocking            (verify received halo data)
//     ->  exchange_async + wait()      (verify received halo data)
//     ->  exchange_async + test() poll (verify received halo data)
//     ->  plan reuse across timesteps  (amortized RouteHandle pattern)
//     ->  Communicator split/duplicate (sub-communicator lifecycle)
//     ->  empty-plan no-op, invalid-rank rejection
//
// Topology: a periodic ring. Each rank exchanges with its left neighbor
// (rank-1) and right neighbor (rank+1), modulo comm size. Every send block is
// filled with an identical, sender-encoded pattern { rank*1000 + j }, so the
// value a receiver expects from neighbor N depends ONLY on N (not on this
// rank's position in N's neighbor list), making verification exact and simple.
//
// Each of the np ranks runs this GTest binary; an assertion failure on ANY
// rank makes that process exit non-zero, so mpirun (and thus CTest) reports a
// failure. Collective exchange calls line up because every rank runs the same
// tests in the same deterministic order.
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <mpi.h>

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "halo/communicator.hpp"
#include "halo/environment.hpp"
#include "halo/exchange.hpp"
#include "halo/halo_handle.hpp"
#include "halo/halo_plan.hpp"

namespace {

/// Number of elements exchanged per neighbor in each direction.
constexpr std::size_t kCount = 4;

/// Sentinel pre-filled into the receive region so a missing/short transfer is
/// detectable (a correct exchange overwrites every sentinel).
constexpr double kSentinel = -1.0;

using HostView = Kokkos::View<double *, Kokkos::HostSpace>;

/// The encoded value rank `sender` places at intra-block position `j`.
/// Every send block of a given rank is identical, so a receiver can predict the
/// data purely from the sender's rank.
double encoded_value(int sender, std::size_t j) {
    return static_cast<double>(sender) * 1000.0 + static_cast<double>(j);
}

/// Build the distinct ring-neighbor ranks for `rank` in a comm of `size`.
/// For size >= 3 this is {right, left} (two distinct neighbors); for size == 2
/// the two collapse to a single distinct neighbor; for size == 1 it is empty.
/// Distinctness matters because Halo_Plan rejects duplicate neighbor ranks.
std::vector<int> ring_neighbors(int rank, int size) {
    std::vector<int> neighbors;
    if (size < 2) {
        return neighbors;
    }
    const int right = (rank + 1) % size;
    const int left = (rank - 1 + size) % size;
    neighbors.push_back(right);
    if (left != right) {
        neighbors.push_back(left);
    }
    return neighbors;
}

/// Construct symmetric send/recv neighbor lists (same ranks both directions,
/// kCount elements each) for a ring exchange.
std::vector<halo::Neighbor_Info> make_neighbor_info(const std::vector<int> &ranks) {
    std::vector<halo::Neighbor_Info> info;
    info.reserve(ranks.size());
    for (int r : ranks) {
        info.push_back(halo::Neighbor_Info{r, kCount});
    }
    return info;
}

/// Fill the send region [0, K*kCount) so every per-neighbor block holds the
/// identical sender-encoded pattern, and pre-fill the receive region with the
/// sentinel. Layout matches exchange.hpp: [ send region | recv region ].
void initialize_field(HostView &field, int rank, std::size_t num_neighbors) {
    const std::size_t total_send = num_neighbors * kCount;
    for (std::size_t s = 0; s < num_neighbors; ++s) {
        for (std::size_t j = 0; j < kCount; ++j) {
            field(s * kCount + j) = encoded_value(rank, j);
        }
    }
    for (std::size_t i = total_send; i < field.extent(0); ++i) {
        field(i) = kSentinel;
    }
}

/// Verify the receive region holds exactly the data each neighbor sent.
/// The recv block for recv-neighbor i (rank N) sits at offset
/// total_send + i*kCount and must equal { encoded_value(N, j) }.
void verify_received(const HostView &field, const std::vector<int> &neighbors, std::size_t num_send_neighbors) {
    const std::size_t total_send = num_send_neighbors * kCount;
    for (std::size_t i = 0; i < neighbors.size(); ++i) {
        const int neighbor_rank = neighbors[i];
        for (std::size_t j = 0; j < kCount; ++j) {
            const double got = field(total_send + i * kCount + j);
            const double expected = encoded_value(neighbor_rank, j);
            EXPECT_DOUBLE_EQ(got, expected) << "recv block " << i << " (from rank " << neighbor_rank << ") element " << j << " mismatch";
        }
    }
}

/// Test fixture providing the current rank/size and a built ring topology.
class ExchangeLifecycle : public ::testing::Test {
   protected:
    void SetUp() override {
        comm_ = std::make_unique<halo::Communicator>(MPI_COMM_WORLD);
        rank_ = comm_->rank();
        size_ = comm_->size();
        neighbors_ = ring_neighbors(rank_, size_);
    }

    /// Allocate a field view sized to hold the send region followed by the
    /// receive region for the given symmetric neighbor count.
    HostView make_field() const {
        const std::size_t total = 2 * neighbors_.size() * kCount;
        return HostView(Kokkos::view_alloc(std::string("field"), Kokkos::WithoutInitializing), total);
    }

    std::unique_ptr<halo::Communicator> comm_;
    int rank_{0};
    int size_{0};
    std::vector<int> neighbors_;
};

// ─── Blocking ring exchange with data verification ──────────────────────────
TEST_F(ExchangeLifecycle, BlockingRingExchangeTransfersCorrectData) {
    if (size_ < 2) GTEST_SKIP() << "ring exchange needs >= 2 ranks";

    const auto info = make_neighbor_info(neighbors_);
    halo::Halo_Plan plan(*comm_, info, info);

    EXPECT_EQ(plan.num_send_neighbors(), neighbors_.size());
    EXPECT_EQ(plan.num_recv_neighbors(), neighbors_.size());
    EXPECT_EQ(plan.total_send_elements(), neighbors_.size() * kCount);

    HostView field = make_field();
    initialize_field(field, rank_, neighbors_.size());

    halo::exchange_blocking(plan, field);

    verify_received(field, neighbors_, neighbors_.size());
}

// ─── Async ring exchange completed via wait() ───────────────────────────────
TEST_F(ExchangeLifecycle, AsyncRingExchangeWaitTransfersCorrectData) {
    if (size_ < 2) GTEST_SKIP() << "ring exchange needs >= 2 ranks";

    const auto info = make_neighbor_info(neighbors_);
    halo::Halo_Plan plan(*comm_, info, info);

    HostView field = make_field();
    initialize_field(field, rank_, neighbors_.size());

    halo::Halo_Handle handle = halo::exchange_async(plan, field);
    EXPECT_FALSE(handle.empty());
    handle.wait();

    verify_received(field, neighbors_, neighbors_.size());
}

// ─── Async ring exchange completed via test() polling ───────────────────────
TEST_F(ExchangeLifecycle, AsyncRingExchangeTestPollTransfersCorrectData) {
    if (size_ < 2) GTEST_SKIP() << "ring exchange needs >= 2 ranks";

    const auto info = make_neighbor_info(neighbors_);
    halo::Halo_Plan plan(*comm_, info, info);

    HostView field = make_field();
    initialize_field(field, rank_, neighbors_.size());

    halo::Halo_Handle handle = halo::exchange_async(plan, field);

    // Poll to completion. test() returns true once all sends/receives finish
    // (and performs the post-receive deep-copy when staging is in play).
    int spins = 0;
    while (!handle.test()) {
        ++spins;
        ASSERT_LT(spins, 10'000'000) << "async exchange failed to complete";
    }

    verify_received(field, neighbors_, neighbors_.size());
}

// ─── Plan reuse across timesteps (the amortized RouteHandle pattern) ────────
TEST_F(ExchangeLifecycle, PlanReuseAcrossMultipleTimesteps) {
    if (size_ < 2) GTEST_SKIP() << "ring exchange needs >= 2 ranks";

    const auto info = make_neighbor_info(neighbors_);
    halo::Halo_Plan plan(*comm_, info, info);  // built once, reused below

    HostView field = make_field();
    for (int step = 0; step < 5; ++step) {
        initialize_field(field, rank_, neighbors_.size());
        halo::exchange_blocking(plan, field);
        verify_received(field, neighbors_, neighbors_.size());
    }
}

// ─── Empty plan is a no-op (Requirement 6.7) ────────────────────────────────
TEST_F(ExchangeLifecycle, EmptyPlanIsNoOp) {
    halo::Halo_Plan empty_plan(*comm_, {}, {});
    EXPECT_EQ(empty_plan.num_send_neighbors(), 0u);
    EXPECT_EQ(empty_plan.num_recv_neighbors(), 0u);

    HostView field("field", kCount);
    Kokkos::deep_copy(field, kSentinel);

    // Must return immediately without posting any MPI operations and without
    // touching the buffer.
    halo::exchange_blocking(empty_plan, field);
    for (std::size_t i = 0; i < field.extent(0); ++i) {
        EXPECT_DOUBLE_EQ(field(i), kSentinel);
    }

    halo::Halo_Handle handle = halo::exchange_async(empty_plan, field);
    EXPECT_TRUE(handle.empty());
    handle.wait();  // no-op
}

// ─── Communicator sub-communicator lifecycle ────────────────────────────────
TEST_F(ExchangeLifecycle, CommunicatorSplitAndDuplicate) {
    // Duplicate: same membership, independent context.
    halo::Communicator dup = comm_->duplicate();
    EXPECT_EQ(dup.size(), size_);
    EXPECT_EQ(dup.rank(), rank_);
    EXPECT_NE(dup.handle(), MPI_COMM_NULL);

    // Split into even/odd colour groups; key 0 preserves relative ranking.
    const int color = rank_ % 2;
    halo::Communicator sub = comm_->split(color, 0);
    EXPECT_NE(sub.handle(), MPI_COMM_NULL);

    // The sub-communicator's size is the count of ranks sharing this colour.
    int expected_sub_size = 0;
    for (int r = 0; r < size_; ++r) {
        if (r % 2 == color) ++expected_sub_size;
    }
    EXPECT_EQ(sub.size(), expected_sub_size);

    // A halo exchange can run on the sub-communicator independently. Each group
    // forms its own ring over its local ranks.
    const int sub_rank = sub.rank();
    const int sub_size = sub.size();
    const auto sub_neighbors = ring_neighbors(sub_rank, sub_size);
    if (sub_size >= 2) {
        const auto info = make_neighbor_info(sub_neighbors);
        halo::Halo_Plan sub_plan(sub, info, info);

        const std::size_t total = 2 * sub_neighbors.size() * kCount;
        HostView field(Kokkos::view_alloc(std::string("sub_field"), Kokkos::WithoutInitializing), total);
        initialize_field(field, sub_rank, sub_neighbors.size());
        halo::exchange_blocking(sub_plan, field);
        verify_received(field, sub_neighbors, sub_neighbors.size());
    }
}

// ─── Halo_Plan validation rejects invalid neighbor ranks ────────────────────
TEST_F(ExchangeLifecycle, PlanRejectsInvalidRank) {
    // A rank equal to comm size is out of range [0, size).
    std::vector<halo::Neighbor_Info> bad_send{halo::Neighbor_Info{size_, kCount}};
    EXPECT_THROW(halo::Halo_Plan(*comm_, bad_send, {}), std::invalid_argument);

    // A negative rank is also invalid.
    std::vector<halo::Neighbor_Info> negative{halo::Neighbor_Info{-1, kCount}};
    EXPECT_THROW(halo::Halo_Plan(*comm_, {}, negative), std::invalid_argument);

    // A duplicate rank within one direction is rejected.
    if (size_ >= 1) {
        std::vector<halo::Neighbor_Info> dup{halo::Neighbor_Info{0, kCount}, halo::Neighbor_Info{0, kCount}};
        EXPECT_THROW(halo::Halo_Plan(*comm_, dup, {}), std::invalid_argument);
    }
}

// ─── Global MPI + Kokkos + HALO environment ─────────────────────────────────
// Initializes the real MPI runtime (MPI_THREAD_MULTIPLE requested), Kokkos, and
// the HALO Environment singleton once for the whole binary, and tears them down
// in the reverse order. Registered before RUN_ALL_TESTS via gtest_main.
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
