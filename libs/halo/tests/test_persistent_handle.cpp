// ─── HALO Persistent Communication Handle Test (real multi-rank MPI) ────────
// Feature: halo-production-hardening
//
// Validates Persistent_Halo_Handle lifecycle and correctness against a REAL
// multi-rank MPI job (run via `mpirun -np 4` by CTest). Exercises:
//   1. start()/wait() delivers correct data in a ring topology
//   2. Multiple start()/wait() cycles reuse the same persistent requests
//   3. Destruction after start() (before wait()) safely frees requests
//   4. Move semantics (source becomes empty, destination works)
//
// Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5
//
// Topology: a 4-rank periodic ring where each rank sends to (rank+1)%4 and
// receives from (rank-1+4)%4. Each rank fills its send region with an encoded
// pattern { rank*1000 + j } so receivers can verify exact values.
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <mpi.h>

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <string>
#include <utility>

#include "halo/communicator.hpp"
#include "halo/environment.hpp"
#include "halo/halo_plan.hpp"
#include "halo/persistent_halo_handle.hpp"

namespace {

/// Number of elements exchanged per neighbor direction.
constexpr std::size_t kCount = 8;

/// Sentinel pre-filled into the receive region.
constexpr double kSentinel = -999.0;

using HostView = Kokkos::View<double *, Kokkos::HostSpace>;

/// Encoded value: sender rank * 1000 + intra-block position.
double encoded_value(int sender, std::size_t j) {
    return static_cast<double>(sender) * 1000.0 + static_cast<double>(j);
}

/// Fill send region with sender-encoded pattern and recv region with sentinel.
/// Layout: [send_region(kCount) | recv_region(kCount)]
void initialize_field(HostView &field, int rank) {
    // Send region: first kCount elements
    for (std::size_t j = 0; j < kCount; ++j) {
        field(j) = encoded_value(rank, j);
    }
    // Recv region: next kCount elements
    for (std::size_t j = kCount; j < 2 * kCount; ++j) {
        field(j) = kSentinel;
    }
}

/// Verify recv region contains the expected data from the sender.
void verify_received(const HostView &field, int expected_sender) {
    for (std::size_t j = 0; j < kCount; ++j) {
        const double got = field(kCount + j);
        const double expected = encoded_value(expected_sender, j);
        EXPECT_DOUBLE_EQ(got, expected) << "recv element " << j << " mismatch: expected from rank " << expected_sender;
    }
}

/// Test fixture for persistent handle tests on a 4-rank ring.
class PersistentHandleTest : public ::testing::Test {
   protected:
    void SetUp() override {
        comm_ = std::make_unique<halo::Communicator>(MPI_COMM_WORLD);
        rank_ = comm_->rank();
        size_ = comm_->size();

        // Ring topology: send to right neighbor, recv from left neighbor
        send_rank_ = (rank_ + 1) % size_;
        recv_rank_ = (rank_ - 1 + size_) % size_;
    }

    /// Create a Halo_Plan for this rank's ring topology (1 send, 1 recv).
    halo::Halo_Plan make_ring_plan() const {
        std::vector<halo::Neighbor_Info> send_info{halo::Neighbor_Info{send_rank_, kCount}};
        std::vector<halo::Neighbor_Info> recv_info{halo::Neighbor_Info{recv_rank_, kCount}};
        return halo::Halo_Plan(*comm_, send_info, recv_info);
    }

    /// Create a field view: [send_region | recv_region]
    HostView make_field() const {
        return HostView(Kokkos::view_alloc(std::string("persistent_field"), Kokkos::WithoutInitializing), 2 * kCount);
    }

    std::unique_ptr<halo::Communicator> comm_;
    int rank_{0};
    int size_{0};
    int send_rank_{0};
    int recv_rank_{0};
};

// ─── Test 1: start()/wait() delivers correct data ──────────────────────────
// Validates: Req 7.1 (binds plan to view), 7.2 (start calls Startall),
//            7.3 (wait calls Waitall)
TEST_F(PersistentHandleTest, StartWaitDeliversCorrectData) {
    if (size_ < 2) GTEST_SKIP() << "needs >= 2 ranks";

    halo::Halo_Plan plan = make_ring_plan();
    HostView field = make_field();
    initialize_field(field, rank_);

    halo::Persistent_Halo_Handle handle(plan, field);
    EXPECT_FALSE(handle.empty());

    handle.start();
    handle.wait();

    // Each rank receives from its left neighbor
    verify_received(field, recv_rank_);
}

// ─── Test 2: Multiple start()/wait() cycles reuse persistent requests ──────
// Validates: Req 7.4 (reusable across timesteps: start→wait→start→wait…)
TEST_F(PersistentHandleTest, MultipleCyclesReuseRequests) {
    if (size_ < 2) GTEST_SKIP() << "needs >= 2 ranks";

    halo::Halo_Plan plan = make_ring_plan();
    HostView field = make_field();

    // Create handle once (MPI_Send_init/Recv_init happen here)
    initialize_field(field, rank_);
    halo::Persistent_Halo_Handle handle(plan, field);

    // Repeat start/wait 3 times — persistent requests are reused each cycle
    for (int cycle = 0; cycle < 3; ++cycle) {
        // Re-fill send region with cycle-varying data to prove each exchange
        // actually transfers fresh data
        for (std::size_t j = 0; j < kCount; ++j) {
            field(j) = encoded_value(rank_, j) + cycle * 100.0;
        }
        // Clear recv region
        for (std::size_t j = kCount; j < 2 * kCount; ++j) {
            field(j) = kSentinel;
        }

        handle.start();
        handle.wait();

        // Verify recv contains the left neighbor's data for this cycle
        for (std::size_t j = 0; j < kCount; ++j) {
            const double got = field(kCount + j);
            const double expected = encoded_value(recv_rank_, j) + cycle * 100.0;
            EXPECT_DOUBLE_EQ(got, expected) << "cycle " << cycle << " recv element " << j << " mismatch";
        }
    }
}

// ─── Test 3: Destruction after start (before wait) safely frees requests ────
// Validates: Req 7.5 (RAII: destruction calls MPI_Request_free)
TEST_F(PersistentHandleTest, DestructionAfterStartIsSafe) {
    if (size_ < 2) GTEST_SKIP() << "needs >= 2 ranks";

    halo::Halo_Plan plan = make_ring_plan();
    HostView field = make_field();
    initialize_field(field, rank_);

    {
        halo::Persistent_Halo_Handle handle(plan, field);
        handle.start();
        // Destructor runs here WITHOUT calling wait().
        // MPI_Request_free should safely free the persistent requests.
        // MPI_Request_free on active persistent requests is valid per the MPI
        // standard — it marks them for deallocation once they complete.
    }

    // If we reach here without hanging or crashing, destruction was safe.
    // Barrier to ensure all ranks survived the destructor.
    MPI_Barrier(comm_->handle());
    SUCCEED();
}

// ─── Test 4: Move semantics (source becomes empty) ──────────────────────────
// Validates: Req 7.5 (move-only semantics)
TEST_F(PersistentHandleTest, MoveConstructorTransfersOwnership) {
    if (size_ < 2) GTEST_SKIP() << "needs >= 2 ranks";

    halo::Halo_Plan plan = make_ring_plan();
    HostView field = make_field();
    initialize_field(field, rank_);

    halo::Persistent_Halo_Handle original(plan, field);
    EXPECT_FALSE(original.empty());

    // Move-construct: transfers ownership
    halo::Persistent_Halo_Handle moved(std::move(original));

    // Source should now be empty
    EXPECT_TRUE(original.empty());
    // Destination should be valid
    EXPECT_FALSE(moved.empty());

    // The moved-to handle should still work correctly
    moved.start();
    moved.wait();

    verify_received(field, recv_rank_);
}

// ─── Global MPI + Kokkos + HALO environment ─────────────────────────────────
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
