// ─── HALO Tier 1 Collective Wrapper Tests (real multi-rank MPI) ─────────────
// Feature: halo-collective-primitives
//
// Example-based GoogleTest unit tests for the Tier 1 collective wrappers
// (halo::allgather / allgatherv / allreduce), exercised against a REAL MPI
// runtime (run via `mpirun -np 4` by CTest). These validate actual collective
// behavior end-to-end rather than through the single-rank MPI spy:
//
//   - Pattern A end-to-end: allgather of one int per rank, then allgatherv with
//     per-rank counts; asserts the ascending-rank concatenation, the receive
//     length (count * size), and the returned prefix-sum displacements.
//   - Zero-count allgather returns an empty buffer.
//   - allreduce MIN / MAX / SUM elementwise correctness on the host path.
//   - Invalid-argument guards: allgatherv rejects wrong-length counts and any
//     negative count with std::invalid_argument BEFORE any MPI call (so every
//     rank can assert these purely locally).
//
// Each of the np ranks runs this GTest binary; an assertion failure on ANY rank
// makes that process exit non-zero, so mpirun (and thus CTest) reports the
// failure. The collective wrappers (allgather/allgatherv/allreduce) are called
// by every rank in the same deterministic order, so they line up across ranks.
//
// Requirements: 1.1, 1.2, 1.5, 2.1, 2.3, 2.4, 2.5, 3.1, 3.2, 3.3
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <mpi.h>

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "halo/collectives.hpp"
#include "halo/communicator.hpp"
#include "halo/environment.hpp"

namespace {

/// Test fixture exposing this rank's rank/size over MPI_COMM_WORLD.
class CollectivesTest : public ::testing::Test {
   protected:
    void SetUp() override {
        comm_ = std::make_unique<halo::Communicator>(MPI_COMM_WORLD);
        rank_ = comm_->rank();
        size_ = comm_->size();
    }

    std::unique_ptr<halo::Communicator> comm_;
    int rank_{0};
    int size_{0};
};

// ─── Pattern A step 1: allgather of one int per rank (Req 1.1, 1.2) ─────────
// Each rank sends its own rank value as a single int; after the gather every
// rank must hold {0, 1, 2, ..., size-1} in ascending rank order, length == size.
TEST_F(CollectivesTest, AllgatherOneIntPerRankConcatenatesInRankOrder) {
    const int my_rank = rank_;
    std::vector<int> gathered = halo::allgather<int>(*comm_, &my_rank, 1);

    ASSERT_EQ(gathered.size(), static_cast<std::size_t>(size_));  // Req 1.2: count * size
    for (int r = 0; r < size_; ++r) {
        EXPECT_EQ(gathered[static_cast<std::size_t>(r)], r)  // Req 1.1: ascending-rank order
            << "gathered[" << r << "] should equal rank " << r;
    }
}

// ─── Pattern A step 2: allgatherv with per-rank counts (Req 2.1, 2.2, 2.3) ──
// counts[r] = r + 1 (rank r contributes r+1 elements). Each rank sends a vector
// of (my_rank + 1) copies of its own rank. The gathered buffer must have length
// sum(counts), the returned displacements must be the prefix sum of counts, and
// the concatenation content must place rank r's block (all value r) at its
// displacement.
TEST_F(CollectivesTest, AllgathervVariableCountsConcatenatesWithPrefixSumDisplacements) {
    // Per-rank counts: rank r contributes r + 1 elements.
    std::vector<int> counts(static_cast<std::size_t>(size_));
    for (int r = 0; r < size_; ++r) {
        counts[static_cast<std::size_t>(r)] = r + 1;
    }

    // This rank's contribution: (my_rank + 1) copies of my_rank.
    const int my_count = counts[static_cast<std::size_t>(rank_)];
    std::vector<int> send(static_cast<std::size_t>(my_count), rank_);

    halo::Allgather_Result<int> result = halo::allgatherv<int>(*comm_, send, counts);

    // Expected total length is the sum of all per-rank counts.
    const int total = std::accumulate(counts.begin(), counts.end(), 0);
    ASSERT_EQ(result.buffer.size(), static_cast<std::size_t>(total));

    // Returned displacements are the prefix sum: displ[0] = 0,
    // displ[k] = sum(counts[0..k-1]) (Req 2.1, 2.3).
    ASSERT_EQ(result.displacements.size(), static_cast<std::size_t>(size_));
    int running = 0;
    for (int r = 0; r < size_; ++r) {
        EXPECT_EQ(result.displacements[static_cast<std::size_t>(r)], running) << "displacement for rank " << r << " should be the prefix sum";
        running += counts[static_cast<std::size_t>(r)];
    }

    // Concatenation content: rank r's block occupies [displ[r], displ[r]+counts[r])
    // and every element in it equals r.
    for (int r = 0; r < size_; ++r) {
        const int base = result.displacements[static_cast<std::size_t>(r)];
        for (int j = 0; j < counts[static_cast<std::size_t>(r)]; ++j) {
            EXPECT_EQ(result.buffer[static_cast<std::size_t>(base + j)], r)
                << "buffer element " << (base + j) << " (rank " << r << " block) should equal " << r;
        }
    }
}

// ─── Zero-count allgather returns an empty buffer (Req 1.5) ─────────────────
// The zero-count fast path returns empty WITHOUT calling MPI_Allgather, so no
// collective is issued and every rank can assert locally.
TEST_F(CollectivesTest, AllgatherZeroCountReturnsEmpty) {
    std::vector<int> gathered = halo::allgather<int>(*comm_, nullptr, 0);
    EXPECT_TRUE(gathered.empty());

    // The vector overload of an empty send buffer also yields an empty result.
    std::vector<int> empty_send;
    std::vector<int> gathered_vec = halo::allgather<int>(*comm_, empty_send);
    EXPECT_TRUE(gathered_vec.empty());
}

// ─── allreduce SUM elementwise correctness (Req 3.1, 3.2, 3.3) ──────────────
// Each rank supplies {my_rank, my_rank + 10}. The SUM reduction yields
// {sum(0..size-1), sum(0..size-1) + 10*size} on every rank.
TEST_F(CollectivesTest, AllreduceSumIsElementwiseSum) {
    std::vector<int> input{rank_, rank_ + 10};
    std::vector<int> reduced = halo::allreduce<int>(*comm_, input, MPI_SUM);

    ASSERT_EQ(reduced.size(), 2u);  // Req 3.2: same length as input

    // sum(0..size-1) = size * (size - 1) / 2
    const int rank_sum = size_ * (size_ - 1) / 2;
    EXPECT_EQ(reduced[0], rank_sum);
    EXPECT_EQ(reduced[1], rank_sum + 10 * size_);
}

// ─── allreduce MIN elementwise correctness (Req 3.1, 3.2, 3.3) ──────────────
// With input {my_rank, my_rank + 10}, the elementwise MIN across ranks is
// {min rank = 0, min (rank+10) = 10}.
TEST_F(CollectivesTest, AllreduceMinIsElementwiseMin) {
    std::vector<int> input{rank_, rank_ + 10};
    std::vector<int> reduced = halo::allreduce<int>(*comm_, input, MPI_MIN);

    ASSERT_EQ(reduced.size(), 2u);
    EXPECT_EQ(reduced[0], 0);
    EXPECT_EQ(reduced[1], 10);
}

// ─── allreduce MAX elementwise correctness (Req 3.1, 3.2, 3.3) ──────────────
// With input {my_rank, my_rank + 10}, the elementwise MAX across ranks is
// {max rank = size-1, max (rank+10) = size-1+10}.
TEST_F(CollectivesTest, AllreduceMaxIsElementwiseMax) {
    std::vector<int> input{rank_, rank_ + 10};
    std::vector<int> reduced = halo::allreduce<int>(*comm_, input, MPI_MAX);

    ASSERT_EQ(reduced.size(), 2u);
    EXPECT_EQ(reduced[0], size_ - 1);
    EXPECT_EQ(reduced[1], size_ - 1 + 10);
}

// ─── allgatherv rejects wrong-length counts before any MPI (Req 2.5) ────────
// A counts array whose length differs from comm.size() is an invalid argument;
// detail::validate_counts throws std::invalid_argument BEFORE any MPI call, so
// this is safe to assert on every rank without lockstep participation.
TEST_F(CollectivesTest, AllgathervWrongLengthCountsThrows) {
    // Deliberately the wrong length (size + 1).
    std::vector<int> counts_wrong_length(static_cast<std::size_t>(size_) + 1, 1);
    std::vector<int> send{rank_};

    EXPECT_THROW(halo::allgatherv<int>(*comm_, send.data(), counts_wrong_length), std::invalid_argument);
}

// ─── allgatherv rejects a negative count before any MPI (Req 2.4) ───────────
// A correct-length counts array with a negative entry is an invalid argument;
// the guard throws std::invalid_argument BEFORE any MPI call.
TEST_F(CollectivesTest, AllgathervNegativeCountThrows) {
    std::vector<int> counts(static_cast<std::size_t>(size_), 1);
    counts[0] = -1;  // negative entry for rank 0
    std::vector<int> send{rank_};

    EXPECT_THROW(halo::allgatherv<int>(*comm_, send.data(), counts), std::invalid_argument);
}

// ─── Global MPI + Kokkos + HALO environment ─────────────────────────────────
// Initializes the real MPI runtime (MPI_THREAD_MULTIPLE requested), Kokkos, and
// the HALO Environment singleton once for the whole binary, tearing them down in
// reverse order. Registered before RUN_ALL_TESTS via gtest_main. Mirrors the
// setup used by test_exchange.cpp.
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
