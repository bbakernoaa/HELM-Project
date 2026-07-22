// ─── HALO Diagnostics Instrumentation Tests (real multi-rank MPI) ────────────
// Feature: halo-production-hardening
//
// Validates the Diagnostics callback hook interface (Requirement 10.1–10.4):
//   - Registered callback receives exchange events with correct fields.
//   - No callback registered → zero overhead (is_active() returns false).
//   - Thread safety: set_callback / emit under concurrent access.
//   - clear_callback stops event delivery.
//
// Run with: mpirun -np 4 ./test_diagnostics
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <mpi.h>

#include <Kokkos_Core.hpp>
#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "halo/communicator.hpp"
#include "halo/diagnostics.hpp"
#include "halo/environment.hpp"
#include "halo/exchange.hpp"
#include "halo/halo_plan.hpp"

namespace {

/// Number of elements exchanged per neighbor.
constexpr std::size_t kCount = 8;

using HostView = Kokkos::View<double *, Kokkos::HostSpace>;

/// Build a periodic ring neighbor list for `rank` in a comm of `size`.
std::vector<int> ring_neighbors(int rank, int size) {
    std::vector<int> neighbors;
    if (size < 2) return neighbors;
    const int right = (rank + 1) % size;
    const int left = (rank - 1 + size) % size;
    neighbors.push_back(right);
    if (left != right) neighbors.push_back(left);
    return neighbors;
}

/// Construct symmetric neighbor info for ring topology.
std::vector<halo::Neighbor_Info> make_neighbor_info(const std::vector<int> &ranks) {
    std::vector<halo::Neighbor_Info> info;
    info.reserve(ranks.size());
    for (int r : ranks) {
        info.push_back(halo::Neighbor_Info{r, kCount});
    }
    return info;
}

/// Allocate and initialize a field for ring exchange.
HostView make_initialized_field(int rank, std::size_t num_neighbors) {
    const std::size_t total = 2 * num_neighbors * kCount;
    HostView field(Kokkos::view_alloc(std::string("diag_field"), Kokkos::WithoutInitializing), total);
    // Fill send region with sender-encoded values.
    const std::size_t total_send = num_neighbors * kCount;
    for (std::size_t s = 0; s < num_neighbors; ++s) {
        for (std::size_t j = 0; j < kCount; ++j) {
            field(s * kCount + j) = static_cast<double>(rank * 1000) + static_cast<double>(j);
        }
    }
    // Fill recv region with sentinel.
    for (std::size_t i = total_send; i < field.extent(0); ++i) {
        field(i) = -1.0;
    }
    return field;
}

// ─── Test fixture ───────────────────────────────────────────────────────────

class DiagnosticsTest : public ::testing::Test {
   protected:
    void SetUp() override {
        comm_ = std::make_unique<halo::Communicator>(MPI_COMM_WORLD);
        rank_ = comm_->rank();
        size_ = comm_->size();
        neighbors_ = ring_neighbors(rank_, size_);
        // Ensure clean state: no callback registered from previous tests.
        halo::Diagnostics::clear_callback();
    }

    void TearDown() override {
        halo::Diagnostics::clear_callback();
    }

    std::unique_ptr<halo::Communicator> comm_;
    int rank_{0};
    int size_{0};
    std::vector<int> neighbors_;
};

// ─── Test 1: Callback receives events with correct fields ───────────────────
// Validates: Requirements 10.1, 10.2
TEST_F(DiagnosticsTest, CallbackReceivesEventsWithCorrectFields) {
    if (size_ < 2) GTEST_SKIP() << "needs >= 2 ranks";

    // Collect events in a vector (thread-safe since exchange is blocking).
    std::vector<halo::Exchange_Event> events;
    std::mutex events_mutex;

    halo::Diagnostics::set_callback([&events, &events_mutex](const halo::Exchange_Event &ev) {
        std::lock_guard<std::mutex> lock(events_mutex);
        events.push_back(ev);
    });

    ASSERT_TRUE(halo::Diagnostics::is_active());

    const auto info = make_neighbor_info(neighbors_);
    halo::Halo_Plan plan(*comm_, info, info);
    HostView field = make_initialized_field(rank_, neighbors_.size());

    halo::exchange_blocking(plan, field);

    // Verify: at least begin and end events were received.
    ASSERT_GE(events.size(), 2u);

    // Find begin and end events.
    const halo::Exchange_Event *begin_ev = nullptr;
    const halo::Exchange_Event *end_ev = nullptr;
    for (const auto &ev : events) {
        if (ev.phase == halo::Exchange_Event::Phase::begin) begin_ev = &ev;
        if (ev.phase == halo::Exchange_Event::Phase::end) end_ev = &ev;
    }

    ASSERT_NE(begin_ev, nullptr) << "no begin event received";
    ASSERT_NE(end_ev, nullptr) << "no end event received";

    // Verify local_rank matches comm.rank().
    EXPECT_EQ(begin_ev->local_rank, rank_);
    EXPECT_EQ(end_ev->local_rank, rank_);

    // Verify total_bytes > 0.
    EXPECT_GT(begin_ev->total_bytes, 0u);
    EXPECT_GT(end_ev->total_bytes, 0u);

    // Verify elapsed for end event > 0 nanoseconds.
    EXPECT_GT(end_ev->elapsed.count(), 0);

    // Verify elapsed for begin event == 0 nanoseconds.
    EXPECT_EQ(begin_ev->elapsed.count(), 0);

    // Verify is_async == false for blocking exchange.
    EXPECT_FALSE(begin_ev->is_async);
    EXPECT_FALSE(end_ev->is_async);

    // Verify neighbor_count is correct.
    const int expected_neighbors = static_cast<int>(neighbors_.size() + neighbors_.size());  // send + recv neighbors
    EXPECT_EQ(begin_ev->neighbor_count, expected_neighbors);
    EXPECT_EQ(end_ev->neighbor_count, expected_neighbors);
}

// ─── Test 2: No callback = no overhead ──────────────────────────────────────
// Validates: Requirement 10.3
TEST_F(DiagnosticsTest, NoCallbackRegisteredMeansNoOverhead) {
    // Ensure no callback is registered.
    halo::Diagnostics::clear_callback();
    EXPECT_FALSE(halo::Diagnostics::is_active());

    if (size_ < 2) GTEST_SKIP() << "needs >= 2 ranks";

    // Perform an exchange without a callback — Diagnostics::is_active()
    // returns false and emit() short-circuits without timing calls.
    const auto info = make_neighbor_info(neighbors_);
    halo::Halo_Plan plan(*comm_, info, info);
    HostView field = make_initialized_field(rank_, neighbors_.size());

    // Timing sanity: the exchange should still work correctly.
    halo::exchange_blocking(plan, field);

    // is_active() should still be false after the exchange.
    EXPECT_FALSE(halo::Diagnostics::is_active());
}

// ─── Test 3: Thread safety of set_callback / emit under concurrency ─────────
// Validates: Requirement 10.1 (thread-safe hook)
TEST_F(DiagnosticsTest, ThreadSafetyOfSetCallbackAndEmit) {
    // This test verifies the mutex protection doesn't deadlock or crash
    // when set_callback, clear_callback, and emit are called concurrently.
    std::atomic<int> callback_count{0};
    std::atomic<bool> stop{false};

    // Launch a thread that continuously toggles the callback on/off.
    std::thread toggler([&stop, &callback_count]() {
        while (!stop.load(std::memory_order_relaxed)) {
            halo::Diagnostics::set_callback(
                [&callback_count](const halo::Exchange_Event & /*ev*/) { callback_count.fetch_add(1, std::memory_order_relaxed); });
            std::this_thread::yield();
            halo::Diagnostics::clear_callback();
            std::this_thread::yield();
        }
    });

    // Launch a thread that continuously emits events.
    std::thread emitter([&stop]() {
        halo::Exchange_Event ev{halo::Exchange_Event::Phase::begin, 0, 2, 1024, std::chrono::nanoseconds{0}, false};
        while (!stop.load(std::memory_order_relaxed)) {
            halo::Diagnostics::emit(ev);
            std::this_thread::yield();
        }
    });

    // Let them race for a short time.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    stop.store(true, std::memory_order_relaxed);

    toggler.join();
    emitter.join();

    // If we reach here without deadlock or crash, the test passes.
    // callback_count may be > 0 depending on timing — that's fine.
    SUCCEED() << "No deadlock or crash under concurrent set_callback/emit";
}

// ─── Test 4: Clear callback stops events ────────────────────────────────────
// Validates: Requirement 10.3
TEST_F(DiagnosticsTest, ClearCallbackStopsEventDelivery) {
    if (size_ < 2) GTEST_SKIP() << "needs >= 2 ranks";

    std::atomic<int> event_count{0};

    // Register, then immediately clear.
    halo::Diagnostics::set_callback([&event_count](const halo::Exchange_Event & /*ev*/) { event_count.fetch_add(1, std::memory_order_relaxed); });
    ASSERT_TRUE(halo::Diagnostics::is_active());

    halo::Diagnostics::clear_callback();
    ASSERT_FALSE(halo::Diagnostics::is_active());

    // Perform an exchange — no events should be delivered.
    const auto info = make_neighbor_info(neighbors_);
    halo::Halo_Plan plan(*comm_, info, info);
    HostView field = make_initialized_field(rank_, neighbors_.size());

    halo::exchange_blocking(plan, field);

    EXPECT_EQ(event_count.load(), 0) << "events received after clear_callback()";
}

// ─── Test 5: Async exchange emits events with is_async=true ─────────────────
// Validates: Requirement 10.2
TEST_F(DiagnosticsTest, AsyncExchangeEmitsEventsWithIsAsyncTrue) {
    if (size_ < 2) GTEST_SKIP() << "needs >= 2 ranks";

    std::vector<halo::Exchange_Event> events;
    std::mutex events_mutex;

    halo::Diagnostics::set_callback([&events, &events_mutex](const halo::Exchange_Event &ev) {
        std::lock_guard<std::mutex> lock(events_mutex);
        events.push_back(ev);
    });

    const auto info = make_neighbor_info(neighbors_);
    halo::Halo_Plan plan(*comm_, info, info);
    HostView field = make_initialized_field(rank_, neighbors_.size());

    auto handle = halo::exchange_async(plan, field);
    handle.wait();

    // Find the begin event and verify is_async == true.
    const halo::Exchange_Event *begin_ev = nullptr;
    for (const auto &ev : events) {
        if (ev.phase == halo::Exchange_Event::Phase::begin) {
            begin_ev = &ev;
            break;
        }
    }
    ASSERT_NE(begin_ev, nullptr) << "no begin event for async exchange";
    EXPECT_TRUE(begin_ev->is_async);
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
        halo::Diagnostics::clear_callback();
        Kokkos::finalize();
        MPI_Finalize();
    }
};

}  // namespace

// Register the environment (gtest_main provides main()).
static ::testing::Environment *const halo_mpi_env = ::testing::AddGlobalTestEnvironment(new HaloMpiEnvironment);
