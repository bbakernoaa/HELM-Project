// SPDX-License-Identifier: Apache-2.0
// Threading stress tests for span::TripleBuffer (Task 10.3)
// Verifies concurrent read/write safety of triple-buffer rotation.

#include <gtest/gtest.h>

#include <atomic>
#include <span/triple_buffer.hpp>
#include <thread>
#include <vector>

TEST(TripleBufferThreading, ConcurrentSwapRelease) {
    constexpr std::size_t N = 1000;
    std::vector<double> bufA(N, 1.0), bufB(N, 2.0), bufC(N, 3.0);
    span::TripleBuffer<double> tb(bufA.data(), bufB.data(), bufC.data(), N);

    std::atomic<int> successful_swaps{0};
    std::atomic<int> failed_swaps{0};
    constexpr int ITERATIONS = 1000;

    // Thread 1: model thread — repeatedly try to swap
    auto model_thread = [&]() {
        for (int i = 0; i < ITERATIONS; ++i) {
            if (tb.swap_buffers_for_amio()) {
                ++successful_swaps;
                // Simulate brief work then release
                std::this_thread::yield();
                tb.release_io();
            } else {
                ++failed_swaps;
            }
        }
    };

    // Thread 2: AMIO thread — also trying to swap/release
    auto amio_thread = [&]() {
        for (int i = 0; i < ITERATIONS; ++i) {
            if (tb.swap_buffers_for_amio()) {
                ++successful_swaps;
                std::this_thread::yield();
                tb.release_io();
            } else {
                ++failed_swaps;
            }
        }
    };

    std::thread t1(model_thread);
    std::thread t2(amio_thread);
    t1.join();
    t2.join();

    // All swaps were either successful or correctly rejected
    EXPECT_EQ(successful_swaps + failed_swaps, 2 * ITERATIONS);
    // At least some swaps must have succeeded
    EXPECT_GT(successful_swaps.load(), 0);
    // TripleBuffer must be in a consistent state after concurrent access
    EXPECT_TRUE(tb.write_ptr() != nullptr);
    EXPECT_TRUE(tb.read_ptr() != nullptr);
    EXPECT_TRUE(tb.io_ptr() != nullptr);
    // All three pointers distinct
    EXPECT_NE(tb.write_ptr(), tb.read_ptr());
    EXPECT_NE(tb.write_ptr(), tb.io_ptr());
    EXPECT_NE(tb.read_ptr(), tb.io_ptr());
}

TEST(TripleBufferThreading, HighContentionSwapRelease) {
    constexpr std::size_t N = 100;
    std::vector<double> bufA(N), bufB(N), bufC(N);
    span::TripleBuffer<double> tb(bufA.data(), bufB.data(), bufC.data(), N);

    constexpr int NUM_THREADS = 4;
    constexpr int ITERATIONS = 500;
    std::atomic<int> total_ops{0};

    auto worker = [&]() {
        for (int i = 0; i < ITERATIONS; ++i) {
            if (tb.swap_buffers_for_amio()) {
                tb.release_io();
            }
            ++total_ops;
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back(worker);
    }
    for (auto &t : threads) {
        t.join();
    }

    EXPECT_EQ(total_ops.load(), NUM_THREADS * ITERATIONS);
    // Invariant: pointers are still distinct
    EXPECT_NE(tb.write_ptr(), tb.read_ptr());
    EXPECT_NE(tb.write_ptr(), tb.io_ptr());
    EXPECT_NE(tb.read_ptr(), tb.io_ptr());
}
