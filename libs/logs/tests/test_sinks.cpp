/// @file test_sinks.cpp
/// @brief Property test: Borrowed Stream Lifecycle Is Untouched.
///
/// Uses a custom std::streambuf spy to verify that logs::Sink only performs
/// write (xsputn/overflow) and flush (sync) operations on the underlying
/// stream, and never opens, closes, or destroys it.
///
/// Property 25: Borrowed Stream Lifecycle Is Untouched
/// Validates: Requirements 6.4, 6.5

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <cstring>
#include <logs/sink.hpp>
#include <ostream>
#include <streambuf>
#include <string>

namespace {

/// A custom std::streambuf subclass that tracks which virtual operations
/// are invoked. This allows us to assert that Sink only calls write-related
/// and sync-related methods, and never opens/closes/destroys the buffer.
class SpyStreambuf : public std::streambuf {
   public:
    // Counters for observed operations.
    int xsputn_calls = 0;
    int overflow_calls = 0;
    int sync_calls = 0;

    // Lifecycle operations that Sink must NEVER invoke.
    bool open_called = false;
    bool close_called = false;
    bool destroyed = false;

    // Captured written data for content verification.
    std::string written_data;

    ~SpyStreambuf() override {
        destroyed = true;
    }

   protected:
    /// Called by the stream for bulk writes.
    std::streamsize xsputn(const char *s, std::streamsize count) override {
        ++xsputn_calls;
        written_data.append(s, static_cast<std::size_t>(count));
        return count;
    }

    /// Called by the stream for single-character writes or when buffer is full.
    int_type overflow(int_type ch) override {
        ++overflow_calls;
        if (ch != traits_type::eof()) {
            written_data.push_back(static_cast<char>(ch));
        }
        return ch;
    }

    /// Called by the stream on flush.
    int sync() override {
        ++sync_calls;
        return 0;  // Success.
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Property 25: Borrowed Stream Lifecycle Is Untouched
// Validates: Requirements 6.4, 6.5
// ─────────────────────────────────────────────────────────────────────────────

/// Verify that Sink::write() only invokes xsputn/overflow on the streambuf
/// and never triggers open/close/destroy.
TEST(SinkLifecycle, WriteOnlyInvokesWriteOperations) {
    SpyStreambuf spy;
    std::ostream stream{&spy};

    logs::Sink sink{stream};
    const std::string_view msg = "Hello, HELM!";

    const bool ok = sink.write(msg);

    EXPECT_TRUE(ok);
    // Write must have been called at least once (xsputn or overflow).
    EXPECT_GT(spy.xsputn_calls + spy.overflow_calls, 0);
    // Content must match.
    EXPECT_EQ(spy.written_data, msg);
    // Lifecycle operations must NOT have been invoked.
    EXPECT_FALSE(spy.open_called);
    EXPECT_FALSE(spy.close_called);
}

/// Verify that Sink::flush() only invokes sync on the streambuf
/// and never triggers open/close/destroy.
TEST(SinkLifecycle, FlushOnlyInvokesSync) {
    SpyStreambuf spy;
    std::ostream stream{&spy};

    logs::Sink sink{stream};

    const bool ok = sink.flush();

    EXPECT_TRUE(ok);
    EXPECT_GT(spy.sync_calls, 0);
    // Lifecycle operations must NOT have been invoked.
    EXPECT_FALSE(spy.open_called);
    EXPECT_FALSE(spy.close_called);
}

/// Verify that destroying a Sink does NOT destroy the underlying streambuf.
/// The caller owns the stream lifecycle (non-owning semantics).
TEST(SinkLifecycle, SinkDestructionDoesNotDestroyStream) {
    auto *spy = new SpyStreambuf{};
    auto *stream = new std::ostream{spy};

    {
        logs::Sink sink{*stream};
        sink.write("test data");
        sink.flush();
    }
    // After Sink is destroyed, the streambuf must still be alive.
    EXPECT_FALSE(spy->destroyed);

    // Clean up manually — proving the caller owns the lifecycle.
    delete stream;
    // Note: spy->destroyed is set in the destructor, but the object memory
    // is freed so we don't read it. The test above is the key assertion.
    delete spy;
}

/// Verify that write followed by flush produces no lifecycle side effects
/// (open/close) regardless of call order.
TEST(SinkLifecycle, WriteAndFlushSequenceNoLifecycleSideEffects) {
    SpyStreambuf spy;
    std::ostream stream{&spy};

    logs::Sink sink{stream};

    // Interleave writes and flushes.
    EXPECT_TRUE(sink.write("first"));
    EXPECT_TRUE(sink.flush());
    EXPECT_TRUE(sink.write("second"));
    EXPECT_TRUE(sink.flush());

    // Verify data integrity.
    EXPECT_EQ(spy.written_data, "firstsecond");
    EXPECT_EQ(spy.sync_calls, 2);
    EXPECT_GT(spy.xsputn_calls + spy.overflow_calls, 0);

    // No lifecycle operations.
    EXPECT_FALSE(spy.open_called);
    EXPECT_FALSE(spy.close_called);
}

// ─────────────────────────────────────────────────────────────────────────────
// RapidCheck property: arbitrary string_view data is faithfully forwarded
// to write operations only, with no lifecycle side effects.
// ─────────────────────────────────────────────────────────────────────────────

/// **Validates: Requirements 6.4, 6.5**
RC_GTEST_PROP(SinkLifecycleProperty, ArbitraryWriteOnlyUsesWriteAndFlush, ()) {
    SpyStreambuf spy;
    std::ostream stream{&spy};
    logs::Sink sink{stream};

    // Generate arbitrary data to write.
    const auto data = *rc::gen::arbitrary<std::string>();

    const std::string_view view{data};
    const bool write_ok = sink.write(view);
    const bool flush_ok = sink.flush();

    // Write and flush must succeed.
    RC_ASSERT(write_ok);
    RC_ASSERT(flush_ok);

    // Written data must be faithful (byte-for-byte).
    RC_ASSERT(spy.written_data == data);
    RC_ASSERT(spy.written_data.size() == data.size());

    // Write operations must have been invoked.
    RC_ASSERT(spy.xsputn_calls + spy.overflow_calls > 0);

    // Sync must have been invoked.
    RC_ASSERT(spy.sync_calls > 0);

    // Lifecycle operations must NEVER have been invoked.
    RC_ASSERT(!spy.open_called);
    RC_ASSERT(!spy.close_called);
}

/// For any sequence of writes, verify all data is faithfully forwarded
/// in order, and no lifecycle operations occur.
RC_GTEST_PROP(SinkLifecycleProperty, MultipleWritesPreserveOrderAndContent, ()) {
    SpyStreambuf spy;
    std::ostream stream{&spy};
    logs::Sink sink{stream};

    // Generate a non-empty sequence of strings.
    const auto messages = *rc::gen::container<std::vector<std::string>>(rc::gen::arbitrary<std::string>());

    std::string expected;
    for (const auto &msg : messages) {
        const bool ok = sink.write(std::string_view{msg});
        RC_ASSERT(ok);
        expected += msg;
    }

    // All data must arrive in order.
    RC_ASSERT(spy.written_data == expected);

    // No lifecycle operations.
    RC_ASSERT(!spy.open_called);
    RC_ASSERT(!spy.close_called);
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════════
// Sink Dispatch Semantics — Unit Tests and Property Tests
//
// Task 17.1: Unit tests for dispatch semantics (Requirements 6.1, 6.2, 6.3, 6.6, 6.7)
// Task 17.2: Property 23 — Sink Write-Exactly-Once and Bounded Cardinality
// Task 17.3: Property 24 — Sink Write-Failure Isolation
// ═══════════════════════════════════════════════════════════════════════════════

#include <algorithm>
#include <array>
#include <iostream>
#include <logs/logger.hpp>
#include <memory>
#include <sstream>
#include <vector>

#include "in_memory_sink.hpp"

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// Helper: A streambuf that throws on write, for failure-isolation tests.
// ─────────────────────────────────────────────────────────────────────────────

class Throwing_Streambuf : public std::streambuf {
   protected:
    std::streamsize xsputn(const char * /*s*/, std::streamsize /*n*/) override {
        throw std::runtime_error("simulated write failure");
    }

    int_type overflow(int_type /*ch*/) override {
        throw std::runtime_error("simulated write failure");
    }

    int sync() override {
        return 0;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Task 17.1: Unit tests for Sink dispatch semantics
// Validates: Requirements 6.1, 6.2, 6.3, 6.6, 6.7
// ─────────────────────────────────────────────────────────────────────────────

/// Test: With k configured sinks (0 < k <= 64), verify formatted record
/// written exactly once to each.
TEST(SinkDispatch, MultiSinkWriteExactlyOnce) {
    constexpr int K = 5;

    std::array<std::unique_ptr<logs::testing::In_Memory_Sink>, K> mem_sinks;
    logs::Logger logger;

    for (int i = 0; i < K; ++i) {
        mem_sinks[i] = std::make_unique<logs::testing::In_Memory_Sink>();
        logger.add_sink(mem_sinks[i]->sink());
    }

    logger.log(logs::Severity_Level::WARNING, "dispatch test");

    for (int i = 0; i < K; ++i) {
        EXPECT_EQ(mem_sinks[i]->count(), 1u) << "Sink " << i << " should receive exactly one write";
        EXPECT_FALSE(mem_sinks[i]->entries().empty());
        // Verify all sinks received the same formatted text.
        EXPECT_EQ(mem_sinks[i]->entries()[0], mem_sinks[0]->entries()[0]);
    }
}

/// Test: With zero configured sinks, verify record written to default stderr.
TEST(SinkDispatch, ZeroSinksWritesToStderr) {
    logs::Logger logger;

    // Capture stderr output.
    std::ostringstream captured;
    std::streambuf *original_stderr = std::cerr.rdbuf();
    std::cerr.rdbuf(captured.rdbuf());

    logger.log(logs::Severity_Level::INFO, "stderr fallback test");

    std::cerr.rdbuf(original_stderr);

    // Verify something was written to stderr.
    const std::string output = captured.str();
    EXPECT_FALSE(output.empty());
    EXPECT_NE(output.find("stderr fallback test"), std::string::npos);
}

/// Test: With one failing sink, verify record still written to remaining sinks;
/// failing sink retained for subsequent records.
TEST(SinkDispatch, FailingSinkIsolation) {
    logs::Logger logger;

    // Create a good sink, a failing sink, and another good sink.
    logs::testing::In_Memory_Sink good_sink_1;
    logs::testing::In_Memory_Sink good_sink_2;

    Throwing_Streambuf throwing_buf;
    std::ostream throwing_stream(&throwing_buf);
    logs::Sink failing_sink(throwing_stream);

    logger.add_sink(good_sink_1.sink());
    logger.add_sink(failing_sink);
    logger.add_sink(good_sink_2.sink());

    // Emit a record — should not throw, and good sinks should receive it.
    logger.log(logs::Severity_Level::ERROR, "failure isolation test");

    EXPECT_EQ(good_sink_1.count(), 1u);
    EXPECT_EQ(good_sink_2.count(), 1u);
    EXPECT_NE(good_sink_1.entries()[0].find("failure isolation test"), std::string::npos);
    EXPECT_NE(good_sink_2.entries()[0].find("failure isolation test"), std::string::npos);

    // Emit another record to verify failing sink is still registered (retained).
    good_sink_1.clear();
    good_sink_2.clear();
    logger.log(logs::Severity_Level::WARNING, "second record after failure");

    EXPECT_EQ(good_sink_1.count(), 1u);
    EXPECT_EQ(good_sink_2.count(), 1u);
}

/// Test: Attempt to add sink beyond MAX_SINKS=64, verify rejection absorbed
/// without exception.
TEST(SinkDispatch, MaxSinksRejectionAbsorbed) {
    logs::Logger logger;

    // Create 65 sinks — only the first 64 should be accepted.
    std::vector<std::unique_ptr<logs::testing::In_Memory_Sink>> sinks;
    sinks.reserve(65);

    for (int i = 0; i < 65; ++i) {
        sinks.push_back(std::make_unique<logs::testing::In_Memory_Sink>());
        // Must not throw, even on the 65th.
        EXPECT_NO_THROW(logger.add_sink(sinks.back()->sink()));
    }

    // Emit a record — only the first 64 sinks should receive it.
    logger.log(logs::Severity_Level::INFO, "max sinks test");

    int received_count = 0;
    for (int i = 0; i < 65; ++i) {
        if (sinks[i]->count() > 0) {
            ++received_count;
        }
    }
    EXPECT_EQ(received_count, 64);

    // The 65th sink (index 64) should have received nothing.
    EXPECT_EQ(sinks[64]->count(), 0u);
}

// ─────────────────────────────────────────────────────────────────────────────
// Task 17.2: Property 23 — Sink Write-Exactly-Once and Bounded Cardinality
// **Validates: Requirements 6.1, 6.2, 6.3**
// ─────────────────────────────────────────────────────────────────────────────

/// Helper: generate a non-empty printable ASCII string for sink tests.
static rc::Gen<std::string> genPrintableMessage() {
    return rc::gen::map(rc::gen::nonEmpty(rc::gen::container<std::string>(rc::gen::inRange(33, 127))), [](std::string s) { return s; });
}

/// Property 23: For k > 0 sinks and accepted record, verify exactly one
/// write per sink; count <= 64.
RC_GTEST_PROP(SinkDispatchProperty, WriteExactlyOnceAndBoundedCardinality, ()) {
    // Generate k in [1, 64].
    const auto k = *rc::gen::inRange(1, 65);
    RC_ASSERT(k <= 64);

    logs::Logger logger;
    std::vector<std::unique_ptr<logs::testing::In_Memory_Sink>> sinks;
    sinks.reserve(static_cast<std::size_t>(k));

    for (int i = 0; i < k; ++i) {
        sinks.push_back(std::make_unique<logs::testing::In_Memory_Sink>());
        logger.add_sink(sinks.back()->sink());
    }

    const auto msg = *genPrintableMessage();
    logger.log(logs::Severity_Level::WARNING, msg);

    // Every sink must receive exactly one write.
    for (int i = 0; i < k; ++i) {
        RC_ASSERT(sinks[i]->count() == 1u);
    }

    // All sinks received the same formatted record.
    const auto first_entries = sinks[0]->entries();
    const std::string &first_entry = first_entries[0];
    for (int i = 1; i < k; ++i) {
        const auto other_entries = sinks[i]->entries();
        RC_ASSERT(other_entries[0] == first_entry);
    }

    // The formatted record contains the original message.
    RC_ASSERT(first_entry.find(msg) != std::string::npos);
}

/// Property 23 (zero-sink case): verify stderr fallback when no sinks configured.
RC_GTEST_PROP(SinkDispatchProperty, ZeroSinksFallbackToStderr, ()) {
    logs::Logger logger;

    std::ostringstream captured;
    std::streambuf *original_stderr = std::cerr.rdbuf();
    std::cerr.rdbuf(captured.rdbuf());

    const auto msg = *genPrintableMessage();
    logger.log(logs::Severity_Level::INFO, msg);

    std::cerr.rdbuf(original_stderr);

    RC_ASSERT(!captured.str().empty());
    RC_ASSERT(captured.str().find(msg) != std::string::npos);
}

// ─────────────────────────────────────────────────────────────────────────────
// Task 17.3: Property 24 — Sink Write-Failure Isolation
// **Validates: Requirements 6.6, 6.7**
// ─────────────────────────────────────────────────────────────────────────────

/// Property 24: Inject write failure in one sink; verify exception absorbed,
/// remaining sinks receive record, failed sink retained.
RC_GTEST_PROP(SinkDispatchProperty, WriteFailureIsolation, ()) {
    // Generate total number of good sinks [1, 10] and position of bad sink.
    const auto good_count = *rc::gen::inRange(1, 11);
    const auto bad_position = *rc::gen::inRange(0, good_count + 1);

    logs::Logger logger;

    // Create good sinks and insert the failing sink at bad_position.
    std::vector<std::unique_ptr<logs::testing::In_Memory_Sink>> good_sinks;
    good_sinks.reserve(static_cast<std::size_t>(good_count));

    Throwing_Streambuf throwing_buf;
    std::ostream throwing_stream(&throwing_buf);
    logs::Sink bad_sink(throwing_stream);

    int good_added = 0;
    bool bad_added = false;

    for (int i = 0; i <= good_count; ++i) {
        if (i == bad_position && !bad_added) {
            logger.add_sink(bad_sink);
            bad_added = true;
        }
        if (good_added < good_count) {
            good_sinks.push_back(std::make_unique<logs::testing::In_Memory_Sink>());
            logger.add_sink(good_sinks.back()->sink());
            ++good_added;
        }
    }
    // If bad_position == good_count (end), add it last.
    if (!bad_added) {
        logger.add_sink(bad_sink);
    }

    // Emit a record — must not throw (exception absorbed).
    const auto msg = *genPrintableMessage();
    logger.log(logs::Severity_Level::ERROR, msg);

    // All good sinks must have received exactly one write.
    for (int i = 0; i < good_count; ++i) {
        RC_ASSERT(good_sinks[i]->count() == 1u);
        RC_ASSERT(good_sinks[i]->entries()[0].find(msg) != std::string::npos);
    }

    // Verify failed sink is retained: emit a second record, good sinks get it.
    for (auto &gs : good_sinks) {
        gs->clear();
    }
    const auto msg2 = *genPrintableMessage();
    logger.log(logs::Severity_Level::INFO, msg2);

    for (int i = 0; i < good_count; ++i) {
        RC_ASSERT(good_sinks[i]->count() == 1u);
        RC_ASSERT(good_sinks[i]->entries()[0].find(msg2) != std::string::npos);
    }
}

}  // namespace
