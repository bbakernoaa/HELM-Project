// ─── LOGS In-Memory Test Sink ────────────────────────────────────────────────
// Provides a test utility that records all strings written through a logs::Sink
// for assertion in unit and property tests. Thread-safe for concurrency tests.
//
// Usage:
//   logs::testing::In_Memory_Sink mem_sink;
//   logger.add_sink(mem_sink.sink());
//   // ... emit log records ...
//   EXPECT_EQ(mem_sink.count(), 3);
//   EXPECT_THAT(mem_sink.entries()[0], HasSubstr("expected text"));
//
// Feature: helm-logs
// Requirements: 13.1
// ─────────────────────────────────────────────────────────────────────────────

#ifndef LOGS_TESTS_IN_MEMORY_SINK_HPP
#define LOGS_TESTS_IN_MEMORY_SINK_HPP

#include <logs/sink.hpp>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace logs::testing {

/// A custom streambuf that intercepts writes and records each write operation
/// as a separate entry in a thread-safe vector. This allows us to track
/// individual write calls rather than just accumulating text.
class Recording_Streambuf : public std::streambuf {
   public:
    Recording_Streambuf() = default;

    /// Get a thread-safe copy of all recorded entries.
    [[nodiscard]] std::vector<std::string> entries() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return entries_;
    }

    /// Get the number of recorded entries (thread-safe).
    [[nodiscard]] std::size_t count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return entries_.size();
    }

    /// Clear all recorded entries (thread-safe).
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        entries_.clear();
    }

   protected:
    /// Called by the stream when characters are written via sputn/xsputn.
    std::streamsize xsputn(const char *s, std::streamsize n) override {
        std::lock_guard<std::mutex> lock(mutex_);
        entries_.emplace_back(s, static_cast<std::size_t>(n));
        return n;
    }

    /// Called by the stream for single-character output (e.g., overflow).
    int_type overflow(int_type ch) override {
        if (ch != traits_type::eof()) {
            std::lock_guard<std::mutex> lock(mutex_);
            entries_.emplace_back(1, static_cast<char>(ch));
        }
        return ch;
    }

    /// Sync (flush) is a no-op for in-memory recording.
    int sync() override {
        return 0;
    }

   private:
    mutable std::mutex mutex_;
    std::vector<std::string> entries_;
};

/// An in-memory Sink that records all written strings for test assertion.
/// Owns a custom streambuf and std::ostream, and wraps them in a logs::Sink
/// that can be registered with the Logger via add_sink().
///
/// Thread-safe: all accessors use mutex protection so multiple threads can
/// write concurrently (as in concurrency tests) while assertions can safely
/// read entries from another thread.
class In_Memory_Sink {
   public:
    In_Memory_Sink() : stream_(&buf_), sink_(stream_) {}

    /// Get the logs::Sink suitable for registration with Logger::add_sink().
    [[nodiscard]] logs::Sink &sink() noexcept {
        return sink_;
    }

    /// Get a thread-safe copy of all recorded written strings.
    /// Each entry corresponds to one write() call on the Sink.
    [[nodiscard]] std::vector<std::string> entries() const {
        return buf_.entries();
    }

    /// Get the number of write operations recorded (thread-safe).
    [[nodiscard]] std::size_t count() const {
        return buf_.count();
    }

    /// Clear all recorded entries (thread-safe).
    void clear() {
        buf_.clear();
    }

    // Non-copyable, non-movable (owns stream resources).
    In_Memory_Sink(const In_Memory_Sink &) = delete;
    In_Memory_Sink &operator=(const In_Memory_Sink &) = delete;
    In_Memory_Sink(In_Memory_Sink &&) = delete;
    In_Memory_Sink &operator=(In_Memory_Sink &&) = delete;

   private:
    Recording_Streambuf buf_;
    std::ostream stream_;
    logs::Sink sink_;
};

}  // namespace logs::testing

#endif  // LOGS_TESTS_IN_MEMORY_SINK_HPP
