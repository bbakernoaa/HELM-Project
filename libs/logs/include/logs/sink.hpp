#ifndef LOGS_SINK_HPP
#define LOGS_SINK_HPP

/// @file sink.hpp
/// @brief Sink output-stream-only destination wrapper.

#include <cstddef>
#include <iosfwd>
#include <string_view>

namespace logs {

/// A Sink wraps exactly one std::ostream (stderr, stdout, or a caller-supplied
/// stream). It performs write and flush ONLY. It never opens, reads, closes,
/// or parses any file or input source. The caller owns the stream lifecycle.
class Sink {
public:
    /// Wrap a caller-owned output stream. Does not take ownership.
    explicit Sink(std::ostream& stream) noexcept;

    /// Write the fully formatted record text to the stream exactly once.
    [[nodiscard]] bool write(std::string_view formatted) noexcept;

    /// Flush the underlying stream.
    [[nodiscard]] bool flush() noexcept;

    Sink(const Sink&) = default;
    Sink& operator=(const Sink&) = default;

private:
    std::ostream* stream_;
};

/// Compile-time upper bound on configured sinks.
inline constexpr std::size_t MAX_SINKS = 64;

} // namespace logs

#endif // LOGS_SINK_HPP
