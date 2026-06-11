/// @file sink.cpp
/// @brief Sink write/flush implementation.

#include "logs/sink.hpp"

#include <ostream>

namespace logs {

Sink::Sink(std::ostream& stream) noexcept
    : stream_{&stream} {}

bool Sink::write(std::string_view formatted) noexcept {
    try {
        stream_->write(formatted.data(),
                       static_cast<std::streamsize>(formatted.size()));
        return !stream_->fail();
    } catch (...) {
        return false;
    }
}

bool Sink::flush() noexcept {
    try {
        stream_->flush();
        return !stream_->fail();
    } catch (...) {
        return false;
    }
}

} // namespace logs
