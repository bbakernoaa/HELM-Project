// --- Property-Based Tests: Round-Trip Type Fidelity --------------------------
// Feature: conf-config-parser
//
// Uses RapidCheck to verify that typed values serialized to YAML and read back
// through conf::Config's typed accessors are exactly equal to the originals.
//
// Property 1: Round-Trip
//   - 32-bit ints: exact equality
//   - doubles: bit-exact (serialized with ≥17 significant digits)
//   - bools: exact equality
//   - strings: byte-identical (0–1000 bytes, printable ASCII)
//
// **Validates: Requirements 27.1, 27.2, 27.3, 27.4, 27.5**
// -----------------------------------------------------------------------------

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <charconv>
#include <cmath>
#include <conf/config.hpp>
#include <conf/error.hpp>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>

using conf::Config;

namespace {

// ── YAML serialization helpers ──────────────────────────────────────────────

/// Serialize an int32_t as a simple YAML key-value pair.
std::string yaml_int(int32_t val) {
    return "key: " + std::to_string(val) + "\n";
}

/// Serialize a double with full precision (17 significant digits) as YAML.
std::string yaml_double(double val) {
    // Use std::to_chars for round-trip-safe formatting with max precision.
    char buf[64];
    auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), val, std::chars_format::general, 17);
    std::string repr(buf, static_cast<std::size_t>(ptr - buf));
    return "key: " + repr + "\n";
}

/// Serialize a bool as "true" or "false" in YAML.
std::string yaml_bool(bool val) {
    return std::string("key: ") + (val ? "true" : "false") + "\n";
}

/// Serialize a string as a double-quoted YAML value, escaping special chars.
std::string yaml_string(const std::string &val) {
    std::string escaped;
    escaped.reserve(val.size() + 16);
    escaped += "key: \"";
    for (unsigned char ch : val) {
        switch (ch) {
            case '"':
                escaped += "\\\"";
                break;
            case '\\':
                escaped += "\\\\";
                break;
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\t':
                escaped += "\\t";
                break;
            case '\0':
                escaped += "\\0";
                break;
            default:
                escaped += static_cast<char>(ch);
                break;
        }
    }
    escaped += "\"\n";
    return escaped;
}

/// Generate a printable ASCII character (0x20–0x7E).
rc::Gen<char> genPrintableAscii() {
    return rc::gen::map(rc::gen::inRange(0x20, 0x7F), [](int c) { return static_cast<char>(c); });
}

/// Generate a printable ASCII string of bounded length (0–1000 bytes).
rc::Gen<std::string> genPrintableString() {
    return rc::gen::mapcat(rc::gen::inRange<std::size_t>(0, 1001),
                           [](std::size_t len) { return rc::gen::container<std::string>(len, genPrintableAscii()); });
}

}  // anonymous namespace

// --- Property 1a: Integer round-trip -----------------------------------------
// Feature: conf-config-parser, Property 1: Round-Trip
//
// For any random int32_t value, serializing it as `key: <value>` YAML, loading
// with from_string, and reading with get_int("key") returns the exact original.
//
// **Validates: Requirements 27.1, 27.5**

RC_GTEST_PROP(RoundTripProperty1, IntRoundTrip, ()) {
    const int32_t original = *rc::gen::arbitrary<int32_t>();

    const std::string yaml = yaml_int(original);
    const auto cfg = Config::from_string(yaml);

    const int result = cfg.get_int("key");
    RC_ASSERT(result == original);
}

// --- Property 1b: Double round-trip ------------------------------------------
// Feature: conf-config-parser, Property 1: Round-Trip
//
// For any random finite double value, serializing it with 17 significant digits
// as `key: <value>` YAML, loading with from_string, and reading with
// get_double("key") returns a bit-exact result.
//
// **Validates: Requirements 27.2, 27.5**

RC_GTEST_PROP(RoundTripProperty1, DoubleRoundTrip, ()) {
    double original = *rc::gen::arbitrary<double>();

    // Filter out non-finite values (NaN, Inf) — YAML doesn't round-trip them
    // through standard numeric parsing.
    RC_PRE(std::isfinite(original));

    const std::string yaml = yaml_double(original);
    const auto cfg = Config::from_string(yaml);

    const double result = cfg.get_double("key");

    // Bit-exact comparison via memcmp (handles +0.0 vs -0.0 correctly).
    RC_ASSERT(std::memcmp(&result, &original, sizeof(double)) == 0);
}

// --- Property 1c: Bool round-trip --------------------------------------------
// Feature: conf-config-parser, Property 1: Round-Trip
//
// For any random bool value, serializing it as "true"/"false" YAML, loading
// with from_string, and reading with get_bool("key") returns the exact original.
//
// **Validates: Requirements 27.3, 27.5**

RC_GTEST_PROP(RoundTripProperty1, BoolRoundTrip, ()) {
    const bool original = *rc::gen::arbitrary<bool>();

    const std::string yaml = yaml_bool(original);
    const auto cfg = Config::from_string(yaml);

    const bool result = cfg.get_bool("key");
    RC_ASSERT(result == original);
}

// --- Property 1d: String round-trip ------------------------------------------
// Feature: conf-config-parser, Property 1: Round-Trip
//
// For any random printable ASCII string (0–1000 bytes), serializing it as a
// double-quoted YAML value, loading with from_string, and reading with
// get_string("key") returns a byte-identical result.
//
// **Validates: Requirements 27.4, 27.5**

RC_GTEST_PROP(RoundTripProperty1, StringRoundTrip, ()) {
    const std::string original = *genPrintableString();

    const std::string yaml = yaml_string(original);
    const auto cfg = Config::from_string(yaml);

    const std::string result = cfg.get_string("key");
    RC_ASSERT(result == original);
}
