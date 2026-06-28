// --- Property-Based Tests: Resolver Totality Under Arbitrary Keys ------------
// Feature: conf-config-parser
//
// Uses RapidCheck to verify that the dotted-path resolver is total: for any
// arbitrary key string (empty, dot-heavy, UTF-8, non-UTF-8 bytes, oversized
// digit runs) the non-throwing accessors return a defined result and the
// throwing accessors either return a value or raise a typed Conf_Error —
// with no crash, abort, hang, leak, or undefined behavior.
//
// Property 3: Resolver Totality
//   Validates: Requirements 29.1, 29.2, 29.3, 3.7
// -----------------------------------------------------------------------------

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <conf/config.hpp>
#include <conf/error.hpp>
#include <string>
#include <string_view>

namespace {

// ─── Helpers ────────────────────────────────────────────────────────────────

/// A small but real Config built from a fixed YAML string so there is a valid
/// tree to query against. Contains nested maps, sequences, and scalars.
conf::Config make_fixture_config() {
    static const char *yaml = R"(
server:
  host: localhost
  port: 8080
  tls: true
database:
  name: mydb
  replicas:
    - primary
    - secondary
    - tertiary
grid:
  resolution:
    - 100
    - 200
    - 300
  layers: 42
empty_key: ""
null_key: ~
)";
    return conf::Config::from_string(yaml);
}

/// The set of valid Error_Codes that a throwing accessor may carry.
bool is_valid_error_code(conf::Error_Code code) {
    return code == conf::Error_Code::Invalid_Arg || code == conf::Error_Code::Key_Not_Found || code == conf::Error_Code::Type_Mismatch;
}

/// Generate an arbitrary byte string of 0 to max_len bytes (any byte 0x00-0xFF).
rc::Gen<std::string> genArbitraryPath(std::size_t max_len) {
    return rc::gen::withSize([max_len](int /*size*/) {
        return rc::gen::mapcat(rc::gen::inRange<std::size_t>(0, max_len + 1), [](std::size_t len) {
            return rc::gen::container<std::string>(len, rc::gen::inRange<char>(std::numeric_limits<char>::min(), std::numeric_limits<char>::max()));
        });
    });
}

/// Generate strings that are empty or consist only of dots.
rc::Gen<std::string> genDotHeavyPath() {
    return rc::gen::oneOf(
        // Empty string
        rc::gen::just(std::string("")),
        // Leading dot
        rc::gen::map(rc::gen::container<std::string>(rc::gen::inRange<char>('a', 'z' + 1)), [](std::string s) { return "." + s; }),
        // Trailing dot
        rc::gen::map(rc::gen::container<std::string>(rc::gen::inRange<char>('a', 'z' + 1)), [](std::string s) { return s + "."; }),
        // Consecutive dots
        rc::gen::map(rc::gen::inRange(2, 20), [](int n) { return std::string(static_cast<std::size_t>(n), '.'); }),
        // Mixed: segments with empty segments (dots surrounded by keys)
        rc::gen::map(rc::gen::inRange(1, 5), [](int n) {
            std::string result = "a";
            for (int i = 0; i < n; ++i) {
                result += "..b";
            }
            return result;
        }));
}

/// Generate oversized digit-run strings (numbers way too large for any index).
rc::Gen<std::string> genOversizedDigitRun() {
    return rc::gen::map(rc::gen::inRange(10, 100), [](int len) {
        // Build a string of 'len' 9s — guaranteed to overflow any sane index.
        return std::string(static_cast<std::size_t>(len), '9');
    });
}

}  // anonymous namespace

// --- Property 3a: Arbitrary paths never crash --------------------------------
// Feature: conf-config-parser, Property 3: Resolver Totality
//
// For random byte strings (0-10000 bytes, any byte value 0x00-0xFF) used as
// path arguments, try_int, has, size, and get_or must all return without
// crashing, aborting, hanging, or invoking UB.
//
// **Validates: Requirements 29.1, 29.2, 29.3, 3.7**

RC_GTEST_PROP(ResolverTotalityProperty3, ArbitraryPathsNeverCrash, ()) {
    auto cfg = make_fixture_config();

    // Generate a random byte string up to 10000 bytes.
    const std::string path = *genArbitraryPath(10000);

    // (Req 29.1) Non-throwing accessors return a defined result.
    const auto try_result = cfg.try_int(path);
    (void)try_result;  // Either engaged or disengaged — both are fine.

    const auto try_double_result = cfg.try_double(path);
    (void)try_double_result;

    const auto try_bool_result = cfg.try_bool(path);
    (void)try_bool_result;

    const auto try_string_result = cfg.try_string(path);
    (void)try_string_result;

    // (Req 29.1) get_or returns either the resolved value or the fallback.
    const int fallback_int = cfg.get_or(path, -999);
    (void)fallback_int;

    const std::string fallback_str = cfg.get_or<std::string>(path, std::string("fb"));
    (void)fallback_str;

    // (Req 29.2) Introspection accessors return a defined boolean or count.
    const bool has_result = cfg.has(path);
    (void)has_result;

    const bool is_map_result = cfg.is_map(path);
    (void)is_map_result;

    const bool is_seq_result = cfg.is_sequence(path);
    (void)is_seq_result;

    const std::size_t size_result = cfg.size(path);
    (void)size_result;

    // If we reach here, no crash/abort/hang/UB occurred.
    RC_ASSERT(true);
}

// --- Property 3b: Throwing accessors return value or typed error --------------
// Feature: conf-config-parser, Property 3: Resolver Totality
//
// For random paths, get_int either returns an int or throws Conf_Error with
// one of the valid codes (Invalid_Arg, Key_Not_Found, Type_Mismatch) — never
// any other exception type.
//
// **Validates: Requirements 29.3, 3.7**

RC_GTEST_PROP(ResolverTotalityProperty3, ThrowingAccessorsReturnValueOrTypedError, ()) {
    auto cfg = make_fixture_config();

    const std::string path = *genArbitraryPath(10000);

    // Test get_int
    {
        bool returned_value = false;
        bool threw_conf_error = false;
        conf::Error_Code caught_code = conf::Error_Code::Success;

        try {
            (void)cfg.get_int(path);
            returned_value = true;
        } catch (const conf::Conf_Error &e) {
            threw_conf_error = true;
            caught_code = e.code();
        } catch (...) {
            // No other exception type should be thrown.
            RC_FAIL("get_int threw a non-Conf_Error exception");
        }

        RC_ASSERT(returned_value || threw_conf_error);
        if (threw_conf_error) {
            RC_ASSERT(is_valid_error_code(caught_code));
        }
    }

    // Test get_double
    {
        bool returned_value = false;
        bool threw_conf_error = false;
        conf::Error_Code caught_code = conf::Error_Code::Success;

        try {
            (void)cfg.get_double(path);
            returned_value = true;
        } catch (const conf::Conf_Error &e) {
            threw_conf_error = true;
            caught_code = e.code();
        } catch (...) {
            RC_FAIL("get_double threw a non-Conf_Error exception");
        }

        RC_ASSERT(returned_value || threw_conf_error);
        if (threw_conf_error) {
            RC_ASSERT(is_valid_error_code(caught_code));
        }
    }

    // Test get_bool
    {
        bool returned_value = false;
        bool threw_conf_error = false;
        conf::Error_Code caught_code = conf::Error_Code::Success;

        try {
            (void)cfg.get_bool(path);
            returned_value = true;
        } catch (const conf::Conf_Error &e) {
            threw_conf_error = true;
            caught_code = e.code();
        } catch (...) {
            RC_FAIL("get_bool threw a non-Conf_Error exception");
        }

        RC_ASSERT(returned_value || threw_conf_error);
        if (threw_conf_error) {
            RC_ASSERT(is_valid_error_code(caught_code));
        }
    }

    // Test get_string
    {
        bool returned_value = false;
        bool threw_conf_error = false;
        conf::Error_Code caught_code = conf::Error_Code::Success;

        try {
            (void)cfg.get_string(path);
            returned_value = true;
        } catch (const conf::Conf_Error &e) {
            threw_conf_error = true;
            caught_code = e.code();
        } catch (...) {
            RC_FAIL("get_string threw a non-Conf_Error exception");
        }

        RC_ASSERT(returned_value || threw_conf_error);
        if (threw_conf_error) {
            RC_ASSERT(is_valid_error_code(caught_code));
        }
    }
}

// --- Property 3c: Empty and dot-heavy paths ----------------------------------
// Feature: conf-config-parser, Property 3: Resolver Totality
//
// Strings that are empty or contain only dots (leading, trailing, or
// consecutive) must never crash. try_* returns nullopt, has returns false.
//
// **Validates: Requirements 29.1, 29.2, 3.7**

RC_GTEST_PROP(ResolverTotalityProperty3, EmptyAndDotHeavyPaths, ()) {
    auto cfg = make_fixture_config();

    const std::string path = *genDotHeavyPath();

    // (Req 29.1) Non-throwing accessors return nullopt for malformed paths.
    RC_ASSERT(!cfg.try_int(path).has_value());
    RC_ASSERT(!cfg.try_double(path).has_value());
    RC_ASSERT(!cfg.try_bool(path).has_value());
    RC_ASSERT(!cfg.try_string(path).has_value());

    // (Req 29.2) Introspection returns false/0 for malformed paths.
    RC_ASSERT(!cfg.has(path));
    RC_ASSERT(!cfg.is_map(path));
    RC_ASSERT(!cfg.is_sequence(path));
    RC_ASSERT(cfg.size(path) == 0);

    // (Req 29.1) get_or returns the fallback.
    RC_ASSERT(cfg.get_or(path, 42) == 42);
    RC_ASSERT(cfg.get_or<std::string>(path, std::string("default")) == "default");

    // (Req 29.3) Throwing accessors raise Conf_Error (Invalid_Arg expected).
    bool threw_conf_error = false;
    try {
        (void)cfg.get_int(path);
    } catch (const conf::Conf_Error &e) {
        threw_conf_error = true;
        RC_ASSERT(e.code() == conf::Error_Code::Invalid_Arg);
    } catch (...) {
        RC_FAIL("get_int threw a non-Conf_Error exception for dot-heavy path");
    }
    RC_ASSERT(threw_conf_error);
}

// --- Property 3d: Oversized digit runs ---------------------------------------
// Feature: conf-config-parser, Property 3: Resolver Totality
//
// Strings like "999999999999999999999" (digit runs too large for any index)
// must not cause integer overflow, crash, or UB.
//
// **Validates: Requirements 29.1, 29.2, 29.3, 3.7**

RC_GTEST_PROP(ResolverTotalityProperty3, OversizedDigitRuns, ()) {
    auto cfg = make_fixture_config();

    const std::string path = *genOversizedDigitRun();

    // (Req 29.1) Non-throwing accessors return nullopt (no valid key).
    RC_ASSERT(!cfg.try_int(path).has_value());
    RC_ASSERT(!cfg.try_double(path).has_value());
    RC_ASSERT(!cfg.try_bool(path).has_value());
    RC_ASSERT(!cfg.try_string(path).has_value());

    // (Req 29.2) Introspection returns false/0.
    RC_ASSERT(!cfg.has(path));
    RC_ASSERT(!cfg.is_map(path));
    RC_ASSERT(!cfg.is_sequence(path));
    RC_ASSERT(cfg.size(path) == 0);

    // (Req 29.1) get_or returns the fallback.
    RC_ASSERT(cfg.get_or(path, -1) == -1);

    // (Req 29.3) Throwing accessor raises Key_Not_Found (valid single segment,
    // but no map key matches a string of only digits of this magnitude).
    bool threw_conf_error = false;
    try {
        (void)cfg.get_int(path);
    } catch (const conf::Conf_Error &e) {
        threw_conf_error = true;
        RC_ASSERT(is_valid_error_code(e.code()));
    } catch (...) {
        RC_FAIL("get_int threw a non-Conf_Error exception for oversized digit run");
    }
    RC_ASSERT(threw_conf_error);

    // Also test as a sub-path (e.g., "grid.resolution.999...9")
    const std::string sub_path = "grid.resolution." + path;

    RC_ASSERT(!cfg.try_int(sub_path).has_value());
    RC_ASSERT(!cfg.has(sub_path));
    RC_ASSERT(cfg.size(sub_path) == 0);
    RC_ASSERT(cfg.get_or(sub_path, -1) == -1);

    bool threw_sub = false;
    try {
        (void)cfg.get_int(sub_path);
    } catch (const conf::Conf_Error &e) {
        threw_sub = true;
        RC_ASSERT(is_valid_error_code(e.code()));
    } catch (...) {
        RC_FAIL("get_int threw a non-Conf_Error for oversized digit sub-path");
    }
    RC_ASSERT(threw_sub);
}
