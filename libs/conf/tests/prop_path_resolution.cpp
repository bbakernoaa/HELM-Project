// ─── Property-Based Tests: conf::Config dotted-path resolver totality ────────
// Feature: conf-config-parser, Property 3: Resolver Totality
//
// Uses RapidCheck to verify that for ANY arbitrary key string (0..1,048,576
// bytes: empty, leading/trailing/consecutive dots, random ASCII, UTF-8, raw
// non-UTF-8 bytes, and oversized all-digit runs) the public conf::Config
// resolver is TOTAL:
//
//   * Non-throwing accessors (try_int/try_double/try_bool/try_string, get_or,
//     and the introspection accessors has/is_map/is_sequence/size) are noexcept
//     and ALWAYS return a defined result — completing the call proves totality.
//   * Throwing accessors (get_int/get_double/get_bool/get_string, at, and the
//     list accessors) either return a value OR raise a typed conf::Conf_Error;
//     NO other exception type may escape, and the carried code() is always one
//     of Invalid_Arg / Key_Not_Found / Type_Mismatch.
//   * No crash / abort / hang / leak / UB on any input. Simply completing each
//     bounded iteration demonstrates the absence of a crash or hang.
//
// PROPERTY EXPECTATION (feature spec): this property MUST pass on the correct
// total resolver. A non-Conf_Error escaping, a crash, or UB is a GENUINE
// implementation bug in src/detail/yaml_tree.cpp or src/config.cpp, not a test
// defect — it must be reported with its counterexample, never masked.
//
// The Config under test is a fixed, modestly nested document built ONCE (nested
// maps, sequences, a sequence of maps, scalars of every flavour, and a null).
// Per-iteration work is bounded: random keys are capped at a few KB (totality,
// not throughput, is the point); the full 1 MiB upper bound from Requirement 29
// is covered deterministically by ExplicitEdgeAndOversizedKeys below.
//
// **Validates: Requirements 29.1, 29.2, 29.3, 3.7**
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include <conf/config.hpp>
#include <conf/error.hpp>
#include <conf/value.hpp>

namespace {

// ─── Fixed configuration document (built exactly once) ───────────────────────
// A modestly nested tree so resolution walks varied branches: top-level map,
// nested maps, a scalar sequence, a sequence of maps, scalars of each type, and
// an explicit null leaf.
constexpr const char* kFixtureYaml = R"YAML(
model:
  name: helm
  physics:
    layers: 42
    timestep: 0.25
    enabled: true
  grid:
    resolution: [10, 20, 30]
    labels: [alpha, beta, gamma]
items:
  - id: 1
    kind: alpha
  - id: 2
    kind: beta
scalars:
  pi: 3.14159
  flag: false
  empty:
  text: "hello world"
)YAML";

/// @brief The single shared Config under test, parsed once on first use.
/// Config is move-only; we hand out a const reference to a function-local
/// static. Every accessor exercised here is const (at() mutates only a private
/// mutable node store), so a const reference is sufficient and thread-safe for
/// single-process property runs.
const conf::Config& fixed_config() {
    static const conf::Config cfg = conf::Config::from_string(kFixtureYaml);
    return cfg;
}

// ─── Cap on randomly generated key length ────────────────────────────────────
// A few-KB cap keeps each generated iteration fast while still exercising long
// inputs, leading/trailing/double dots, and oversized digit runs. The Req-29
// ceiling of 1,048,576 bytes is hit explicitly (and deterministically) below.
constexpr std::size_t kMaxRandomKeyLen = 4096;

// Generator 1: truly arbitrary bytes. Covers the empty string, random ASCII,
// UTF-8, and raw non-UTF-8 byte sequences (char spans the full byte range).
rc::Gen<std::string> genArbitraryBytes() {
    return rc::gen::resize(
        kMaxRandomKeyLen,
        rc::gen::container<std::string>(rc::gen::arbitrary<char>()));
}

// Generator 2: dot/digit/letter-biased strings, to densely exercise the
// dotted-path machinery — leading/trailing/consecutive dots, empty segments,
// and short indices interleaved with map-key-like tokens. Dots are
// over-represented via repeated alternatives in oneOf.
rc::Gen<std::string> genDottyKey() {
    auto sym = rc::gen::oneOf(
        rc::gen::just('.'), rc::gen::just('.'), rc::gen::just('.'),
        rc::gen::just('.'), rc::gen::just('.'), rc::gen::just('.'),
        rc::gen::just('0'), rc::gen::just('1'), rc::gen::just('2'),
        rc::gen::just('9'), rc::gen::just('a'), rc::gen::just('b'),
        rc::gen::just('m'));
    return rc::gen::resize(256, rc::gen::container<std::string>(sym));
}

// Generator 3: oversized all-digit runs (arbitrary-magnitude sequence indices),
// exercising the std::from_chars out-of-range path without overflow or UB.
rc::Gen<std::string> genHugeDigitRun() {
    return rc::gen::map(
        rc::gen::inRange<std::size_t>(1, kMaxRandomKeyLen + 1),
        [](std::size_t n) { return std::string(n, '9'); });
}

/// @brief Union of the three key-shape generators.
rc::Gen<std::string> genAnyKey() {
    return rc::gen::oneOf(genArbitraryBytes(), genDottyKey(), genHugeDigitRun());
}

// ─── Throwing-accessor probe ─────────────────────────────────────────────────
/// @brief Result of driving every throwing accessor with one key.
/// Both flags MUST be false on a correct total resolver.
struct Throw_Probe {
    bool escaped_non_conf_error = false;  ///< A non-Conf_Error exception escaped.
    bool bad_code               = false;  ///< A Conf_Error carried an unexpected code.
};

/// @brief Call every throwing accessor (and Config::at and the list accessors)
/// with @p key, swallowing the result. Records whether any non-Conf_Error
/// escaped or any Conf_Error carried a code outside the resolution/conversion
/// taxonomy. Never rethrows. The per-call try/catch deliberately only catches
/// the accessor's own throw — it does not wrap any RC_ASSERT, so a property
/// failure is never accidentally absorbed here.
Throw_Probe probe_throwing(const conf::Config& cfg, const std::string& key) {
    Throw_Probe probe;

    auto run = [&](auto&& invoke) {
        try {
            invoke();
        } catch (const conf::Conf_Error& e) {
            const conf::Error_Code c = e.code();
            const bool allowed = (c == conf::Error_Code::Invalid_Arg) ||
                                 (c == conf::Error_Code::Key_Not_Found) ||
                                 (c == conf::Error_Code::Type_Mismatch);
            if (!allowed) {
                probe.bad_code = true;
            }
        } catch (...) {
            probe.escaped_non_conf_error = true;
        }
    };

    run([&] { (void)cfg.get_int(key); });
    run([&] { (void)cfg.get_double(key); });
    run([&] { (void)cfg.get_bool(key); });
    run([&] { (void)cfg.get_string(key); });
    run([&] { (void)cfg.at(key); });
    run([&] { (void)cfg.get_int_list(key); });
    run([&] { (void)cfg.get_double_list(key); });
    run([&] { (void)cfg.get_string_list(key); });

    return probe;
}

} // namespace

// ─── Property 3: arbitrary key strings never break the resolver ──────────────
// Feature: conf-config-parser, Property 3: Resolver Totality
//
// For any generated key string the non-throwing surface returns defined results
// (and obeys basic consistency invariants), and the throwing surface raises only
// typed conf::Conf_Error with an expected code. Completing each iteration proves
// no crash, abort, hang, or UB occurred.
//
// **Validates: Requirements 29.1, 29.2, 29.3, 3.7**
RC_GTEST_PROP(ResolverTotalityProperty3,
              ArbitraryKeyYieldsDefinedResultOrTypedError,
              ()) {
    const conf::Config& cfg = fixed_config();
    const std::string key = *genAnyKey();

    // ── Non-throwing surface (Req 29.1, 29.2): noexcept; calling them at all
    //    demonstrates totality. We also assert the few consistency invariants
    //    that must hold so the returned results are genuinely checked. ──
    const std::optional<int>         oi = cfg.try_int(key);
    const std::optional<double>      od = cfg.try_double(key);
    const std::optional<bool>        ob = cfg.try_bool(key);
    const std::optional<std::string> os = cfg.try_string(key);

    const bool        h  = cfg.has(key);
    const bool        im = cfg.is_map(key);
    const bool        is = cfg.is_sequence(key);
    const std::size_t sz = cfg.size(key);

    // Defaulted accessor must also never throw and always return a value.
    (void)cfg.get_or<int>(key, -1);
    (void)cfg.get_or<double>(key, -1.0);
    (void)cfg.get_or<bool>(key, false);
    (void)cfg.get_or<std::string>(key, std::string{});

    // A resolved map or sequence implies the key resolves; a node is never both.
    RC_ASSERT(!im || h);
    RC_ASSERT(!is || h);
    RC_ASSERT(!(im && is));

    // An unresolvable / malformed key yields no typed value and zero size.
    if (!h) {
        RC_ASSERT(!oi.has_value());
        RC_ASSERT(!od.has_value());
        RC_ASSERT(!ob.has_value());
        RC_ASSERT(!os.has_value());
        RC_ASSERT(sz == 0);
    }

    // ── Throwing surface (Req 29.3, 3.7): only typed Conf_Error may escape. ──
    const Throw_Probe probe = probe_throwing(cfg, key);
    RC_ASSERT(probe.escaped_non_conf_error == false);
    RC_ASSERT(probe.bad_code == false);
}

// ─── Deterministic coverage of edge cases and the 1 MiB ceiling ──────────────
// Feature: conf-config-parser, Property 3: Resolver Totality
//
// Pins the corner cases that random generation may hit only rarely (or not at
// the full byte ceiling): the empty string, every dotted-path malformation,
// embedded NUL / non-UTF-8 bytes, keys that DO resolve (exercising the
// value-returning path of the throwing accessors), and ~1,048,576-byte inputs
// (all-digit, single-token, all-dots, alternating, and an oversized index on a
// real sequence). None may crash, leak, or throw a non-Conf_Error.
//
// **Validates: Requirements 29.1, 29.2, 29.3, 3.7**
TEST(ResolverTotalityProperty3, ExplicitEdgeAndOversizedKeys) {
    const conf::Config& cfg = fixed_config();

    constexpr std::size_t kMiB = 1048576;  // Requirement 29 upper bound.

    std::vector<std::string> keys;

    // Malformed paths (Invalid_Arg territory).
    keys.emplace_back("");                 // empty path
    keys.emplace_back(".");                // single dot
    keys.emplace_back("..");               // two empty segments
    keys.emplace_back(".lead");            // leading dot
    keys.emplace_back("trail.");           // trailing dot
    keys.emplace_back("a..b");             // consecutive dots
    keys.emplace_back("model.");           // trailing dot after a real key
    keys.emplace_back(".model");           // leading dot before a real key
    keys.emplace_back("model..physics");   // empty interior segment

    // Paths that resolve — exercise the value-returning branch of every accessor.
    keys.emplace_back("model");                    // -> map
    keys.emplace_back("model.physics");            // -> nested map
    keys.emplace_back("model.physics.layers");     // -> scalar int
    keys.emplace_back("model.physics.timestep");   // -> scalar double
    keys.emplace_back("model.physics.enabled");    // -> scalar bool
    keys.emplace_back("model.grid.resolution");    // -> sequence
    keys.emplace_back("model.grid.resolution.0");  // -> sequence element
    keys.emplace_back("items");                    // -> sequence of maps
    keys.emplace_back("items.0.id");               // -> nested via sequence
    keys.emplace_back("scalars.text");             // -> string scalar
    keys.emplace_back("scalars.empty");            // -> null leaf

    // Non-resolving but well-formed (Key_Not_Found territory).
    keys.emplace_back("model.grid.resolution.99"); // out-of-range index
    keys.emplace_back("model.grid.resolution.x");  // non-digit index
    keys.emplace_back("model.physics.layers.0");   // descend past a scalar
    keys.emplace_back("does.not.exist");           // missing keys

    // Embedded NUL and raw non-UTF-8 bytes.
    keys.emplace_back(std::string{'\x00', '\xff', '\xfe', '\x80', '.', '\x01'});
    keys.emplace_back(std::string{'\xc3', '\x28'});            // invalid UTF-8
    keys.emplace_back(std::string{'k', 'e', 'y', '\x00', 'x'}); // interior NUL

    // ~1 MiB inputs (Requirement 29 ceiling).
    keys.emplace_back(std::string(kMiB, '9'));   // all-digit run
    keys.emplace_back(std::string(kMiB, 'a'));   // single oversized token
    keys.emplace_back(std::string(kMiB, '.'));   // all dots (empty first segment)

    {   // alternating "7.7.7..." — ~500k single-char segments
        std::string big;
        big.reserve(kMiB);
        for (std::size_t i = 0; i < kMiB; ++i) {
            big.push_back((i % 2 == 0) ? '7' : '.');
        }
        keys.push_back(std::move(big));
    }

    // Oversized index applied to a REAL sequence node (exercises the
    // from_chars out-of-range branch on a Sequence rather than a missing key).
    keys.emplace_back(std::string("model.grid.resolution.") +
                      std::string(kMiB, '9'));

    for (const std::string& key : keys) {
        // Non-throwing surface must complete (noexcept) for every key.
        (void)cfg.try_int(key);
        (void)cfg.try_double(key);
        (void)cfg.try_bool(key);
        (void)cfg.try_string(key);
        (void)cfg.has(key);
        (void)cfg.is_map(key);
        (void)cfg.is_sequence(key);
        (void)cfg.size(key);
        (void)cfg.get_or<int>(key, -1);
        (void)cfg.get_or<double>(key, -1.0);
        (void)cfg.get_or<bool>(key, false);
        (void)cfg.get_or<std::string>(key, std::string{});

        // Throwing surface must raise only typed Conf_Error with an expected code.
        const Throw_Probe probe = probe_throwing(cfg, key);
        EXPECT_FALSE(probe.escaped_non_conf_error)
            << "a non-Conf_Error escaped for a key of size " << key.size();
        EXPECT_FALSE(probe.bad_code)
            << "a Conf_Error carried an unexpected code for a key of size "
            << key.size();
    }
}
