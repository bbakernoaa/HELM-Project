// ─── CONF Missing-Key Unit Tests ─────────────────────────────────────────────
// Proves the mandated missing-key behavior across the throwing and non-throwing
// access flavors (design "Error Scenario 1: Missing Key"). A dotted-path key
// fails to resolve in four distinct ways:
//   * missing leaf            — a final map segment names an absent key
//   * missing intermediate    — a non-final map segment names an absent key
//   * out-of-range index      — a sequence segment indexes past the last element
//   * descend past a scalar    — a non-final segment lands on a scalar leaf
//
// For every such path the throwing accessors (get_int/get_double/get_bool/
// get_string, the list accessors, and Config::at) must raise
// Conf_Error{Key_Not_Found}; the non-throwing accessors (try_int/try_double/
// try_bool/try_string) must yield std::nullopt; and has() must return false.
//
// Feature: conf-config-parser
// Requirements: 32.3, 13.1, 13.2
// ─────────────────────────────────────────────────────────────────────────────

#include <conf/config.hpp>
#include <conf/error.hpp>
#include <conf/value.hpp>

#include <gtest/gtest.h>

#include <string>
#include <utility>

using conf::Conf_Error;
using conf::Config;
using conf::Error_Code;

namespace {

// A known document exercising every node shape the resolver can land on or walk
// through: a top-level map (model) containing a nested map (physics) with scalar
// leaves, a sequence (list) of three elements, and a top-level scalar leaf.
constexpr const char* kDoc =
    "model:\n"
    "  physics:\n"
    "    layers: 42\n"
    "    scheme: spherical\n"
    "list:\n"
    "  - 10\n"
    "  - 20\n"
    "  - 30\n"
    "scalar: hello\n";

// The four canonical non-resolving paths over kDoc.
//   * "model.physics.absent" — missing leaf (final segment absent on a map)
//   * "nope.layers"          — missing intermediate (first segment absent on root)
//   * "list.99"              — out-of-range sequence index
//   * "scalar.child"         — descend past a scalar leaf
struct MissingCase {
    const char* path;
    const char* description;
};

const MissingCase kMissingCases[] = {
    {"model.physics.absent", "missing leaf key"},
    {"nope.layers", "missing intermediate key"},
    {"list.99", "out-of-range sequence index"},
    {"scalar.child", "descend past a scalar leaf"},
};

// Assert that a callable throws Conf_Error whose code() == Key_Not_Found.
template <typename Fn>
void ExpectKeyNotFound(Fn&& fn, const std::string& context) {
    try {
        std::forward<Fn>(fn)();
        FAIL() << "expected Conf_Error{Key_Not_Found} for " << context;
    } catch (const Conf_Error& e) {
        EXPECT_EQ(e.code(), Error_Code::Key_Not_Found)
            << "wrong Error_Code for " << context;
    } catch (...) {
        FAIL() << "expected Conf_Error (not another exception type) for "
               << context;
    }
}

} // namespace

// ─── Sanity: the known-good paths resolve as expected ────────────────────────
// Guards against a vacuous suite (a typo'd fixture where every path is missing
// would otherwise let the negative tests pass for the wrong reason).

TEST(MissingKeys, KnownGoodPathsResolve) {
    const Config cfg = Config::from_string(kDoc);

    EXPECT_EQ(cfg.get_int("model.physics.layers"), 42);
    EXPECT_EQ(cfg.get_string("model.physics.scheme"), "spherical");
    EXPECT_EQ(cfg.get_int("list.0"), 10);
    EXPECT_EQ(cfg.get_int("list.2"), 30);
    EXPECT_EQ(cfg.get_string("scalar"), "hello");

    EXPECT_TRUE(cfg.has("model.physics.layers"));
    EXPECT_TRUE(cfg.has("list.2"));
    EXPECT_TRUE(cfg.has("scalar"));
}

// ─── Throwing scalar accessors → Key_Not_Found (Req 13.1) ────────────────────

TEST(MissingKeys, ThrowingScalarAccessorsReportKeyNotFound) {
    const Config cfg = Config::from_string(kDoc);

    for (const MissingCase& c : kMissingCases) {
        const std::string ctx = c.description;
        ExpectKeyNotFound([&] { (void)cfg.get_int(c.path); }, ctx + " (get_int)");
        ExpectKeyNotFound([&] { (void)cfg.get_double(c.path); },
                          ctx + " (get_double)");
        ExpectKeyNotFound([&] { (void)cfg.get_bool(c.path); },
                          ctx + " (get_bool)");
        ExpectKeyNotFound([&] { (void)cfg.get_string(c.path); },
                          ctx + " (get_string)");
    }
}

// ─── Throwing list accessors and at() → Key_Not_Found (Req 13.1) ─────────────

TEST(MissingKeys, ThrowingListAndAtAccessorsReportKeyNotFound) {
    const Config cfg = Config::from_string(kDoc);

    for (const MissingCase& c : kMissingCases) {
        const std::string ctx = c.description;
        ExpectKeyNotFound([&] { (void)cfg.get_int_list(c.path); },
                          ctx + " (get_int_list)");
        ExpectKeyNotFound([&] { (void)cfg.get_double_list(c.path); },
                          ctx + " (get_double_list)");
        ExpectKeyNotFound([&] { (void)cfg.get_string_list(c.path); },
                          ctx + " (get_string_list)");
        ExpectKeyNotFound([&] { (void)cfg.at(c.path); }, ctx + " (at)");
    }
}

// ─── Non-throwing scalar accessors → std::nullopt (Req 13.2) ─────────────────

TEST(MissingKeys, NonThrowingScalarAccessorsReturnNullopt) {
    const Config cfg = Config::from_string(kDoc);

    for (const MissingCase& c : kMissingCases) {
        EXPECT_FALSE(cfg.try_int(c.path).has_value())
            << "try_int should be nullopt for " << c.description;
        EXPECT_FALSE(cfg.try_double(c.path).has_value())
            << "try_double should be nullopt for " << c.description;
        EXPECT_FALSE(cfg.try_bool(c.path).has_value())
            << "try_bool should be nullopt for " << c.description;
        EXPECT_FALSE(cfg.try_string(c.path).has_value())
            << "try_string should be nullopt for " << c.description;
    }
}

// ─── has() → false for every missing path (Req 13.2) ─────────────────────────

TEST(MissingKeys, HasReturnsFalseForMissingPaths) {
    const Config cfg = Config::from_string(kDoc);

    for (const MissingCase& c : kMissingCases) {
        EXPECT_FALSE(cfg.has(c.path))
            << "has should be false for " << c.description;
    }
}

// ─── get_or() → fallback for every missing path (Req 13.2) ───────────────────

TEST(MissingKeys, GetOrReturnsFallbackForMissingPaths) {
    const Config cfg = Config::from_string(kDoc);

    for (const MissingCase& c : kMissingCases) {
        EXPECT_EQ(cfg.get_or<int>(c.path, -7), -7)
            << "get_or<int> should fall back for " << c.description;
        EXPECT_EQ(cfg.get_or<std::string>(c.path, std::string("fallback")),
                  "fallback")
            << "get_or<std::string> should fall back for " << c.description;
    }
}

// ─── Per-case focused assertions (matches the task's named examples) ─────────

TEST(MissingKeys, MissingLeafKey) {
    const Config cfg = Config::from_string(kDoc);
    ExpectKeyNotFound([&] { (void)cfg.get_int("model.physics.absent"); },
                      "missing leaf");
    EXPECT_FALSE(cfg.try_int("model.physics.absent").has_value());
    EXPECT_FALSE(cfg.has("model.physics.absent"));
}

TEST(MissingKeys, MissingIntermediateKey) {
    const Config cfg = Config::from_string(kDoc);
    ExpectKeyNotFound([&] { (void)cfg.get_int("nope.layers"); },
                      "missing intermediate");
    EXPECT_FALSE(cfg.try_int("nope.layers").has_value());
    EXPECT_FALSE(cfg.has("nope.layers"));
}

TEST(MissingKeys, OutOfRangeSequenceIndex) {
    const Config cfg = Config::from_string(kDoc);
    ExpectKeyNotFound([&] { (void)cfg.get_int("list.99"); },
                      "out-of-range index");
    EXPECT_FALSE(cfg.try_int("list.99").has_value());
    EXPECT_FALSE(cfg.has("list.99"));
}

TEST(MissingKeys, DescendPastScalar) {
    const Config cfg = Config::from_string(kDoc);
    ExpectKeyNotFound([&] { (void)cfg.get_int("scalar.child"); },
                      "descend past scalar");
    EXPECT_FALSE(cfg.try_int("scalar.child").has_value());
    EXPECT_FALSE(cfg.has("scalar.child"));
}
