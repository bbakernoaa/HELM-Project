// ─── CONF Type-Casting Unit Tests ────────────────────────────────────────────
// Verifies the typed scalar conversion behavior of conf::Config:
//   * Happy paths — a scalar whose text is a valid representation of the
//     requested type converts to the expected value (int / double / bool /
//     string), and every scalar is a valid representation for get_string.
//   * Mismatches — a scalar whose text is NOT a valid representation of the
//     requested type raises Conf_Error{Type_Mismatch} from the throwing
//     accessors, yields std::nullopt from the non-throwing accessors, and
//     collapses to the supplied fallback from get_or.
//
// The mismatch inputs are chosen to be unambiguous non-numeric / non-boolean
// scalars ("spherical", "hello") so the underlying yaml-cpp 0.8.0 conversion
// reliably fails (BadConversion -> Type_Mismatch). A SanityCheck test asserts,
// in-container, that these inputs really are rejected as int/bool, so the
// mismatch expectations below cannot silently rot against a backend quirk.
//
// Feature: conf-config-parser
// Requirements: 32.4, 5.1, 5.3, 6.1, 6.3, 7.3, 15.1, 15.2
// ─────────────────────────────────────────────────────────────────────────────

#include <conf/config.hpp>
#include <conf/error.hpp>

#include <gtest/gtest.h>

#include <optional>
#include <string>

using conf::Config;
using conf::Conf_Error;
using conf::Error_Code;

namespace {

// A document holding one scalar of each supported type plus two arbitrary,
// unambiguously non-numeric / non-boolean strings used to drive mismatches.
//   num   -> a valid int
//   pi    -> a valid double
//   flag  -> a valid bool
//   name  -> an arbitrary word (not an int, not a bool)
//   word  -> a second arbitrary word (not an int, not a bool)
constexpr const char* kYaml = R"(
num: 42
pi: 3.14159
flag: true
name: spherical
word: hello
)";

// Test fixture: parse the document once and expose it to every case.
class TypeCasting : public ::testing::Test {
protected:
    Config cfg = Config::from_string(kYaml);
};

// ─── Sanity: the chosen mismatch inputs really are unconvertible ─────────────
// Guards against a yaml-cpp version quirk where as<int>()/as<bool>() might
// accept an unexpected string. If this ever fails, the mismatch fixtures below
// must be re-chosen — the rest of the suite would otherwise be vacuous.

TEST_F(TypeCasting, SanityCheckMismatchInputsAreUnconvertible) {
    // "spherical" and "hello" must fail to parse as both int and bool so they
    // serve as reliable int-on-string and bool-on-arbitrary-string mismatches.
    EXPECT_FALSE(cfg.try_int("name").has_value());
    EXPECT_FALSE(cfg.try_bool("name").has_value());
    EXPECT_FALSE(cfg.try_int("word").has_value());
    EXPECT_FALSE(cfg.try_bool("word").has_value());
}

// ─── Happy paths: throwing accessors (Requirement 5.1) ───────────────────────

TEST_F(TypeCasting, GetIntReturnsExpectedValue) {
    EXPECT_EQ(cfg.get_int("num"), 42);
}

TEST_F(TypeCasting, GetDoubleReturnsExpectedValue) {
    EXPECT_DOUBLE_EQ(cfg.get_double("pi"), 3.14159);
}

TEST_F(TypeCasting, GetBoolReturnsExpectedValue) {
    EXPECT_TRUE(cfg.get_bool("flag"));
}

TEST_F(TypeCasting, GetStringReturnsExpectedValue) {
    EXPECT_EQ(cfg.get_string("name"), "spherical");
}

// Every scalar node is a valid representation for get_string, including ones
// that also parse as another scalar type (Requirement 5.1).
TEST_F(TypeCasting, GetStringOnNumericScalarReturnsItsText) {
    EXPECT_EQ(cfg.get_string("num"), "42");
    EXPECT_EQ(cfg.get_string("flag"), "true");
}

// ─── Happy paths: non-throwing accessors agree with throwing (Req 6.1) ───────

TEST_F(TypeCasting, TryAccessorsReturnEngagedOptionalsOnSuccess) {
    const std::optional<int>         i = cfg.try_int("num");
    const std::optional<double>      d = cfg.try_double("pi");
    const std::optional<bool>        b = cfg.try_bool("flag");
    const std::optional<std::string> s = cfg.try_string("name");

    ASSERT_TRUE(i.has_value());
    ASSERT_TRUE(d.has_value());
    ASSERT_TRUE(b.has_value());
    ASSERT_TRUE(s.has_value());

    EXPECT_EQ(*i, 42);
    EXPECT_DOUBLE_EQ(*d, 3.14159);
    EXPECT_TRUE(*b);
    EXPECT_EQ(*s, "spherical");
}

// get_or returns the converted value when the path resolves and converts.
TEST_F(TypeCasting, GetOrReturnsConvertedValueOnSuccess) {
    EXPECT_EQ(cfg.get_or<int>("num", -1), 42);
    EXPECT_DOUBLE_EQ(cfg.get_or<double>("pi", -1.0), 3.14159);
    EXPECT_TRUE(cfg.get_or<bool>("flag", false));
    EXPECT_EQ(cfg.get_or<std::string>("name", std::string("fallback")), "spherical");
}

// ─── Mismatches: throwing accessors raise Type_Mismatch (Req 5.3, 15.1) ──────

TEST_F(TypeCasting, GetIntOnNonNumericStringThrowsTypeMismatch) {
    // int-on-string: "spherical" is not a valid int.
    try {
        static_cast<void>(cfg.get_int("name"));
        FAIL() << "get_int on a non-numeric string should throw";
    } catch (const Conf_Error& e) {
        EXPECT_EQ(e.code(), Error_Code::Type_Mismatch);
    }
}

TEST_F(TypeCasting, GetBoolOnArbitraryStringThrowsTypeMismatch) {
    // bool-on-arbitrary-string: "spherical" is not a valid bool.
    try {
        static_cast<void>(cfg.get_bool("name"));
        FAIL() << "get_bool on an arbitrary string should throw";
    } catch (const Conf_Error& e) {
        EXPECT_EQ(e.code(), Error_Code::Type_Mismatch);
    }
}

// The Conf_Error is also catchable as a std::exception via its base class, which
// the C bridge relies on; assert the exact type is thrown.
TEST_F(TypeCasting, GetIntMismatchThrowsConfErrorType) {
    EXPECT_THROW(static_cast<void>(cfg.get_int("word")), Conf_Error);
    EXPECT_THROW(static_cast<void>(cfg.get_bool("word")), Conf_Error);
}

// ─── Mismatches: non-throwing accessors return nullopt (Req 6.3, 15.2) ───────

TEST_F(TypeCasting, TryAccessorsReturnNulloptOnMismatch) {
    EXPECT_EQ(cfg.try_int("name"), std::nullopt);
    EXPECT_EQ(cfg.try_bool("name"), std::nullopt);
}

// ─── Mismatches: get_or returns the fallback (Req 7.3, 15.2) ─────────────────

TEST_F(TypeCasting, GetOrReturnsFallbackOnMismatch) {
    EXPECT_EQ(cfg.get_or<int>("name", -1), -1);
    EXPECT_FALSE(cfg.get_or<bool>("name", false));
}

} // namespace
