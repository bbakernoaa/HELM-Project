// ─── CONF conf::Value Unit Tests ─────────────────────────────────────────────
// Exercises conf::Value, the lightweight non-owning view returned by
// conf::Config::at, across every node kind in a representative document:
//   * Config::at on a defined node returns a Value viewing that node.
//   * Config::at on a missing path raises Conf_Error{Key_Not_Found}.
//   * Config::at on a malformed path raises Conf_Error{Invalid_Arg}.
//   * Value::kind / Value::size are correct for map, sequence, scalar, and null.
//   * The throwing (as_*) and non-throwing (try_*) conversions agree on success
//     and on failure (as_int succeeds <=> try_int engaged; as_int throws
//     Type_Mismatch <=> try_int == nullopt).
//
// LIFETIME CONTRACT: a Value views a node owned by the parent Config. Every
// test keeps the Config alive (named local, never a temporary) for the entire
// duration the Value is used, per the Value validity rule.
//
// Feature: conf-config-parser
// Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8
// ─────────────────────────────────────────────────────────────────────────────

#include <conf/config.hpp>
#include <conf/error.hpp>
#include <conf/value.hpp>

#include <gtest/gtest.h>

#include <optional>
#include <string>

using conf::Config;
using conf::Conf_Error;
using conf::Error_Code;
using conf::Node_Kind;
using conf::Value;

namespace {

// A representative document holding every node kind addressable under "root":
//   root.m       -> map with two entries (a, b)
//   root.seq     -> sequence of three integers
//   root.num     -> integer scalar (42)
//   root.name    -> string scalar ("hello")
//   root.nothing -> explicit null
constexpr const char* kDoc =
    "root:\n"
    "  m: { a: 1, b: 2 }\n"
    "  seq: [10, 20, 30]\n"
    "  num: 42\n"
    "  name: hello\n"
    "  nothing: null\n";

} // namespace

// ─── at on a defined map node returns a Map Value (Req 10.1, 10.4, 10.5) ─────

TEST(ValueKind, MapNodeReportsMapKindAndChildCount) {
    Config cfg = Config::from_string(kDoc);

    const Value v = cfg.at("root.m");

    EXPECT_EQ(v.kind(), Node_Kind::Map);
    EXPECT_TRUE(v.is_defined());
    EXPECT_EQ(v.size(), 2u);
}

// ─── at on a sequence node reports Sequence kind and element count (10.4, 10.5)

TEST(ValueKind, SequenceNodeReportsSequenceKindAndElementCount) {
    Config cfg = Config::from_string(kDoc);

    const Value v = cfg.at("root.seq");

    EXPECT_EQ(v.kind(), Node_Kind::Sequence);
    EXPECT_TRUE(v.is_defined());
    EXPECT_EQ(v.size(), 3u);
}

// ─── at on an integer scalar: kind Scalar, size 0, converts to int (10.4-10.6)

TEST(ValueKind, IntScalarReportsScalarKindSizeZeroAndConverts) {
    Config cfg = Config::from_string(kDoc);

    const Value v = cfg.at("root.num");

    EXPECT_EQ(v.kind(), Node_Kind::Scalar);
    EXPECT_TRUE(v.is_defined());
    EXPECT_EQ(v.size(), 0u);

    // Every scalar is a valid string; an integer scalar also converts to int.
    EXPECT_EQ(v.as_int(), 42);
    EXPECT_EQ(v.as_string(), "42");
}

// ─── at on a null node reports Null kind and size 0 (Req 10.4, 10.5) ─────────

TEST(ValueKind, NullNodeReportsNullKindAndSizeZero) {
    Config cfg = Config::from_string(kDoc);

    const Value v = cfg.at("root.nothing");

    EXPECT_EQ(v.kind(), Node_Kind::Null);
    // A Null node is a defined node (is_defined() is true unless Undefined).
    EXPECT_TRUE(v.is_defined());
    EXPECT_EQ(v.size(), 0u);
}

// ─── at on a missing path raises Key_Not_Found (Req 10.2) ────────────────────

TEST(ValueAtErrors, MissingPathRaisesKeyNotFound) {
    Config cfg = Config::from_string(kDoc);

    try {
        const Value v = cfg.at("root.absent");
        FAIL() << "at on a non-resolving path must throw Conf_Error";
    } catch (const Conf_Error& e) {
        EXPECT_EQ(e.code(), Error_Code::Key_Not_Found);
    } catch (...) {
        FAIL() << "at must throw conf::Conf_Error, not another exception type";
    }
}

// ─── at on a malformed path raises Invalid_Arg (Req 10.3) ────────────────────
// An empty path and a path with an empty segment (consecutive dots) are both
// malformed and must be rejected before any node traversal.

TEST(ValueAtErrors, EmptyPathRaisesInvalidArg) {
    Config cfg = Config::from_string(kDoc);

    try {
        const Value v = cfg.at("");
        FAIL() << "at on an empty path must throw Conf_Error";
    } catch (const Conf_Error& e) {
        EXPECT_EQ(e.code(), Error_Code::Invalid_Arg);
    } catch (...) {
        FAIL() << "at must throw conf::Conf_Error, not another exception type";
    }
}

TEST(ValueAtErrors, ConsecutiveDotsRaiseInvalidArg) {
    Config cfg = Config::from_string(kDoc);

    try {
        const Value v = cfg.at("root..m");
        FAIL() << "at on a path with an empty segment must throw Conf_Error";
    } catch (const Conf_Error& e) {
        EXPECT_EQ(e.code(), Error_Code::Invalid_Arg);
    } catch (...) {
        FAIL() << "at must throw conf::Conf_Error, not another exception type";
    }
}

// ─── as_*/try_* agree on success (Req 10.6, 10.8) ────────────────────────────
// For a scalar that converts, the throwing and non-throwing flavors return the
// same value.

TEST(ValueConversionAgreement, ThrowingAndNonThrowingAgreeOnSuccess) {
    Config cfg = Config::from_string(kDoc);

    const Value v = cfg.at("root.num");

    const std::optional<int> maybe = v.try_int();
    ASSERT_TRUE(maybe.has_value());
    EXPECT_EQ(*maybe, 42);
    EXPECT_EQ(v.as_int(), *maybe);

    const std::optional<std::string> maybe_str = v.try_string();
    ASSERT_TRUE(maybe_str.has_value());
    EXPECT_EQ(*maybe_str, "42");
    EXPECT_EQ(v.as_string(), *maybe_str);
}

// ─── as_*/try_* agree on failure (Req 10.7, 10.8) ────────────────────────────
// A string scalar ("hello") converts to string but NOT to int: as_int throws
// Type_Mismatch and try_int returns nullopt — the two flavors agree.

TEST(ValueConversionAgreement, ThrowingAndNonThrowingAgreeOnFailure) {
    Config cfg = Config::from_string(kDoc);

    const Value v = cfg.at("root.name");
    ASSERT_EQ(v.kind(), Node_Kind::Scalar);

    // Succeeds as a string.
    EXPECT_EQ(v.as_string(), "hello");
    const std::optional<std::string> as_str = v.try_string();
    ASSERT_TRUE(as_str.has_value());
    EXPECT_EQ(*as_str, "hello");

    // Fails as an int: as_int throws Type_Mismatch, try_int yields nullopt.
    EXPECT_FALSE(v.try_int().has_value());
    try {
        const int unused = v.as_int();
        static_cast<void>(unused);
        FAIL() << "as_int on a non-numeric scalar must throw Conf_Error";
    } catch (const Conf_Error& e) {
        EXPECT_EQ(e.code(), Error_Code::Type_Mismatch);
    } catch (...) {
        FAIL() << "as_int must throw conf::Conf_Error, not another exception type";
    }
}

// ─── as_*/try_* agree on a non-scalar node (Req 10.7, 10.8) ──────────────────
// A map node is not convertible to any scalar type: every as_* throws
// Type_Mismatch and every try_* returns nullopt.

TEST(ValueConversionAgreement, NonScalarNodeFailsEveryConversion) {
    Config cfg = Config::from_string(kDoc);

    const Value v = cfg.at("root.m");
    ASSERT_EQ(v.kind(), Node_Kind::Map);

    EXPECT_FALSE(v.try_int().has_value());
    EXPECT_FALSE(v.try_double().has_value());
    EXPECT_FALSE(v.try_bool().has_value());
    EXPECT_FALSE(v.try_string().has_value());

    EXPECT_THROW(static_cast<void>(v.as_int()), Conf_Error);
    EXPECT_THROW(static_cast<void>(v.as_double()), Conf_Error);
    EXPECT_THROW(static_cast<void>(v.as_bool()), Conf_Error);
    EXPECT_THROW(static_cast<void>(v.as_string()), Conf_Error);
}
