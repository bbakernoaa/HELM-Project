// ─── CONF Dotted-Path Resolver Unit Tests ────────────────────────────────────
// Exercises conf::Config's dotted-path key resolution through its public API:
//   * Deep nesting walks map-of-map chains ("a.b.c.d").
//   * Sequence indices descend into list elements ("list.0.x").
//   * Malformed paths (empty, leading/trailing/double dot) raise
//     Conf_Error{Invalid_Arg} on every throwing accessor and on at(), checked
//     before any node traversal; the non-throwing flavors collapse to
//     nullopt / false / 0 instead of throwing.
//   * Map keys are matched literally byte-for-byte: whitespace is significant
//     and never trimmed, so a padded key resolves only under its exact spelling.
//
// Feature: conf-config-parser
// Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 16.1, 16.2
// ─────────────────────────────────────────────────────────────────────────────

#include <conf/config.hpp>
#include <conf/error.hpp>
#include <conf/value.hpp>

#include <gtest/gtest.h>

using conf::Config;
using conf::Conf_Error;
using conf::Error_Code;
using conf::Node_Kind;

namespace {

// A document exercising every resolver path the task targets:
//   * a deeply nested map chain (a.b.c.d),
//   * a sequence of maps indexed by position (list.0.x / list.1.x),
//   * keys whose literal text carries internal and surrounding whitespace.
// The two quoted keys preserve their spaces exactly; neither contains a '.',
// so each is addressable as a single dotted-path segment.
constexpr const char* kDoc = R"YAML(
a:
  b:
    c:
      d: 7
list:
  - x: 1
  - x: 2
"key with space": 5
"  pad  ": 9
)YAML";

// Assert that evaluating `expr` throws a Conf_Error whose code() equals
// `expected_code`. Using a macro keeps the failing expression visible in the
// GTest output and avoids early-returning out of the test body.
#define EXPECT_CONF_CODE(expr, expected_code)                                  \
    do {                                                                       \
        try {                                                                  \
            (void)(expr);                                                      \
            ADD_FAILURE() << "expected Conf_Error{" #expected_code "} from "   \
                          << #expr << " but no exception was thrown";          \
        } catch (const Conf_Error& e) {                                        \
            EXPECT_EQ(e.code(), (expected_code))                               \
                << "from " << #expr;                                           \
        } catch (...) {                                                        \
            ADD_FAILURE() << "expected Conf_Error from " << #expr              \
                          << " but a different exception type was thrown";     \
        }                                                                      \
    } while (0)

class DottedPath : public ::testing::Test {
protected:
    Config cfg = Config::from_string(kDoc);
};

// ─── Deep nesting (Requirements 3.1, 3.3) ────────────────────────────────────

TEST_F(DottedPath, DeepNestingResolvesToLeafScalar) {
    // Each segment advances one map level; the final segment lands on the leaf.
    EXPECT_EQ(cfg.get_int("a.b.c.d"), 7);
}

TEST_F(DottedPath, IntermediateSegmentsResolveToMaps) {
    // A full-path prefix that stops on an interior node returns that map node,
    // confirming the walk descends one level per segment (Req 3.3).
    EXPECT_TRUE(cfg.is_map("a"));
    EXPECT_TRUE(cfg.is_map("a.b"));
    EXPECT_TRUE(cfg.is_map("a.b.c"));
    EXPECT_EQ(cfg.at("a.b.c").kind(), Node_Kind::Map);
    EXPECT_EQ(cfg.at("a.b.c.d").kind(), Node_Kind::Scalar);
}

// ─── Sequence index into map (Requirements 3.2, 3.3) ─────────────────────────

TEST_F(DottedPath, SequenceIndexDescendsIntoElement) {
    // "list" is a sequence; the numeric segment selects an element by position,
    // and the trailing key indexes the map at that element.
    EXPECT_EQ(cfg.get_int("list.0.x"), 1);
    EXPECT_EQ(cfg.get_int("list.1.x"), 2);
}

TEST_F(DottedPath, SequenceNodeReportsAsSequenceWithSize) {
    EXPECT_TRUE(cfg.is_sequence("list"));
    EXPECT_EQ(cfg.size("list"), 2u);
    EXPECT_EQ(cfg.at("list").kind(), Node_Kind::Sequence);
}

// ─── Malformed paths raise Invalid_Arg — throwing accessors (Req 3.6, 16.1) ──
// The empty string, a leading dot, a trailing dot, and a double dot are each
// rejected before any traversal, in preference to Key_Not_Found / Type_Mismatch.

TEST_F(DottedPath, EmptyPathRaisesInvalidArg) {
    EXPECT_CONF_CODE(cfg.get_int(""), Error_Code::Invalid_Arg);
    EXPECT_CONF_CODE(cfg.get_double(""), Error_Code::Invalid_Arg);
    EXPECT_CONF_CODE(cfg.get_bool(""), Error_Code::Invalid_Arg);
    EXPECT_CONF_CODE(cfg.get_string(""), Error_Code::Invalid_Arg);
    EXPECT_CONF_CODE(cfg.at(""), Error_Code::Invalid_Arg);
    EXPECT_CONF_CODE(cfg.get_int_list(""), Error_Code::Invalid_Arg);
}

TEST_F(DottedPath, LeadingDotRaisesInvalidArg) {
    EXPECT_CONF_CODE(cfg.get_int(".a"), Error_Code::Invalid_Arg);
    EXPECT_CONF_CODE(cfg.get_string(".a"), Error_Code::Invalid_Arg);
    EXPECT_CONF_CODE(cfg.at(".a"), Error_Code::Invalid_Arg);
}

TEST_F(DottedPath, TrailingDotRaisesInvalidArg) {
    EXPECT_CONF_CODE(cfg.get_int("a."), Error_Code::Invalid_Arg);
    EXPECT_CONF_CODE(cfg.get_string("a."), Error_Code::Invalid_Arg);
    EXPECT_CONF_CODE(cfg.at("a."), Error_Code::Invalid_Arg);
}

TEST_F(DottedPath, DoubleDotRaisesInvalidArg) {
    // "a..b" splits into an empty middle segment.
    EXPECT_CONF_CODE(cfg.get_int("a..b"), Error_Code::Invalid_Arg);
    EXPECT_CONF_CODE(cfg.get_string("a..b"), Error_Code::Invalid_Arg);
    EXPECT_CONF_CODE(cfg.at("a..b"), Error_Code::Invalid_Arg);
    EXPECT_CONF_CODE(cfg.get_int_list("a..b"), Error_Code::Invalid_Arg);
}

// Invalid_Arg takes precedence over Key_Not_Found even when the malformed path
// names segments that could never resolve (Req 16.1: checked before traversal).
TEST_F(DottedPath, MalformedPathTakesPrecedenceOverMissingKey) {
    EXPECT_CONF_CODE(cfg.get_int(".missing"), Error_Code::Invalid_Arg);
    EXPECT_CONF_CODE(cfg.get_int("missing."), Error_Code::Invalid_Arg);
    EXPECT_CONF_CODE(cfg.get_int("nope..nope"), Error_Code::Invalid_Arg);
}

// ─── Malformed paths — non-throwing accessors (Requirement 16.2) ─────────────
// The same malformed inputs collapse to nullopt / fallback / false / 0 rather
// than propagating an exception.

TEST_F(DottedPath, MalformedPathYieldsNulloptAndFalseForNonThrowing) {
    const char* malformed[] = {"", ".a", "a.", "a..b"};
    for (const char* path : malformed) {
        EXPECT_FALSE(cfg.try_int(path).has_value()) << "path=[" << path << "]";
        EXPECT_FALSE(cfg.try_string(path).has_value()) << "path=[" << path << "]";
        EXPECT_FALSE(cfg.has(path)) << "path=[" << path << "]";
        EXPECT_FALSE(cfg.is_map(path)) << "path=[" << path << "]";
        EXPECT_FALSE(cfg.is_sequence(path)) << "path=[" << path << "]";
        EXPECT_EQ(cfg.size(path), 0u) << "path=[" << path << "]";
        EXPECT_EQ(cfg.get_or<int>(path, -1), -1) << "path=[" << path << "]";
    }
}

// ─── Whitespace-significant keys matched literally (Requirements 3.1, 3.4) ───
// Keys carry their whitespace verbatim: a key with internal spaces and a key
// padded with surrounding spaces each resolve only under their exact spelling.

TEST_F(DottedPath, InternalWhitespaceKeyMatchedLiterally) {
    EXPECT_EQ(cfg.get_int("key with space"), 5);
    EXPECT_TRUE(cfg.has("key with space"));
}

TEST_F(DottedPath, PaddedKeyMatchedLiterallyWithoutTrimming) {
    // The literal key is two leading + two trailing spaces around "pad".
    EXPECT_EQ(cfg.get_int("  pad  "), 9);
    EXPECT_TRUE(cfg.has("  pad  "));
}

TEST_F(DottedPath, TrimmedVariantOfPaddedKeyDoesNotResolve) {
    // No trimming/normalization: the bare and partially-padded spellings are
    // simply absent keys, so they resolve to Key_Not_Found (not Invalid_Arg).
    EXPECT_CONF_CODE(cfg.get_int("pad"), Error_Code::Key_Not_Found);
    EXPECT_CONF_CODE(cfg.get_int(" pad "), Error_Code::Key_Not_Found);
    EXPECT_FALSE(cfg.has("pad"));
    EXPECT_FALSE(cfg.try_int("pad").has_value());
}

TEST_F(DottedPath, CollapsedInternalWhitespaceKeyDoesNotResolve) {
    // Collapsing the internal spaces yields a different, absent key.
    EXPECT_CONF_CODE(cfg.get_int("keywithspace"), Error_Code::Key_Not_Found);
    EXPECT_FALSE(cfg.has("key  with  space"));
}

} // namespace
