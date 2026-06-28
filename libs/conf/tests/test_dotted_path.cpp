// ─── CONF Dotted-Path Resolver Unit Tests ────────────────────────────────────
// Validates Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 16.1, 16.2:
//   - Deep nesting (a.b.c.d) resolves correctly through nested maps (3.1, 3.3).
//   - Sequence index (list.0.x) navigates sequence elements by integer index (3.2).
//   - Empty path / leading-dot / trailing-dot / double-dot raise Invalid_Arg (3.6, 16.1).
//   - Non-throwing accessors return nullopt/fallback/false for malformed paths (16.2).
//   - Whitespace-significant keys matched literally (3.4).
//   - Absent keys / out-of-range indices raise Key_Not_Found (3.5).
// ──────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>

#include <conf/config.hpp>
#include <conf/error.hpp>
#include <optional>
#include <string>

namespace {

// ═══════════════════════════════════════════════════════════════════════════════
// Requirement 3.1, 3.3: Deep nesting a.b.c.d resolves correctly
// ═══════════════════════════════════════════════════════════════════════════════

TEST(DottedPathTest, DeepNesting_FourLevels) {
    auto cfg = conf::Config::from_string(R"(
a:
  b:
    c:
      d: 99
)");
    EXPECT_EQ(cfg.get_int("a.b.c.d"), 99);
}

TEST(DottedPathTest, DeepNesting_IntermediateHasReturnsTrue) {
    auto cfg = conf::Config::from_string(R"(
a:
  b:
    c:
      d: 99
)");
    // has() confirms intermediate map nodes are reachable
    EXPECT_TRUE(cfg.has("a.b.c"));
}

TEST(DottedPathTest, DeepNesting_LeafHasReturnsTrue) {
    auto cfg = conf::Config::from_string(R"(
a:
  b:
    c:
      d: 99
)");
    EXPECT_TRUE(cfg.has("a.b.c.d"));
}

TEST(DottedPathTest, DeepNesting_StringAtDepth) {
    auto cfg = conf::Config::from_string(R"(
x:
  y:
    z:
      w: deep_value
)");
    EXPECT_EQ(cfg.get_string("x.y.z.w"), "deep_value");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Requirement 3.2: Sequence indexing — list.0.x, list.1.x
// ═══════════════════════════════════════════════════════════════════════════════

TEST(DottedPathTest, SequenceIndex_FirstElement) {
    auto cfg = conf::Config::from_string(R"(
list:
  - x: hello
  - x: world
)");
    EXPECT_EQ(cfg.get_string("list.0.x"), "hello");
}

TEST(DottedPathTest, SequenceIndex_SecondElement) {
    auto cfg = conf::Config::from_string(R"(
list:
  - x: hello
  - x: world
)");
    EXPECT_EQ(cfg.get_string("list.1.x"), "world");
}

TEST(DottedPathTest, SequenceIndex_NumericValue) {
    auto cfg = conf::Config::from_string(R"(
items:
  - value: 10
  - value: 20
  - value: 30
)");
    EXPECT_EQ(cfg.get_int("items.0.value"), 10);
    EXPECT_EQ(cfg.get_int("items.2.value"), 30);
}

TEST(DottedPathTest, SequenceIndex_NestedInMap) {
    // Sequence nested inside maps: map.seq.0
    auto cfg = conf::Config::from_string(R"(
model:
  grid:
    resolution: [0.25, 0.5, 1.0]
)");
    EXPECT_DOUBLE_EQ(cfg.get_double("model.grid.resolution.0"), 0.25);
    EXPECT_DOUBLE_EQ(cfg.get_double("model.grid.resolution.2"), 1.0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Requirement 3.4: Whitespace-significant keys matched literally
// ═══════════════════════════════════════════════════════════════════════════════

TEST(DottedPathTest, WhitespaceKey_LeadingSpace) {
    auto cfg = conf::Config::from_string(R"(
" leading": preserved_leading
)");
    EXPECT_EQ(cfg.get_string(" leading"), "preserved_leading");
}

TEST(DottedPathTest, WhitespaceKey_TrailingSpace) {
    auto cfg = conf::Config::from_string(R"(
"trailing ": preserved_trailing
)");
    EXPECT_EQ(cfg.get_string("trailing "), "preserved_trailing");
}

TEST(DottedPathTest, WhitespaceKey_InternalSpaces) {
    auto cfg = conf::Config::from_string(R"(
"key with spaces": spaced_value
)");
    EXPECT_EQ(cfg.get_string("key with spaces"), "spaced_value");
}

TEST(DottedPathTest, WhitespaceKey_BothSidesPadded) {
    auto cfg = conf::Config::from_string(R"(
"  padded  ": padded_value
)");
    EXPECT_EQ(cfg.get_string("  padded  "), "padded_value");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Requirement 3.6 / 16.1: Malformed paths raise Invalid_Arg (throwing)
// ═══════════════════════════════════════════════════════════════════════════════

class DottedPathInvalidTest : public ::testing::Test {
   protected:
    void SetUp() override {
        cfg_ = conf::Config::from_string("a:\n  b: 1\n");
    }
    conf::Config cfg_{conf::Config::from_string("")};
};

TEST_F(DottedPathInvalidTest, EmptyPath_ThrowsInvalidArg) {
    try {
        (void)cfg_.get_int("");
        FAIL() << "Expected Conf_Error to be thrown";
    } catch (const conf::Conf_Error &e) {
        EXPECT_EQ(e.code(), conf::Error_Code::Invalid_Arg);
    }
}

TEST_F(DottedPathInvalidTest, LeadingDot_ThrowsInvalidArg) {
    try {
        (void)cfg_.get_int(".a.b");
        FAIL() << "Expected Conf_Error to be thrown";
    } catch (const conf::Conf_Error &e) {
        EXPECT_EQ(e.code(), conf::Error_Code::Invalid_Arg);
    }
}

TEST_F(DottedPathInvalidTest, TrailingDot_ThrowsInvalidArg) {
    try {
        (void)cfg_.get_int("a.b.");
        FAIL() << "Expected Conf_Error to be thrown";
    } catch (const conf::Conf_Error &e) {
        EXPECT_EQ(e.code(), conf::Error_Code::Invalid_Arg);
    }
}

TEST_F(DottedPathInvalidTest, ConsecutiveDots_ThrowsInvalidArg) {
    try {
        (void)cfg_.get_int("a..b");
        FAIL() << "Expected Conf_Error to be thrown";
    } catch (const conf::Conf_Error &e) {
        EXPECT_EQ(e.code(), conf::Error_Code::Invalid_Arg);
    }
}

TEST_F(DottedPathInvalidTest, OnlyDot_ThrowsInvalidArg) {
    try {
        (void)cfg_.get_int(".");
        FAIL() << "Expected Conf_Error to be thrown";
    } catch (const conf::Conf_Error &e) {
        EXPECT_EQ(e.code(), conf::Error_Code::Invalid_Arg);
    }
}

TEST_F(DottedPathInvalidTest, MultipleDots_ThrowsInvalidArg) {
    try {
        (void)cfg_.get_int("...");
        FAIL() << "Expected Conf_Error to be thrown";
    } catch (const conf::Conf_Error &e) {
        EXPECT_EQ(e.code(), conf::Error_Code::Invalid_Arg);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Requirement 16.2: Non-throwing accessors return nullopt/false/0/fallback
//                   on malformed paths
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(DottedPathInvalidTest, EmptyPath_TryReturnsNullopt) {
    EXPECT_EQ(cfg_.try_int(""), std::nullopt);
    EXPECT_EQ(cfg_.try_string(""), std::nullopt);
}

TEST_F(DottedPathInvalidTest, LeadingDot_TryReturnsNullopt) {
    EXPECT_EQ(cfg_.try_int(".a.b"), std::nullopt);
}

TEST_F(DottedPathInvalidTest, TrailingDot_TryReturnsNullopt) {
    EXPECT_EQ(cfg_.try_int("a.b."), std::nullopt);
}

TEST_F(DottedPathInvalidTest, ConsecutiveDots_TryReturnsNullopt) {
    EXPECT_EQ(cfg_.try_int("a..b"), std::nullopt);
}

TEST_F(DottedPathInvalidTest, EmptyPath_HasReturnsFalse) {
    EXPECT_FALSE(cfg_.has(""));
}

TEST_F(DottedPathInvalidTest, LeadingDot_HasReturnsFalse) {
    EXPECT_FALSE(cfg_.has(".a"));
}

TEST_F(DottedPathInvalidTest, TrailingDot_HasReturnsFalse) {
    EXPECT_FALSE(cfg_.has("a."));
}

TEST_F(DottedPathInvalidTest, ConsecutiveDots_HasReturnsFalse) {
    EXPECT_FALSE(cfg_.has("a..b"));
}

TEST_F(DottedPathInvalidTest, EmptyPath_SizeReturnsZero) {
    EXPECT_EQ(cfg_.size(""), 0u);
}

TEST_F(DottedPathInvalidTest, MalformedPath_GetOrReturnsFallback) {
    EXPECT_EQ(cfg_.get_or<int>("", -1), -1);
    EXPECT_EQ(cfg_.get_or<int>(".a", -1), -1);
    EXPECT_EQ(cfg_.get_or<int>("a.", -1), -1);
    EXPECT_EQ(cfg_.get_or<int>("a..b", -1), -1);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Requirement 3.5: Key_Not_Found for absent key vs Invalid_Arg distinction
// ═══════════════════════════════════════════════════════════════════════════════

TEST(DottedPathTest, AbsentKey_ThrowsKeyNotFound) {
    auto cfg = conf::Config::from_string(R"(
a:
  b:
    c: 42
)");
    try {
        (void)cfg.get_int("a.b.nonexistent");
        FAIL() << "Expected Conf_Error to be thrown";
    } catch (const conf::Conf_Error &e) {
        EXPECT_EQ(e.code(), conf::Error_Code::Key_Not_Found);
    }
}

TEST(DottedPathTest, OutOfRangeIndex_ThrowsKeyNotFound) {
    auto cfg = conf::Config::from_string(R"(
list:
  - x: 1
  - x: 2
)");
    try {
        (void)cfg.get_string("list.99.x");
        FAIL() << "Expected Conf_Error to be thrown";
    } catch (const conf::Conf_Error &e) {
        EXPECT_EQ(e.code(), conf::Error_Code::Key_Not_Found);
    }
}

TEST(DottedPathTest, DescendPastScalar_ThrowsKeyNotFound) {
    auto cfg = conf::Config::from_string(R"(
a:
  b: 42
)");
    try {
        (void)cfg.get_int("a.b.deeper");
        FAIL() << "Expected Conf_Error to be thrown";
    } catch (const conf::Conf_Error &e) {
        EXPECT_EQ(e.code(), conf::Error_Code::Key_Not_Found);
    }
}

TEST(DottedPathTest, NonDigitSequenceIndex_ThrowsKeyNotFound) {
    auto cfg = conf::Config::from_string(R"(
list:
  - 10
  - 20
)");
    try {
        (void)cfg.get_int("list.abc");
        FAIL() << "Expected Conf_Error to be thrown";
    } catch (const conf::Conf_Error &e) {
        EXPECT_EQ(e.code(), conf::Error_Code::Key_Not_Found);
    }
}

}  // namespace
