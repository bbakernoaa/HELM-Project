// ─── CONF Missing-Key Unit Tests ─────────────────────────────────────────────
// Validates Requirements 13.1, 13.2 (and implicitly 3.5):
//   - Missing leaf, missing intermediate, out-of-range sequence index,
//     non-integer sequence index, and descend-past-scalar each raise
//     Key_Not_Found for throwing accessors.
//   - The same scenarios yield nullopt for try_* non-throwing accessors.
//   - get_or returns its fallback on missing keys.
//   - has() returns false for all missing paths.
// ──────────────────────────────────────────────────────────────────────────────

#include <conf/config.hpp>
#include <conf/error.hpp>

#include <gtest/gtest.h>
#include <optional>
#include <string>

namespace {

// ─── Test fixture with a shared YAML document ────────────────────────────────

class MissingKeysTest : public ::testing::Test {
protected:
    void SetUp() override {
        const std::string yaml = R"(
model:
  physics:
    layers: 42
  grid:
    resolution: [0.25, 0.5, 1.0]
)";
        cfg_ = conf::Config::from_string(yaml);
    }

    conf::Config cfg_{conf::Config::from_string("")};
};

// ═══════════════════════════════════════════════════════════════════════════════
// Requirement 13.1: Throwing accessors raise Key_Not_Found
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(MissingKeysTest, MissingLeaf_ThrowsKeyNotFound) {
    // A leaf key that does not exist under an existing parent
    try {
        cfg_.get_int("model.physics.missing");
        FAIL() << "Expected Conf_Error to be thrown";
    } catch (const conf::Conf_Error& e) {
        EXPECT_EQ(e.code(), conf::Error_Code::Key_Not_Found);
    }
}

TEST_F(MissingKeysTest, MissingIntermediate_ThrowsKeyNotFound) {
    // An intermediate path segment that does not exist
    try {
        cfg_.get_int("model.nonexistent.layers");
        FAIL() << "Expected Conf_Error to be thrown";
    } catch (const conf::Conf_Error& e) {
        EXPECT_EQ(e.code(), conf::Error_Code::Key_Not_Found);
    }
}

TEST_F(MissingKeysTest, OutOfRangeSequenceIndex_ThrowsKeyNotFound) {
    // A sequence index beyond the array bounds (only 3 elements: 0, 1, 2)
    try {
        cfg_.get_double("model.grid.resolution.5");
        FAIL() << "Expected Conf_Error to be thrown";
    } catch (const conf::Conf_Error& e) {
        EXPECT_EQ(e.code(), conf::Error_Code::Key_Not_Found);
    }
}

TEST_F(MissingKeysTest, DescendPastScalar_ThrowsKeyNotFound) {
    // Attempting to descend past a scalar node (layers is int 42)
    try {
        cfg_.get_int("model.physics.layers.deeper");
        FAIL() << "Expected Conf_Error to be thrown";
    } catch (const conf::Conf_Error& e) {
        EXPECT_EQ(e.code(), conf::Error_Code::Key_Not_Found);
    }
}

TEST_F(MissingKeysTest, NonIntegerSequenceIndex_ThrowsKeyNotFound) {
    // A non-integer segment used as a sequence index
    try {
        cfg_.get_double("model.grid.resolution.abc");
        FAIL() << "Expected Conf_Error to be thrown";
    } catch (const conf::Conf_Error& e) {
        EXPECT_EQ(e.code(), conf::Error_Code::Key_Not_Found);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Requirement 13.2: Non-throwing accessors return nullopt
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(MissingKeysTest, MissingLeaf_TryReturnsNullopt) {
    EXPECT_EQ(cfg_.try_int("model.physics.missing"), std::nullopt);
}

TEST_F(MissingKeysTest, MissingIntermediate_TryReturnsNullopt) {
    EXPECT_EQ(cfg_.try_int("model.nonexistent.layers"), std::nullopt);
}

TEST_F(MissingKeysTest, OutOfRangeSequenceIndex_TryReturnsNullopt) {
    EXPECT_EQ(cfg_.try_double("model.grid.resolution.5"), std::nullopt);
}

TEST_F(MissingKeysTest, DescendPastScalar_TryReturnsNullopt) {
    EXPECT_EQ(cfg_.try_int("model.physics.layers.deeper"), std::nullopt);
}

TEST_F(MissingKeysTest, NonIntegerSequenceIndex_TryReturnsNullopt) {
    EXPECT_EQ(cfg_.try_double("model.grid.resolution.abc"), std::nullopt);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Requirement 13.2: has() returns false for all missing paths
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(MissingKeysTest, MissingLeaf_HasReturnsFalse) {
    EXPECT_FALSE(cfg_.has("model.physics.missing"));
}

TEST_F(MissingKeysTest, MissingIntermediate_HasReturnsFalse) {
    EXPECT_FALSE(cfg_.has("model.nonexistent.layers"));
}

TEST_F(MissingKeysTest, OutOfRangeSequenceIndex_HasReturnsFalse) {
    EXPECT_FALSE(cfg_.has("model.grid.resolution.5"));
}

TEST_F(MissingKeysTest, DescendPastScalar_HasReturnsFalse) {
    EXPECT_FALSE(cfg_.has("model.physics.layers.deeper"));
}

TEST_F(MissingKeysTest, NonIntegerSequenceIndex_HasReturnsFalse) {
    EXPECT_FALSE(cfg_.has("model.grid.resolution.abc"));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Requirement 13.2: get_or returns fallback for all missing paths
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(MissingKeysTest, MissingLeaf_GetOrReturnsFallback) {
    EXPECT_EQ(cfg_.get_or<int>("model.physics.missing", -1), -1);
}

TEST_F(MissingKeysTest, MissingIntermediate_GetOrReturnsFallback) {
    EXPECT_EQ(cfg_.get_or<int>("model.nonexistent.layers", -1), -1);
}

TEST_F(MissingKeysTest, OutOfRangeSequenceIndex_GetOrReturnsFallback) {
    EXPECT_DOUBLE_EQ(cfg_.get_or<double>("model.grid.resolution.5", 9.99), 9.99);
}

TEST_F(MissingKeysTest, DescendPastScalar_GetOrReturnsFallback) {
    EXPECT_EQ(cfg_.get_or<int>("model.physics.layers.deeper", -1), -1);
}

TEST_F(MissingKeysTest, NonIntegerSequenceIndex_GetOrReturnsFallback) {
    EXPECT_DOUBLE_EQ(cfg_.get_or<double>("model.grid.resolution.abc", 9.99), 9.99);
}

} // namespace
