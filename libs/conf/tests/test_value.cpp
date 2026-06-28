// ─── CONF Value Unit Tests ───────────────────────────────────────────────────
// Validates Requirements 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8:
//   - Config::at on a defined node returns a Value (10.1)
//   - Config::at on a missing key raises Key_Not_Found (10.2)
//   - Config::at on a malformed path raises Invalid_Arg (10.3)
//   - Value::kind reports the correct Node_Kind (10.4)
//   - Value::size returns child count for map/seq, 0 for scalar/null (10.5)
//   - Throwing Value conversions succeed on valid scalars (10.6)
//   - Throwing Value conversions raise Type_Mismatch on failure (10.7)
//   - Non-throwing Value conversions agree with throwing ones (10.8)
// ──────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>

#include <conf/config.hpp>
#include <conf/error.hpp>
#include <conf/value.hpp>
#include <optional>
#include <string>

namespace {

// YAML fixture covering all node kinds.
constexpr const char *kValueYaml = R"(
int_val: 42
str_val: hello
bool_val: true
map_node:
  a: 1
  b: 2
seq_node:
  - x
  - y
  - z
null_node: ~
)";

class ValueTest : public ::testing::Test {
   protected:
    void SetUp() override {
        cfg_ = conf::Config::from_string(kValueYaml);
    }

    conf::Config cfg_{conf::Config::from_string("")};
};

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════════
// Config::at returns a Value for defined nodes (Req 10.1)
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(ValueTest, At_ScalarNode_ReturnsDefinedValue) {
    auto val = cfg_.at("int_val");
    EXPECT_TRUE(val.is_defined());
}

TEST_F(ValueTest, At_MapNode_ReturnsDefinedValue) {
    auto val = cfg_.at("map_node");
    EXPECT_TRUE(val.is_defined());
}

TEST_F(ValueTest, At_SequenceNode_ReturnsDefinedValue) {
    auto val = cfg_.at("seq_node");
    EXPECT_TRUE(val.is_defined());
}

TEST_F(ValueTest, At_NullNode_ReturnsDefinedValue) {
    auto val = cfg_.at("null_node");
    EXPECT_TRUE(val.is_defined());
}

// ═══════════════════════════════════════════════════════════════════════════════
// Value::kind reports the correct Node_Kind (Req 10.4)
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(ValueTest, Kind_IntScalar_ReportsScalar) {
    auto val = cfg_.at("int_val");
    EXPECT_EQ(val.kind(), conf::Node_Kind::Scalar);
}

TEST_F(ValueTest, Kind_StringScalar_ReportsScalar) {
    auto val = cfg_.at("str_val");
    EXPECT_EQ(val.kind(), conf::Node_Kind::Scalar);
}

TEST_F(ValueTest, Kind_MapNode_ReportsMap) {
    auto val = cfg_.at("map_node");
    EXPECT_EQ(val.kind(), conf::Node_Kind::Map);
}

TEST_F(ValueTest, Kind_SequenceNode_ReportsSequence) {
    auto val = cfg_.at("seq_node");
    EXPECT_EQ(val.kind(), conf::Node_Kind::Sequence);
}

TEST_F(ValueTest, Kind_NullNode_ReportsNull) {
    auto val = cfg_.at("null_node");
    EXPECT_EQ(val.kind(), conf::Node_Kind::Null);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Value::size returns correct child counts (Req 10.5)
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(ValueTest, Size_ScalarNode_ReturnsZero) {
    auto val = cfg_.at("int_val");
    EXPECT_EQ(val.size(), 0u);
}

TEST_F(ValueTest, Size_MapNode_ReturnsChildCount) {
    auto val = cfg_.at("map_node");
    EXPECT_EQ(val.size(), 2u);
}

TEST_F(ValueTest, Size_SequenceNode_ReturnsChildCount) {
    auto val = cfg_.at("seq_node");
    EXPECT_EQ(val.size(), 3u);
}

TEST_F(ValueTest, Size_NullNode_ReturnsZero) {
    auto val = cfg_.at("null_node");
    EXPECT_EQ(val.size(), 0u);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Config::at on missing key throws Key_Not_Found (Req 10.2)
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(ValueTest, At_MissingKey_ThrowsKeyNotFound) {
    try {
        (void)cfg_.at("nonexistent_key");
        FAIL() << "Expected Conf_Error to be thrown";
    } catch (const conf::Conf_Error &e) {
        EXPECT_EQ(e.code(), conf::Error_Code::Key_Not_Found);
    }
}

TEST_F(ValueTest, At_MissingNestedKey_ThrowsKeyNotFound) {
    try {
        (void)cfg_.at("map_node.nonexistent");
        FAIL() << "Expected Conf_Error to be thrown";
    } catch (const conf::Conf_Error &e) {
        EXPECT_EQ(e.code(), conf::Error_Code::Key_Not_Found);
    }
}

TEST_F(ValueTest, At_DescendPastScalar_ThrowsKeyNotFound) {
    try {
        (void)cfg_.at("int_val.child");
        FAIL() << "Expected Conf_Error to be thrown";
    } catch (const conf::Conf_Error &e) {
        EXPECT_EQ(e.code(), conf::Error_Code::Key_Not_Found);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Config::at on malformed path throws Invalid_Arg (Req 10.3)
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(ValueTest, At_EmptyPath_ThrowsInvalidArg) {
    try {
        (void)cfg_.at("");
        FAIL() << "Expected Conf_Error to be thrown";
    } catch (const conf::Conf_Error &e) {
        EXPECT_EQ(e.code(), conf::Error_Code::Invalid_Arg);
    }
}

TEST_F(ValueTest, At_LeadingDot_ThrowsInvalidArg) {
    try {
        (void)cfg_.at(".int_val");
        FAIL() << "Expected Conf_Error to be thrown";
    } catch (const conf::Conf_Error &e) {
        EXPECT_EQ(e.code(), conf::Error_Code::Invalid_Arg);
    }
}

TEST_F(ValueTest, At_TrailingDot_ThrowsInvalidArg) {
    try {
        (void)cfg_.at("int_val.");
        FAIL() << "Expected Conf_Error to be thrown";
    } catch (const conf::Conf_Error &e) {
        EXPECT_EQ(e.code(), conf::Error_Code::Invalid_Arg);
    }
}

TEST_F(ValueTest, At_ConsecutiveDots_ThrowsInvalidArg) {
    try {
        (void)cfg_.at("map_node..a");
        FAIL() << "Expected Conf_Error to be thrown";
    } catch (const conf::Conf_Error &e) {
        EXPECT_EQ(e.code(), conf::Error_Code::Invalid_Arg);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Throwing Value conversions succeed on valid scalars (Req 10.6)
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(ValueTest, AsInt_IntScalar_ReturnsValue) {
    auto val = cfg_.at("int_val");
    EXPECT_EQ(val.as_int(), 42);
}

TEST_F(ValueTest, AsString_StringScalar_ReturnsValue) {
    auto val = cfg_.at("str_val");
    EXPECT_EQ(val.as_string(), "hello");
}

TEST_F(ValueTest, AsBool_BoolScalar_ReturnsValue) {
    auto val = cfg_.at("bool_val");
    EXPECT_TRUE(val.as_bool());
}

TEST_F(ValueTest, AsString_IntScalar_ReturnsStringRepresentation) {
    // Every scalar is a valid representation for as_string (Req 10.6)
    auto val = cfg_.at("int_val");
    EXPECT_EQ(val.as_string(), "42");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Throwing Value conversions raise Type_Mismatch on failure (Req 10.7)
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(ValueTest, AsInt_NonNumericScalar_ThrowsTypeMismatch) {
    auto val = cfg_.at("str_val");
    try {
        (void)val.as_int();
        FAIL() << "Expected Conf_Error to be thrown";
    } catch (const conf::Conf_Error &e) {
        EXPECT_EQ(e.code(), conf::Error_Code::Type_Mismatch);
    }
}

TEST_F(ValueTest, AsInt_MapNode_ThrowsTypeMismatch) {
    auto val = cfg_.at("map_node");
    try {
        (void)val.as_int();
        FAIL() << "Expected Conf_Error to be thrown";
    } catch (const conf::Conf_Error &e) {
        EXPECT_EQ(e.code(), conf::Error_Code::Type_Mismatch);
    }
}

TEST_F(ValueTest, AsDouble_SequenceNode_ThrowsTypeMismatch) {
    auto val = cfg_.at("seq_node");
    try {
        (void)val.as_double();
        FAIL() << "Expected Conf_Error to be thrown";
    } catch (const conf::Conf_Error &e) {
        EXPECT_EQ(e.code(), conf::Error_Code::Type_Mismatch);
    }
}

TEST_F(ValueTest, AsBool_NonBoolScalar_ThrowsTypeMismatch) {
    auto val = cfg_.at("str_val");
    try {
        (void)val.as_bool();
        FAIL() << "Expected Conf_Error to be thrown";
    } catch (const conf::Conf_Error &e) {
        EXPECT_EQ(e.code(), conf::Error_Code::Type_Mismatch);
    }
}

TEST_F(ValueTest, AsString_NullNode_ThrowsTypeMismatch) {
    auto val = cfg_.at("null_node");
    try {
        (void)val.as_string();
        FAIL() << "Expected Conf_Error to be thrown";
    } catch (const conf::Conf_Error &e) {
        EXPECT_EQ(e.code(), conf::Error_Code::Type_Mismatch);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Non-throwing Value conversions (Req 10.8)
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(ValueTest, TryInt_IntScalar_ReturnsEngagedOptional) {
    auto val = cfg_.at("int_val");
    auto result = val.try_int();
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 42);
}

TEST_F(ValueTest, TryInt_NonNumericScalar_ReturnsNullopt) {
    auto val = cfg_.at("str_val");
    EXPECT_EQ(val.try_int(), std::nullopt);
}

TEST_F(ValueTest, TryString_StringScalar_ReturnsEngagedOptional) {
    auto val = cfg_.at("str_val");
    auto result = val.try_string();
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), "hello");
}

TEST_F(ValueTest, TryInt_MapNode_ReturnsNullopt) {
    auto val = cfg_.at("map_node");
    EXPECT_EQ(val.try_int(), std::nullopt);
}

TEST_F(ValueTest, TryDouble_SequenceNode_ReturnsNullopt) {
    auto val = cfg_.at("seq_node");
    EXPECT_EQ(val.try_double(), std::nullopt);
}

// ═══════════════════════════════════════════════════════════════════════════════
// try_* and as_* agree: both succeed or both fail (Req 10.8)
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(ValueTest, TryIntAndAsInt_AgreeOnSuccess) {
    auto val = cfg_.at("int_val");
    auto try_result = val.try_int();
    ASSERT_TRUE(try_result.has_value());
    EXPECT_EQ(try_result.value(), val.as_int());
}

TEST_F(ValueTest, TryStringAndAsString_AgreeOnSuccess) {
    auto val = cfg_.at("str_val");
    auto try_result = val.try_string();
    ASSERT_TRUE(try_result.has_value());
    EXPECT_EQ(try_result.value(), val.as_string());
}

TEST_F(ValueTest, TryIntAndAsInt_AgreeOnFailure) {
    auto val = cfg_.at("str_val");
    // try_int returns nullopt
    EXPECT_EQ(val.try_int(), std::nullopt);
    // as_int throws
    EXPECT_THROW(val.as_int(), conf::Conf_Error);
}

TEST_F(ValueTest, TryBoolAndAsBool_AgreeOnFailure) {
    auto val = cfg_.at("str_val");
    // try_bool returns nullopt
    EXPECT_EQ(val.try_bool(), std::nullopt);
    // as_bool throws
    EXPECT_THROW(val.as_bool(), conf::Conf_Error);
}

TEST_F(ValueTest, TryDoubleAndAsDouble_AgreeOnFailureForMap) {
    auto val = cfg_.at("map_node");
    // try_double returns nullopt
    EXPECT_EQ(val.try_double(), std::nullopt);
    // as_double throws
    EXPECT_THROW(val.as_double(), conf::Conf_Error);
}
