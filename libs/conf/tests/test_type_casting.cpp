// ─── CONF Type Casting Unit Tests ────────────────────────────────────────────
// Validates Requirements 5.1, 5.3, 6.1, 6.3, 7.3, 15.1, 15.2:
//   - Throwing typed accessors return correct values for valid conversions
//   - Throwing typed accessors raise Type_Mismatch on unconvertible scalars
//   - Throwing typed accessors raise Type_Mismatch on non-scalar nodes
//   - Non-throwing accessors return nullopt on type mismatch
//   - Non-throwing accessors return engaged optional on valid conversions
//   - Defaulted accessor returns fallback on type mismatch
//   - Defaulted accessor returns converted value on success
// ──────────────────────────────────────────────────────────────────────────────

#include <conf/config.hpp>
#include <conf/error.hpp>

#include <gtest/gtest.h>
#include <cmath>
#include <optional>
#include <string>

namespace {

// YAML fixture covering scalars of each type plus non-scalar nodes.
constexpr const char* kTypeCastingYaml = R"(
int_val: 42
double_val: 3.14159
bool_val: true
string_val: "hello world"
text_val: not_a_number
nested:
  key: value
list: [1, 2, 3]
)";

class TypeCasting : public ::testing::Test {
protected:
    void SetUp() override {
        cfg_ = conf::Config::from_string(kTypeCastingYaml);
    }

    conf::Config cfg_{conf::Config::from_string("")};
};

} // namespace

// ═══════════════════════════════════════════════════════════════════════════════
// Happy Path: Throwing accessors return expected values (Req 5.1)
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(TypeCasting, GetInt_ValidScalar_ReturnsValue) {
    EXPECT_EQ(cfg_.get_int("int_val"), 42);
}

TEST_F(TypeCasting, GetDouble_ValidScalar_ReturnsApproximateValue) {
    EXPECT_NEAR(cfg_.get_double("double_val"), 3.14159, 1e-5);
}

TEST_F(TypeCasting, GetBool_TrueScalar_ReturnsTrue) {
    EXPECT_TRUE(cfg_.get_bool("bool_val"));
}

TEST_F(TypeCasting, GetString_QuotedScalar_ReturnsString) {
    EXPECT_EQ(cfg_.get_string("string_val"), "hello world");
}

// get_string accepts any scalar (Req 5.1: every scalar is valid for get_string)
TEST_F(TypeCasting, GetString_NonQuotedScalar_ReturnsText) {
    EXPECT_EQ(cfg_.get_string("text_val"), "not_a_number");
}

TEST_F(TypeCasting, GetString_IntScalar_ReturnsStringRepresentation) {
    // "42" is a valid string representation of the int scalar
    EXPECT_EQ(cfg_.get_string("int_val"), "42");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Type Mismatch: Throwing accessors raise Type_Mismatch (Req 5.3, 15.1)
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(TypeCasting, GetInt_NonNumericScalar_ThrowsTypeMismatch) {
    try {
        (void)cfg_.get_int("text_val");
        FAIL() << "Expected Conf_Error to be thrown";
    } catch (const conf::Conf_Error& e) {
        EXPECT_EQ(e.code(), conf::Error_Code::Type_Mismatch);
    }
}

TEST_F(TypeCasting, GetDouble_NonNumericScalar_ThrowsTypeMismatch) {
    try {
        (void)cfg_.get_double("text_val");
        FAIL() << "Expected Conf_Error to be thrown";
    } catch (const conf::Conf_Error& e) {
        EXPECT_EQ(e.code(), conf::Error_Code::Type_Mismatch);
    }
}

TEST_F(TypeCasting, GetBool_NonBoolScalar_ThrowsTypeMismatch) {
    try {
        (void)cfg_.get_bool("text_val");
        FAIL() << "Expected Conf_Error to be thrown";
    } catch (const conf::Conf_Error& e) {
        EXPECT_EQ(e.code(), conf::Error_Code::Type_Mismatch);
    }
}

// Non-scalar nodes: map/sequence requested as scalar types (Req 5.3, 15.1)

TEST_F(TypeCasting, GetInt_MapNode_ThrowsTypeMismatch) {
    try {
        (void)cfg_.get_int("nested");
        FAIL() << "Expected Conf_Error to be thrown";
    } catch (const conf::Conf_Error& e) {
        EXPECT_EQ(e.code(), conf::Error_Code::Type_Mismatch);
    }
}

TEST_F(TypeCasting, GetInt_SequenceNode_ThrowsTypeMismatch) {
    try {
        (void)cfg_.get_int("list");
        FAIL() << "Expected Conf_Error to be thrown";
    } catch (const conf::Conf_Error& e) {
        EXPECT_EQ(e.code(), conf::Error_Code::Type_Mismatch);
    }
}

TEST_F(TypeCasting, GetDouble_MapNode_ThrowsTypeMismatch) {
    try {
        (void)cfg_.get_double("nested");
        FAIL() << "Expected Conf_Error to be thrown";
    } catch (const conf::Conf_Error& e) {
        EXPECT_EQ(e.code(), conf::Error_Code::Type_Mismatch);
    }
}

TEST_F(TypeCasting, GetBool_MapNode_ThrowsTypeMismatch) {
    try {
        (void)cfg_.get_bool("nested");
        FAIL() << "Expected Conf_Error to be thrown";
    } catch (const conf::Conf_Error& e) {
        EXPECT_EQ(e.code(), conf::Error_Code::Type_Mismatch);
    }
}

TEST_F(TypeCasting, GetString_MapNode_ThrowsTypeMismatch) {
    try {
        (void)cfg_.get_string("nested");
        FAIL() << "Expected Conf_Error to be thrown";
    } catch (const conf::Conf_Error& e) {
        EXPECT_EQ(e.code(), conf::Error_Code::Type_Mismatch);
    }
}

TEST_F(TypeCasting, GetString_SequenceNode_ThrowsTypeMismatch) {
    try {
        (void)cfg_.get_string("list");
        FAIL() << "Expected Conf_Error to be thrown";
    } catch (const conf::Conf_Error& e) {
        EXPECT_EQ(e.code(), conf::Error_Code::Type_Mismatch);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Non-throwing: try_* returns nullopt on type mismatch (Req 6.1, 6.3, 15.2)
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(TypeCasting, TryInt_NonNumericScalar_ReturnsNullopt) {
    EXPECT_EQ(cfg_.try_int("text_val"), std::nullopt);
}

TEST_F(TypeCasting, TryDouble_NonNumericScalar_ReturnsNullopt) {
    EXPECT_EQ(cfg_.try_double("text_val"), std::nullopt);
}

TEST_F(TypeCasting, TryBool_ArbitraryScalar_ReturnsNullopt) {
    // "not_a_number" is not a valid bool ("true"/"false"/"yes"/"no")
    EXPECT_EQ(cfg_.try_bool("text_val"), std::nullopt);
}

TEST_F(TypeCasting, TryInt_MapNode_ReturnsNullopt) {
    EXPECT_EQ(cfg_.try_int("nested"), std::nullopt);
}

TEST_F(TypeCasting, TryDouble_SequenceNode_ReturnsNullopt) {
    EXPECT_EQ(cfg_.try_double("list"), std::nullopt);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Non-throwing: try_* returns engaged optional on success (Req 6.1)
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(TypeCasting, TryInt_ValidScalar_ReturnsEngagedOptional) {
    auto result = cfg_.try_int("int_val");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), 42);
}

TEST_F(TypeCasting, TryDouble_ValidScalar_ReturnsEngagedOptional) {
    auto result = cfg_.try_double("double_val");
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result.value(), 3.14159, 1e-5);
}

TEST_F(TypeCasting, TryBool_ValidScalar_ReturnsEngagedOptional) {
    auto result = cfg_.try_bool("bool_val");
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result.value());
}

TEST_F(TypeCasting, TryString_ValidScalar_ReturnsEngagedOptional) {
    auto result = cfg_.try_string("string_val");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), "hello world");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Defaulted: get_or returns converted value on success (Req 7.3)
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(TypeCasting, GetOr_Int_ValidScalar_ReturnsConvertedValue) {
    EXPECT_EQ(cfg_.get_or("int_val", 99), 42);
}

TEST_F(TypeCasting, GetOr_Double_ValidScalar_ReturnsConvertedValue) {
    EXPECT_NEAR(cfg_.get_or("double_val", 0.0), 3.14159, 1e-5);
}

TEST_F(TypeCasting, GetOr_Bool_ValidScalar_ReturnsConvertedValue) {
    EXPECT_TRUE(cfg_.get_or("bool_val", false));
}

TEST_F(TypeCasting, GetOr_String_ValidScalar_ReturnsConvertedValue) {
    EXPECT_EQ(cfg_.get_or<std::string>("string_val", std::string("default")),
              "hello world");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Defaulted: get_or returns fallback on type mismatch (Req 7.3, 15.2)
// ═══════════════════════════════════════════════════════════════════════════════

TEST_F(TypeCasting, GetOr_Int_NonNumericScalar_ReturnsFallback) {
    EXPECT_EQ(cfg_.get_or("text_val", 99), 99);
}

TEST_F(TypeCasting, GetOr_Double_NonNumericScalar_ReturnsFallback) {
    EXPECT_DOUBLE_EQ(cfg_.get_or("text_val", 1.5), 1.5);
}

TEST_F(TypeCasting, GetOr_Bool_ArbitraryScalar_ReturnsFallback) {
    // "not_a_number" is not a valid bool, so fallback should be returned
    EXPECT_FALSE(cfg_.get_or("text_val", false));
}

TEST_F(TypeCasting, GetOr_Int_MapNode_ReturnsFallback) {
    EXPECT_EQ(cfg_.get_or("nested", -1), -1);
}

TEST_F(TypeCasting, GetOr_Double_SequenceNode_ReturnsFallback) {
    EXPECT_DOUBLE_EQ(cfg_.get_or("list", 7.77), 7.77);
}
