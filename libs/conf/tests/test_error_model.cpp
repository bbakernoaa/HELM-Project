// ─── CONF Error Model Unit Tests ─────────────────────────────────────────────
// Validates Requirements 11.1, 11.2, 11.3:
//   - Error_Code enumerators have the exact specified integer values
//   - Conf_Error carries the code supplied at construction
//   - Conf_Error::what() is non-empty and contains the diagnostic message
//   - Conf_Error inherits from std::runtime_error
// ──────────────────────────────────────────────────────────────────────────────

#include <conf/error.hpp>

#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

// ─── Requirement 11.1: Exact integer values for all nine Error_Code enumerators

TEST(ErrorCode, SuccessIsZero) {
    EXPECT_EQ(static_cast<int>(conf::Error_Code::Success), 0);
}

TEST(ErrorCode, InvalidArgIsOne) {
    EXPECT_EQ(static_cast<int>(conf::Error_Code::Invalid_Arg), 1);
}

TEST(ErrorCode, FileNotFoundIsTwo) {
    EXPECT_EQ(static_cast<int>(conf::Error_Code::File_Not_Found), 2);
}

TEST(ErrorCode, ParseErrorIsThree) {
    EXPECT_EQ(static_cast<int>(conf::Error_Code::Parse_Error), 3);
}

TEST(ErrorCode, KeyNotFoundIsFour) {
    EXPECT_EQ(static_cast<int>(conf::Error_Code::Key_Not_Found), 4);
}

TEST(ErrorCode, TypeMismatchIsFive) {
    EXPECT_EQ(static_cast<int>(conf::Error_Code::Type_Mismatch), 5);
}

TEST(ErrorCode, BadHandleIsSix) {
    EXPECT_EQ(static_cast<int>(conf::Error_Code::Bad_Handle), 6);
}

TEST(ErrorCode, BufferTooSmallIsSeven) {
    EXPECT_EQ(static_cast<int>(conf::Error_Code::Buffer_Too_Small), 7);
}

TEST(ErrorCode, UnknownIsNinetyNine) {
    EXPECT_EQ(static_cast<int>(conf::Error_Code::Unknown), 99);
}

// ─── Requirement 11.3: Conf_Error carries one Error_Code and a non-empty message

TEST(ConfError, CodeReturnsSuppliedValue) {
    const conf::Conf_Error err(conf::Error_Code::Parse_Error, "syntax error at line 5");
    EXPECT_EQ(err.code(), conf::Error_Code::Parse_Error);
}

TEST(ConfError, CodeReturnsEachEnumerator) {
    // Verify code() round-trips correctly for every enumerator
    struct Case {
        conf::Error_Code code;
        const char* msg;
    };

    const Case cases[] = {
        {conf::Error_Code::Success,          "ok"},
        {conf::Error_Code::Invalid_Arg,      "bad arg"},
        {conf::Error_Code::File_Not_Found,   "no such file"},
        {conf::Error_Code::Parse_Error,      "malformed yaml"},
        {conf::Error_Code::Key_Not_Found,    "missing key"},
        {conf::Error_Code::Type_Mismatch,    "wrong type"},
        {conf::Error_Code::Bad_Handle,       "stale handle"},
        {conf::Error_Code::Buffer_Too_Small, "buffer overflow"},
        {conf::Error_Code::Unknown,          "unknown failure"},
    };

    for (const auto& c : cases) {
        const conf::Conf_Error err(c.code, c.msg);
        EXPECT_EQ(err.code(), c.code)
            << "Failed for message: " << c.msg;
    }
}

TEST(ConfError, WhatIsNonEmpty) {
    const conf::Conf_Error err(conf::Error_Code::Key_Not_Found, "path.does.not.exist");
    const std::string what_str(err.what());
    EXPECT_FALSE(what_str.empty());
}

TEST(ConfError, WhatContainsMessage) {
    const std::string msg = "configuration file not found: /tmp/missing.yaml";
    const conf::Conf_Error err(conf::Error_Code::File_Not_Found, msg);
    const std::string what_str(err.what());
    EXPECT_NE(what_str.find(msg), std::string::npos)
        << "what() should contain the diagnostic message";
}

// ─── Requirement 11.3 (inheritance): Conf_Error inherits from std::runtime_error

TEST(ConfError, InheritsFromRuntimeError) {
    const conf::Conf_Error err(conf::Error_Code::Type_Mismatch, "expected int, got string");

    // Verify the exception can be caught as std::runtime_error
    bool caught_as_runtime_error = false;
    try {
        throw err;
    } catch (const std::runtime_error& e) {
        caught_as_runtime_error = true;
        // The what() message should still be accessible via the base class
        const std::string what_str(e.what());
        EXPECT_FALSE(what_str.empty());
        EXPECT_NE(what_str.find("expected int, got string"), std::string::npos);
    }
    EXPECT_TRUE(caught_as_runtime_error);
}

TEST(ConfError, CatchableAsStdException) {
    const conf::Conf_Error err(conf::Error_Code::Unknown, "unexpected");

    // Verify it's also catchable as std::exception (grandparent)
    bool caught_as_exception = false;
    try {
        throw err;
    } catch (const std::exception& e) {
        caught_as_exception = true;
        EXPECT_NE(std::string(e.what()).size(), 0u);
    }
    EXPECT_TRUE(caught_as_exception);
}
