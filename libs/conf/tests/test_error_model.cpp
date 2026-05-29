// ─── CONF Error Model Unit Tests ─────────────────────────────────────────────
// Verifies the stable error taxonomy shared by the C++ core and the C bridge:
//   * Each Error_Code enumerator holds its exact, ABI-stable integer value.
//   * Conf_Error carries exactly the Error_Code supplied at construction and
//     exposes it via code(), alongside a non-empty diagnostic what() message.
//
// Feature: conf-config-parser
// Requirements: 11.1, 11.2, 11.3
// ─────────────────────────────────────────────────────────────────────────────

#include <conf/error.hpp>

#include <gtest/gtest.h>

#include <cstring>
#include <string>

using conf::Conf_Error;
using conf::Error_Code;

// ─── Enumerator Integer Values (Requirements 11.1, 11.2) ─────────────────────
// The nine values are part of the C ABI and must never drift. Each is asserted
// by casting the enumerator to int.

TEST(ErrorCodeValues, SuccessIsZero) {
    EXPECT_EQ(static_cast<int>(Error_Code::Success), 0);
}

TEST(ErrorCodeValues, InvalidArgIsOne) {
    EXPECT_EQ(static_cast<int>(Error_Code::Invalid_Arg), 1);
}

TEST(ErrorCodeValues, FileNotFoundIsTwo) {
    EXPECT_EQ(static_cast<int>(Error_Code::File_Not_Found), 2);
}

TEST(ErrorCodeValues, ParseErrorIsThree) {
    EXPECT_EQ(static_cast<int>(Error_Code::Parse_Error), 3);
}

TEST(ErrorCodeValues, KeyNotFoundIsFour) {
    EXPECT_EQ(static_cast<int>(Error_Code::Key_Not_Found), 4);
}

TEST(ErrorCodeValues, TypeMismatchIsFive) {
    EXPECT_EQ(static_cast<int>(Error_Code::Type_Mismatch), 5);
}

TEST(ErrorCodeValues, BadHandleIsSix) {
    EXPECT_EQ(static_cast<int>(Error_Code::Bad_Handle), 6);
}

TEST(ErrorCodeValues, BufferTooSmallIsSeven) {
    EXPECT_EQ(static_cast<int>(Error_Code::Buffer_Too_Small), 7);
}

TEST(ErrorCodeValues, UnknownIsNinetyNine) {
    EXPECT_EQ(static_cast<int>(Error_Code::Unknown), 99);
}

// ─── Conf_Error::code() Round-Trips the Constructed Value (Req 11.3) ─────────

TEST(ConfError, CodeReturnsValueSuppliedAtConstruction) {
    const Error_Code codes[] = {
        Error_Code::Success,
        Error_Code::Invalid_Arg,
        Error_Code::File_Not_Found,
        Error_Code::Parse_Error,
        Error_Code::Key_Not_Found,
        Error_Code::Type_Mismatch,
        Error_Code::Bad_Handle,
        Error_Code::Buffer_Too_Small,
        Error_Code::Unknown,
    };

    for (Error_Code code : codes) {
        Conf_Error err(code, "diagnostic message");
        EXPECT_EQ(err.code(), code)
            << "code() must return the exact Error_Code passed at construction "
               "(value " << static_cast<int>(code) << ")";
    }
}

// ─── Conf_Error Carries a Non-Empty Diagnostic Message (Req 11.3) ────────────

TEST(ConfError, WhatIsNonEmpty) {
    const Error_Code codes[] = {
        Error_Code::Success,
        Error_Code::Invalid_Arg,
        Error_Code::File_Not_Found,
        Error_Code::Parse_Error,
        Error_Code::Key_Not_Found,
        Error_Code::Type_Mismatch,
        Error_Code::Bad_Handle,
        Error_Code::Buffer_Too_Small,
        Error_Code::Unknown,
    };

    for (Error_Code code : codes) {
        Conf_Error err(code, "config key did not resolve");
        const char* msg = err.what();
        ASSERT_NE(msg, nullptr);
        EXPECT_GT(std::strlen(msg), 0u)
            << "what() must be non-empty for Error_Code value "
            << static_cast<int>(code);
    }
}

TEST(ConfError, WhatPreservesSuppliedMessage) {
    const std::string message = "model.physics.layers: key not found";
    Conf_Error err(Error_Code::Key_Not_Found, message);

    EXPECT_EQ(err.code(), Error_Code::Key_Not_Found);
    EXPECT_EQ(std::string(err.what()), message);
}

// ─── Conf_Error Is Catchable as a std::runtime_error / std::exception ────────
// Confirms the exception integrates with the standard exception hierarchy, which
// the C bridge relies on when translating exceptions to error codes.

TEST(ConfError, IsCatchableAsStdException) {
    try {
        throw Conf_Error(Error_Code::Type_Mismatch, "wrong type");
    } catch (const std::exception& e) {
        EXPECT_GT(std::strlen(e.what()), 0u);
        const auto* conf_err = dynamic_cast<const Conf_Error*>(&e);
        ASSERT_NE(conf_err, nullptr);
        EXPECT_EQ(conf_err->code(), Error_Code::Type_Mismatch);
        return;
    }
    FAIL() << "Conf_Error should have been caught as a std::exception";
}
