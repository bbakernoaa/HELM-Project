// ─── CONF Malformed-YAML Unit Tests ──────────────────────────────────────────
// Verifies Requirement 14 / 32.2: a Config factory fed syntactically malformed
// YAML reports the failure as a typed Conf_Error carrying Error_Code::Parse_Error
// — never by crashing, aborting, or silently succeeding — and the diagnostic
// what() message is non-empty (it carries the backend's line/column text).
//
// yaml-cpp 0.8.0 is fairly permissive: many "looks wrong" documents still parse
// (e.g. plain bad indentation is often tolerated or reinterpreted). The inputs
// below are chosen to genuinely trigger YAML::ParserException in 0.8.0:
//   * an unclosed flow sequence            ("key: [1, 2")
//   * an unclosed flow mapping             ("key: {a: 1")
//   * a tab character used for indentation  (tabs are illegal as YAML indent)
//   * a block-mapping indentation conflict  ("a:\n  - b\n c: oops\n")
//   * a key with two consecutive mapping values on one line
// Every case must surface as Error_Code::Parse_Error with a non-empty message.
//
// Feature: conf-config-parser
// Requirements: 32.2, 14.1, 14.2
// ─────────────────────────────────────────────────────────────────────────────

#include <conf/config.hpp>
#include <conf/error.hpp>

#include <gtest/gtest.h>

#include <cstring>
#include <string>

using conf::Conf_Error;
using conf::Error_Code;

namespace {

// Assert that loading `yaml_text` fails specifically with Parse_Error and a
// non-empty diagnostic, and that the failure is a controlled exception rather
// than a crash/abort. Centralizing the assertions keeps every case consistent.
void expect_parse_error(const std::string& yaml_text, const char* label) {
    try {
        // Discard the returned Config explicitly: from_string is [[nodiscard]],
        // and here we only care that the call throws on malformed input.
        static_cast<void>(conf::Config::from_string(yaml_text));
        FAIL() << "[" << label << "] expected Conf_Error{Parse_Error} but "
                  "from_string returned normally";
    } catch (const Conf_Error& err) {
        EXPECT_EQ(err.code(), Error_Code::Parse_Error)
            << "[" << label << "] malformed YAML must report Parse_Error, got "
            << static_cast<int>(err.code());
        // Requirement 14.2: the message carries the backend's non-empty text.
        ASSERT_NE(err.what(), nullptr) << "[" << label << "]";
        EXPECT_GT(std::strlen(err.what()), 0u)
            << "[" << label << "] Parse_Error message must be non-empty";
    } catch (...) {
        FAIL() << "[" << label << "] expected Conf_Error{Parse_Error} but a "
                  "different exception type escaped";
    }
}

} // namespace

// ─── Unclosed flow collections ───────────────────────────────────────────────

TEST(MalformedYaml, UnclosedFlowSequenceIsParseError) {
    // A flow sequence opened with '[' but never closed: yaml-cpp reaches EOF
    // while still inside the flow context and throws a ParserException.
    expect_parse_error("key: [1, 2", "unclosed-flow-sequence");
}

TEST(MalformedYaml, UnclosedFlowMappingIsParseError) {
    // A flow mapping opened with '{' but never closed.
    expect_parse_error("key: {a: 1", "unclosed-flow-mapping");
}

TEST(MalformedYaml, UnclosedNestedFlowIsParseError) {
    // Nested flow collections with a missing closing bracket.
    expect_parse_error("root: {list: [1, 2, 3}", "unclosed-nested-flow");
}

// ─── Illegal indentation ─────────────────────────────────────────────────────

TEST(MalformedYaml, TabIndentationIsParseError) {
    // YAML forbids the tab character as indentation. A block-mapping child that
    // is indented with a literal '\t' is rejected by the parser.
    expect_parse_error("parent:\n\tchild: value\n", "tab-indentation");
}

TEST(MalformedYaml, BlockMappingIndentationConflictIsParseError) {
    // A key whose value starts as a block sequence, then a sibling key appears
    // at an inconsistent (shallower-but-not-aligned) indent — an indentation
    // conflict the parser cannot resolve. (Mirrors the design's example.)
    expect_parse_error("a:\n  - b\n c: oops\n", "block-indent-conflict");
}

// ─── Structural key/value errors ─────────────────────────────────────────────

TEST(MalformedYaml, MultipleMapValuesOnOneLineIsParseError) {
    // Two ':' value indicators applied to a single key on one line.
    expect_parse_error("key: value1: value2\n", "double-map-value");
}

// ─── Robustness: many malformed inputs in sequence (no crash / no abort) ──────
// Loading a batch of malformed documents one after another must remain a
// controlled, repeatable operation — the process is never left in a bad state.

TEST(MalformedYaml, RepeatedMalformedLoadsDoNotCrash) {
    const char* inputs[] = {
        "key: [1, 2",
        "key: {a: 1",
        "root: {list: [1, 2, 3}",
        "parent:\n\tchild: value\n",
        "a:\n  - b\n c: oops\n",
        "key: value1: value2\n",
    };

    for (const char* input : inputs) {
        EXPECT_THROW(conf::Config::from_string(input), Conf_Error)
            << "input did not raise Conf_Error: " << input;
    }
}
