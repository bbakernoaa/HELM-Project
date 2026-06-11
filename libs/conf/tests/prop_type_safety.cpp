// --- Property-Based Tests: Type-Safety Across Incompatible Requests ----------
// Feature: conf-config-parser
//
// Uses RapidCheck to verify that requesting a value through an incompatible
// typed accessor never returns garbage and never crashes. For a node of type A,
// an incompatible request B yields:
//   - try_B → nullopt
//   - get_B → raises Conf_Error{Type_Mismatch}
//   - get_or → returns the supplied fallback
//
// Property 2: Type-Safety
//   Validates: Requirements 28.1, 28.2, 28.3, 28.4
// -----------------------------------------------------------------------------

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <conf/config.hpp>
#include <conf/error.hpp>

#include <optional>
#include <sstream>
#include <string>

namespace {

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Build a one-key YAML document: "key: <value>"
/// The value is already formatted for YAML (quoted if string, etc.).
std::string make_yaml(const std::string& yaml_value) {
    return "key: " + yaml_value + "\n";
}

/// Generate a random string that is NOT parseable as an integer.
/// Filters out strings that consist solely of optional sign + digits.
rc::Gen<std::string> genNonNumericString() {
    return rc::gen::suchThat(
        rc::gen::container<std::string>(
            rc::gen::inRange(static_cast<char>('a'), static_cast<char>('z' + 1))),
        [](const std::string& s) {
            return !s.empty();
        });
}

/// Generate a random string that is NOT parseable as a boolean.
/// Filters out yaml-cpp recognized booleans: true/false/yes/no/on/off/y/n.
rc::Gen<std::string> genNonBoolString() {
    return rc::gen::suchThat(
        rc::gen::container<std::string>(
            rc::gen::inRange(static_cast<char>('a'), static_cast<char>('z' + 1))),
        [](const std::string& s) {
            if (s.empty()) return false;
            // Reject any yaml-cpp boolean literals
            return s != "true" && s != "false" &&
                   s != "yes" && s != "no" &&
                   s != "on" && s != "off" &&
                   s != "y" && s != "n";
        });
}

/// Generate an integer that is NOT 0 or 1 (cannot be interpreted as bool).
rc::Gen<int> genNonBoolInt() {
    return rc::gen::suchThat(
        rc::gen::inRange(-1000, 1001),
        [](int v) { return v != 0 && v != 1; });
}

}  // anonymous namespace

// --- Property 2a: String values cannot be read as int ------------------------
// Feature: conf-config-parser, Property 2: Type-Safety
//
// For a generated non-numeric string stored as a YAML scalar:
//   - try_int returns nullopt
//   - get_int raises Conf_Error{Type_Mismatch}
//   - get_or<int> returns the fallback
//   - The Config remains valid after the failed access
//
// **Validates: Requirements 28.1, 28.2, 28.3, 28.4**

RC_GTEST_PROP(TypeSafetyProperty2, StringCannotBeReadAsInt, ()) {
    const std::string val = *genNonNumericString();

    // Quote the value to ensure YAML treats it as a string scalar.
    const std::string yaml = make_yaml("\"" + val + "\"");
    auto cfg = conf::Config::from_string(yaml);

    // (Req 28.2) Non-throwing accessor returns nullopt.
    RC_ASSERT(!cfg.try_int("key").has_value());

    // (Req 28.1) Throwing accessor raises Type_Mismatch.
    bool threw_type_mismatch = false;
    try {
        (void)cfg.get_int("key");
    } catch (const conf::Conf_Error& e) {
        threw_type_mismatch = (e.code() == conf::Error_Code::Type_Mismatch);
    }
    RC_ASSERT(threw_type_mismatch);

    // (Req 28.3) Defaulted accessor returns the fallback.
    const int fallback = 12345;
    RC_ASSERT(cfg.get_or("key", fallback) == fallback);

    // (Req 28.4) Config remains valid for subsequent queries.
    RC_ASSERT(cfg.has("key"));
    RC_ASSERT(cfg.try_string("key").has_value());
}

// --- Property 2b: String values cannot be read as bool -----------------------
// Feature: conf-config-parser, Property 2: Type-Safety
//
// For a generated string that is not a boolean literal:
//   - try_bool returns nullopt
//   - get_bool raises Conf_Error{Type_Mismatch}
//   - get_or<bool> returns the fallback
//   - The Config remains valid after the failed access
//
// **Validates: Requirements 28.1, 28.2, 28.3, 28.4**

RC_GTEST_PROP(TypeSafetyProperty2, StringCannotBeReadAsBool, ()) {
    const std::string val = *genNonBoolString();

    const std::string yaml = make_yaml("\"" + val + "\"");
    auto cfg = conf::Config::from_string(yaml);

    // (Req 28.2) Non-throwing accessor returns nullopt.
    RC_ASSERT(!cfg.try_bool("key").has_value());

    // (Req 28.1) Throwing accessor raises Type_Mismatch.
    bool threw_type_mismatch = false;
    try {
        (void)cfg.get_bool("key");
    } catch (const conf::Conf_Error& e) {
        threw_type_mismatch = (e.code() == conf::Error_Code::Type_Mismatch);
    }
    RC_ASSERT(threw_type_mismatch);

    // (Req 28.3) Defaulted accessor returns the fallback.
    RC_ASSERT(cfg.get_or("key", true) == true);
    RC_ASSERT(cfg.get_or("key", false) == false);

    // (Req 28.4) Config remains valid after the failed access.
    RC_ASSERT(cfg.has("key"));
    RC_ASSERT(cfg.try_string("key").has_value());
}

// --- Property 2c: Maps cannot be read as any scalar --------------------------
// Feature: conf-config-parser, Property 2: Type-Safety
//
// A map node at "key" cannot be read through any scalar accessor.
//   - try_int, try_double, try_bool, try_string all return nullopt
//   - get_int, get_double, get_bool, get_string all raise Type_Mismatch
//   - get_or for all types returns the fallback
//   - The Config remains valid
//
// **Validates: Requirements 28.1, 28.2, 28.3, 28.4**

RC_GTEST_PROP(TypeSafetyProperty2, MapCannotBeReadAsScalar, ()) {
    // Generate a random number of map entries to vary the structure.
    const int num_entries = *rc::gen::inRange(1, 6);

    std::ostringstream yaml_stream;
    yaml_stream << "key:\n";
    for (int i = 0; i < num_entries; ++i) {
        yaml_stream << "  sub" << i << ": value" << i << "\n";
    }
    auto cfg = conf::Config::from_string(yaml_stream.str());

    // Verify it is indeed a map.
    RC_ASSERT(cfg.is_map("key"));

    // (Req 28.2) Non-throwing accessors all return nullopt.
    RC_ASSERT(!cfg.try_int("key").has_value());
    RC_ASSERT(!cfg.try_double("key").has_value());
    RC_ASSERT(!cfg.try_bool("key").has_value());
    RC_ASSERT(!cfg.try_string("key").has_value());

    // (Req 28.1) Throwing accessors all raise Type_Mismatch.
    auto throws_type_mismatch = [&](auto accessor_fn) -> bool {
        try {
            accessor_fn();
            return false;  // Did not throw
        } catch (const conf::Conf_Error& e) {
            return e.code() == conf::Error_Code::Type_Mismatch;
        } catch (...) {
            return false;  // Wrong exception type
        }
    };

    RC_ASSERT(throws_type_mismatch([&]() { (void)cfg.get_int("key"); }));
    RC_ASSERT(throws_type_mismatch([&]() { (void)cfg.get_double("key"); }));
    RC_ASSERT(throws_type_mismatch([&]() { (void)cfg.get_bool("key"); }));
    RC_ASSERT(throws_type_mismatch([&]() { (void)cfg.get_string("key"); }));

    // (Req 28.3) Defaulted accessors return their fallbacks.
    RC_ASSERT(cfg.get_or("key", 999) == 999);
    RC_ASSERT(cfg.get_or("key", 1.5) == 1.5);
    RC_ASSERT(cfg.get_or("key", true) == true);
    RC_ASSERT(cfg.get_or<std::string>("key", std::string("fb")) == "fb");

    // (Req 28.4) Config remains valid.
    RC_ASSERT(cfg.has("key"));
    RC_ASSERT(cfg.is_map("key"));
}

// --- Property 2d: Sequences cannot be read as any scalar ---------------------
// Feature: conf-config-parser, Property 2: Type-Safety
//
// A sequence node at "key" cannot be read through any scalar accessor.
//   - try_int, try_double, try_bool, try_string all return nullopt
//   - get_int, get_double, get_bool, get_string all raise Type_Mismatch
//   - get_or for all types returns the fallback
//   - The Config remains valid
//
// **Validates: Requirements 28.1, 28.2, 28.3, 28.4**

RC_GTEST_PROP(TypeSafetyProperty2, SequenceCannotBeReadAsScalar, ()) {
    // Generate a random-length sequence.
    const int num_items = *rc::gen::inRange(1, 10);

    std::ostringstream yaml_stream;
    yaml_stream << "key:\n";
    for (int i = 0; i < num_items; ++i) {
        yaml_stream << "  - item" << i << "\n";
    }
    auto cfg = conf::Config::from_string(yaml_stream.str());

    // Verify it is indeed a sequence.
    RC_ASSERT(cfg.is_sequence("key"));

    // (Req 28.2) Non-throwing accessors all return nullopt.
    RC_ASSERT(!cfg.try_int("key").has_value());
    RC_ASSERT(!cfg.try_double("key").has_value());
    RC_ASSERT(!cfg.try_bool("key").has_value());
    RC_ASSERT(!cfg.try_string("key").has_value());

    // (Req 28.1) Throwing accessors all raise Type_Mismatch.
    auto throws_type_mismatch = [&](auto accessor_fn) -> bool {
        try {
            accessor_fn();
            return false;
        } catch (const conf::Conf_Error& e) {
            return e.code() == conf::Error_Code::Type_Mismatch;
        } catch (...) {
            return false;
        }
    };

    RC_ASSERT(throws_type_mismatch([&]() { (void)cfg.get_int("key"); }));
    RC_ASSERT(throws_type_mismatch([&]() { (void)cfg.get_double("key"); }));
    RC_ASSERT(throws_type_mismatch([&]() { (void)cfg.get_bool("key"); }));
    RC_ASSERT(throws_type_mismatch([&]() { (void)cfg.get_string("key"); }));

    // (Req 28.3) Defaulted accessors return their fallbacks.
    RC_ASSERT(cfg.get_or("key", -42) == -42);
    RC_ASSERT(cfg.get_or("key", 2.718) == 2.718);
    RC_ASSERT(cfg.get_or("key", false) == false);
    RC_ASSERT(cfg.get_or<std::string>("key", std::string("fallback")) == "fallback");

    // (Req 28.4) Config remains valid.
    RC_ASSERT(cfg.has("key"));
    RC_ASSERT(cfg.is_sequence("key"));
}

// --- Property 2e: Int values succeed for int but fail for bool (non-0/1) -----
// Feature: conf-config-parser, Property 2: Type-Safety
//
// For generated integers that are NOT 0 or 1:
//   - try_int succeeds and returns the value
//   - try_bool returns nullopt (not convertible to boolean)
//   - get_bool raises Type_Mismatch
//   - get_or<bool> returns the fallback
//   - The Config remains valid
//
// **Validates: Requirements 28.1, 28.2, 28.3, 28.4**

RC_GTEST_PROP(TypeSafetyProperty2, IntNotZeroOrOneFailsAsBool, ()) {
    const int val = *genNonBoolInt();

    const std::string yaml = make_yaml(std::to_string(val));
    auto cfg = conf::Config::from_string(yaml);

    // The int accessor succeeds — confirm correctness.
    RC_ASSERT(cfg.try_int("key").has_value());
    RC_ASSERT(cfg.try_int("key").value() == val);

    // (Req 28.2) Bool non-throwing accessor returns nullopt for non-0/1 ints.
    RC_ASSERT(!cfg.try_bool("key").has_value());

    // (Req 28.1) Bool throwing accessor raises Type_Mismatch.
    bool threw_type_mismatch = false;
    try {
        (void)cfg.get_bool("key");
    } catch (const conf::Conf_Error& e) {
        threw_type_mismatch = (e.code() == conf::Error_Code::Type_Mismatch);
    }
    RC_ASSERT(threw_type_mismatch);

    // (Req 28.3) Defaulted bool accessor returns fallback.
    RC_ASSERT(cfg.get_or("key", true) == true);
    RC_ASSERT(cfg.get_or("key", false) == false);

    // (Req 28.4) Config remains valid after the failed access.
    RC_ASSERT(cfg.has("key"));
    RC_ASSERT(cfg.try_int("key").has_value());
}
