// ─── Property-Based Tests: conf::Config type-safety ──────────────────────────
// Feature: conf-config-parser, Property 2: Type-Safety
//
// Uses RapidCheck to verify that for ANY node holding a value of type A, a
// request for an incompatible type B never returns a value:
//
//   * the non-throwing accessor  try_B(path)        yields std::nullopt,
//   * the throwing accessor      get_B(path)         raises Conf_Error whose
//                                                    code() == Type_Mismatch,
//   * the defaulted accessor     get_or<B>(path, f)  returns the fallback f,
//
// and the access never fabricates a value, never returns garbage, and never
// crashes. After any incompatible request the Config remains usable.
//
// Two complementary generators exercise the two ways a request can be
// incompatible (mirroring Requirement 28.1's definition of an incompatible
// node):
//
//   1. A NON-SCALAR node (map or sequence) requested as ANY scalar type — int,
//      double, bool, AND string. A non-scalar can never convert to a scalar,
//      including string (yaml-cpp rejects `as<std::string>()` on a map/sequence),
//      so all four typed scalar accessors must report Type_Mismatch.
//
//   2. A NON-NUMERIC, NON-BOOL string scalar requested as int / double / bool.
//      Every scalar is a valid string, so get_string MUST succeed here; only the
//      numeric and boolean requests are incompatible. The generator prefixes the
//      value with a letter so it can never be parsed as a number, and the whole
//      value can never equal a YAML bool keyword, making the mismatch reliable.
//
// **Validates: Requirements 28.1, 28.2, 28.3, 28.4**
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <optional>
#include <string>

#include <conf/config.hpp>
#include <conf/error.hpp>

namespace {

// ─── Helper: did the call throw exactly Conf_Error{Type_Mismatch}? ───────────
// RapidCheck's RC_ASSERT renders the asserted expression, so every check in this
// file is funnelled down to a printable `bool`. This helper invokes a throwing
// accessor and reports whether it raised a Conf_Error carrying Type_Mismatch —
// not merely "threw something" — so a wrong error code is still a failure.
template <typename Fn>
bool throws_type_mismatch(Fn&& fn) {
    try {
        std::forward<Fn>(fn)();
    } catch (const conf::Conf_Error& e) {
        return e.code() == conf::Error_Code::Type_Mismatch;
    } catch (...) {
        return false;  // wrong exception type
    }
    return false;      // did not throw at all
}

// ─── Generator: a document binding key "node" to a NON-scalar node ───────────
// Randomly chooses a map or a sequence each iteration and varies the child
// count. Children are plain integers so the document always parses; what matters
// for the property is that "node" resolves to a non-scalar node.
struct NonScalarDoc {
    std::string yaml;    ///< YAML text whose top-level key "node" is non-scalar.
    bool        is_map;  ///< true => map, false => sequence (for diagnostics).
};

rc::Gen<NonScalarDoc> genNonScalarDoc() {
    return rc::gen::exec([]() -> NonScalarDoc {
        const bool is_map = *rc::gen::arbitrary<bool>();
        const int  n      = *rc::gen::inRange(1, 6);  // 1..5 children (non-empty)

        std::string yaml = "node:\n";
        for (int i = 0; i < n; ++i) {
            if (is_map) {
                yaml += "  k" + std::to_string(i) + ": " + std::to_string(i) + "\n";
            } else {
                yaml += "  - " + std::to_string(i) + "\n";
            }
        }
        return NonScalarDoc{yaml, is_map};
    });
}

// ─── Generator: a scalar value that is neither numeric nor boolean ───────────
// A leading 'k' guarantees the scalar cannot be parsed as an int or double, and
// no YAML bool keyword starts with 'k', so the whole value can never be a bool
// (nor null). The body is drawn from lowercase ASCII letters only, so it needs
// no YAML quoting/escaping when spliced into `node: <value>`.
rc::Gen<std::string> genNonNumericNonBoolScalar() {
    return rc::gen::map(
        rc::gen::container<std::string>(rc::gen::inRange<char>('a', '{')),  // 'a'..'z'
        [](std::string letters) -> std::string { return "k" + letters; });
}

}  // namespace

// ─── Property 2a: a non-scalar node rejects EVERY scalar request ─────────────
// Feature: conf-config-parser, Property 2: Type-Safety
//
// For a node that is a map or a sequence, get_int/get_double/get_bool/get_string
// all raise Type_Mismatch (28.1), the try_* flavors all yield nullopt (28.2),
// and get_or<T> returns its fallback (28.3). The Config stays valid afterwards
// (28.4).
//
// **Validates: Requirements 28.1, 28.2, 28.3, 28.4**

RC_GTEST_PROP(ConfigTypeSafetyProperty2,
              NonScalarNodeRejectsEveryScalarRequest,
              ()) {
    const NonScalarDoc doc = *genNonScalarDoc();
    const conf::Config cfg = conf::Config::from_string(doc.yaml);

    // Sanity: the generated node really is the non-scalar we intended. (If this
    // ever failed it would indicate a generator bug, not a property violation.)
    // The kind check is reduced to a plain bool before RC_ASSERT, since the
    // macro's expression-capture template cannot wrap a ternary operand.
    const bool is_expected_kind =
        doc.is_map ? cfg.is_map("node") : cfg.is_sequence("node");
    RC_ASSERT(cfg.has("node"));
    RC_ASSERT(is_expected_kind);

    // Fallbacks are generated so "returns the fallback" genuinely verifies the
    // accessor handed back OUR value rather than a fabricated one.
    const int         fb_int    = *rc::gen::arbitrary<int>();
    const double      fb_double = *rc::gen::arbitrary<double>();
    const bool        fb_bool   = *rc::gen::arbitrary<bool>();
    const std::string fb_string = *rc::gen::arbitrary<std::string>();

    // 28.1 — throwing accessors raise Type_Mismatch for ALL scalar types,
    //        including string (a non-scalar is not a valid string either).
    RC_ASSERT(throws_type_mismatch([&] { (void)cfg.get_int("node"); }));
    RC_ASSERT(throws_type_mismatch([&] { (void)cfg.get_double("node"); }));
    RC_ASSERT(throws_type_mismatch([&] { (void)cfg.get_bool("node"); }));
    RC_ASSERT(throws_type_mismatch([&] { (void)cfg.get_string("node"); }));

    // 28.2 — non-throwing accessors yield an empty optional (no fabricated value).
    RC_ASSERT(!cfg.try_int("node").has_value());
    RC_ASSERT(!cfg.try_double("node").has_value());
    RC_ASSERT(!cfg.try_bool("node").has_value());
    RC_ASSERT(!cfg.try_string("node").has_value());

    // 28.3 — get_or returns exactly the supplied fallback.
    RC_ASSERT(cfg.get_or<int>("node", fb_int) == fb_int);
    RC_ASSERT(cfg.get_or<double>("node", fb_double) == fb_double);
    RC_ASSERT(cfg.get_or<bool>("node", fb_bool) == fb_bool);
    RC_ASSERT(cfg.get_or<std::string>("node", fb_string) == fb_string);

    // 28.4 — the Config remains valid for subsequent queries after the failures.
    RC_ASSERT(cfg.has("node"));
}

// ─── Property 2b: a non-numeric / non-bool scalar rejects numeric & bool ─────
// Feature: conf-config-parser, Property 2: Type-Safety
//
// For a scalar whose text is not a valid number or bool, get_int/get_double/
// get_bool raise Type_Mismatch (28.1), the corresponding try_* yield nullopt
// (28.2), and get_or returns the fallback (28.3) — while get_string SUCCEEDS,
// since every scalar is a valid string. The Config stays valid afterwards (28.4).
//
// **Validates: Requirements 28.1, 28.2, 28.3, 28.4**

RC_GTEST_PROP(ConfigTypeSafetyProperty2,
              NonNumericScalarRejectsNumericAndBoolButIsAValidString,
              ()) {
    const std::string value = *genNonNumericNonBoolScalar();
    const conf::Config cfg   = conf::Config::from_string("node: " + value + "\n");

    const int    fb_int    = *rc::gen::arbitrary<int>();
    const double fb_double = *rc::gen::arbitrary<double>();
    const bool   fb_bool   = *rc::gen::arbitrary<bool>();

    // 28.1 — the numeric and boolean requests are incompatible: Type_Mismatch.
    RC_ASSERT(throws_type_mismatch([&] { (void)cfg.get_int("node"); }));
    RC_ASSERT(throws_type_mismatch([&] { (void)cfg.get_double("node"); }));
    RC_ASSERT(throws_type_mismatch([&] { (void)cfg.get_bool("node"); }));

    // 28.2 — and their non-throwing flavors are empty.
    RC_ASSERT(!cfg.try_int("node").has_value());
    RC_ASSERT(!cfg.try_double("node").has_value());
    RC_ASSERT(!cfg.try_bool("node").has_value());

    // 28.3 — get_or returns the fallback for the incompatible types.
    RC_ASSERT(cfg.get_or<int>("node", fb_int) == fb_int);
    RC_ASSERT(cfg.get_or<double>("node", fb_double) == fb_double);
    RC_ASSERT(cfg.get_or<bool>("node", fb_bool) == fb_bool);

    // The string request is COMPATIBLE — every scalar is a valid string — so it
    // must succeed and return the exact value (never a mismatch). This guards the
    // property from being trivially satisfied by an over-broad implementation.
    const std::optional<std::string> as_str = cfg.try_string("node");
    RC_ASSERT(as_str.has_value());
    RC_ASSERT(*as_str == value);
    RC_ASSERT(cfg.get_string("node") == value);

    // 28.4 — the Config remains valid for subsequent queries.
    RC_ASSERT(cfg.has("node"));
}
