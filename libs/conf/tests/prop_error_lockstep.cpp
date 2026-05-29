// ─── Property-Based Tests: Error-Code / Exception Lockstep ───────────────────
// Feature: conf-config-parser, Property 6: Error-Code/Exception Lockstep
//
// Uses RapidCheck to verify that for EVERY failure mode, the Error_Code carried
// by the thrown conf::Conf_Error (from the C++ core accessor) is identical to
// the int returned by the corresponding extern "C" bridge function. The two
// surfaces are driven against the SAME document (one parsed as a conf::Config,
// one loaded as a bridge handle from the identical YAML) so any divergence
// between the C++ exception taxonomy and the Fortran-visible return codes is a
// genuine lockstep violation.
//
// The lockstep holds by construction: the CONF_C_TRY macro maps a thrown
// conf::Conf_Error to static_cast<int>(e.code()), so the bridge returns exactly
// the code the core would have thrown. These properties prove that contract
// across many generated inputs and every failure mode.
//
// Failure modes covered (each a separate property; one combined property also
// selects among the query-path modes each iteration):
//
//   QUERY PATH (same handle/Config, varying key/value):
//     * Key_Not_Found (4)  — a random absent key. get_int throws Key_Not_Found;
//                            conf_get_int_c returns 4.
//     * Type_Mismatch (5)  — a random non-numeric string scalar requested as
//                            int. get_int throws Type_Mismatch; conf_get_int_c
//                            returns 5.
//     * Invalid_Arg   (1)  — a malformed dotted path (leading/trailing/double
//                            dot). Note: the empty-path case cannot be reached
//                            through the bridge because key_len < 1 is itself an
//                            arg-check Invalid_Arg; we therefore use a malformed
//                            but NON-EMPTY key (".a", "a.", "a..b", ...) so the
//                            Invalid_Arg originates from the resolver on both
//                            surfaces, with key_len >= 1.
//
//   LOAD PATH (lockstep on construction):
//     * Parse_Error    (3) — malformed YAML. Config::from_string throws
//                            Parse_Error; conf_load_string_c returns 3.
//     * File_Not_Found (2) — a missing path. Config::from_file throws
//                            File_Not_Found; conf_load_c returns 2.
//
// A helper captures the C++ code via try/catch (returning the int code) so each
// check funnels down to RC_ASSERT(cpp_code == bridge_ret) over two printable
// ints. Every bridge handle opened by a test is closed (conf_close_c) so the
// property leaks nothing.
//
// The tests run single-process with no MPI and >= 100 generated iterations
// (RC_PARAMS=max_success=100, set by the harness).
//
// **Validates: Requirements 11.4, 19.2**
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <array>
#include <string>
#include <utility>

#include <conf/config.hpp>
#include <conf/error.hpp>

// ─── extern "C" bridge prototypes ────────────────────────────────────────────
// CONF ships no public C header (the bridge is an implementation detail consumed
// only by conf_mod.f90), so the prototypes are declared here exactly as defined
// in src/fortran/conf_c_interop.cpp. Only the functions exercised by these
// properties are declared.
extern "C" {
int conf_load_c(const char* path, int path_len, int* handle_out);
int conf_load_string_c(const char* yaml_text, int text_len, int* handle_out);
int conf_close_c(int handle);
int conf_get_int_c(int handle, const char* key, int key_len, int* out);
}

namespace {

// ─── Generalized helper: capture a C++ accessor's Error_Code as an int ───────
// Invokes a throwing callable and returns the int value of the Error_Code it
// raised via conf::Conf_Error: Success (0) if it did not throw, or Unknown (99)
// for any non-Conf_Error escape. This is the C++ half of the lockstep — it is
// compared directly against the bridge function's int return.
template <typename Fn>
int cpp_error_code(Fn&& fn) {
    try {
        std::forward<Fn>(fn)();
        return static_cast<int>(conf::Error_Code::Success);
    } catch (const conf::Conf_Error& e) {
        return static_cast<int>(e.code());
    } catch (...) {
        return static_cast<int>(conf::Error_Code::Unknown);
    }
}

// Open a bridge handle from YAML text, asserting the load succeeds. Returns the
// opaque token (> 0). Callers must close it with conf_close_c.
int open_bridge_handle(const std::string& yaml) {
    int handle = 0;
    const int rc =
        conf_load_string_c(yaml.data(), static_cast<int>(yaml.size()), &handle);
    RC_ASSERT(rc == static_cast<int>(conf::Error_Code::Success));
    RC_ASSERT(handle > 0);
    return handle;
}

// A known, well-formed document shared by the query-path properties. It carries
// a scalar int, a scalar string, and a nested map so that resolution succeeds
// for present keys and the failure modes below are isolated to the generated
// key/value under test.
constexpr const char* kBaseDoc =
    "present_int: 42\n"
    "present_map:\n"
    "  child: 1\n";

// Generator: a random run of lowercase ASCII letters ('a'..'z'), possibly empty.
// Used to build keys/values that need no YAML quoting or escaping.
rc::Gen<std::string> letters() {
    return rc::gen::container<std::string>(rc::gen::inRange<char>('a', '{'));
}

}  // namespace

// ─── Property 6a: Key_Not_Found lockstep ─────────────────────────────────────
// Feature: conf-config-parser, Property 6: Error-Code/Exception Lockstep
//
// A randomly generated, definitely-absent single-segment key. The C++
// get_int(key) throws Conf_Error{Key_Not_Found} and conf_get_int_c returns 4 —
// they are equal.
//
// **Validates: Requirements 11.4, 19.2**

RC_GTEST_PROP(ErrorLockstepProperty6, KeyNotFoundLockstep, ()) {
    // Prefixed so the key can never collide with a present key, and contains no
    // '.', so it is a well-formed single-segment path that simply does not
    // resolve (Key_Not_Found, never Invalid_Arg).
    const std::string missing_key = "no_such_key_" + *letters();

    const std::string doc = kBaseDoc;
    const conf::Config cfg = conf::Config::from_string(doc);

    const int cpp_code =
        cpp_error_code([&] { (void)cfg.get_int(missing_key); });

    const int handle = open_bridge_handle(doc);
    int out = 0;
    const int bridge_ret = conf_get_int_c(
        handle, missing_key.data(), static_cast<int>(missing_key.size()), &out);
    RC_ASSERT(conf_close_c(handle) == static_cast<int>(conf::Error_Code::Success));

    // Both surfaces report exactly Key_Not_Found, and they agree.
    RC_ASSERT(cpp_code == static_cast<int>(conf::Error_Code::Key_Not_Found));
    RC_ASSERT(cpp_code == bridge_ret);
}

// ─── Property 6b: Type_Mismatch lockstep ─────────────────────────────────────
// Feature: conf-config-parser, Property 6: Error-Code/Exception Lockstep
//
// A scalar whose text is never numeric and never a bool keyword (a leading 'k'
// guarantees this) requested as int. The C++ get_int throws
// Conf_Error{Type_Mismatch} and conf_get_int_c returns 5 — they are equal.
//
// **Validates: Requirements 11.4, 19.2**

RC_GTEST_PROP(ErrorLockstepProperty6, TypeMismatchLockstep, ()) {
    // "k" + letters can never parse as int/double and never equals a YAML bool
    // keyword, so requesting it as an int is always a Type_Mismatch.
    const std::string value = "k" + *letters();
    const std::string doc = "text: " + value + "\n";

    const conf::Config cfg = conf::Config::from_string(doc);
    const int cpp_code = cpp_error_code([&] { (void)cfg.get_int("text"); });

    const int handle = open_bridge_handle(doc);
    int out = 0;
    const int bridge_ret = conf_get_int_c(handle, "text", 4, &out);
    RC_ASSERT(conf_close_c(handle) == static_cast<int>(conf::Error_Code::Success));

    RC_ASSERT(cpp_code == static_cast<int>(conf::Error_Code::Type_Mismatch));
    RC_ASSERT(cpp_code == bridge_ret);
}

// ─── Property 6c: Invalid_Arg lockstep ───────────────────────────────────────
// Feature: conf-config-parser, Property 6: Error-Code/Exception Lockstep
//
// A malformed but NON-EMPTY dotted path (leading dot, trailing dot, or
// consecutive dots). key_len >= 1 on the bridge, so the Invalid_Arg originates
// from the resolver — NOT from the bridge's key_len < 1 argument check — on both
// surfaces. The C++ get_int throws Conf_Error{Invalid_Arg} and conf_get_int_c
// returns 1 — they are equal.
//
// **Validates: Requirements 11.4, 19.2**

RC_GTEST_PROP(ErrorLockstepProperty6, InvalidArgLockstep, ()) {
    // Every entry is non-empty (key_len >= 1) yet contains an empty segment, so
    // the resolver rejects it with Invalid_Arg before walking any node.
    static const std::array<const char*, 5> kMalformed = {
        ".a", "a.", "a..b", "..", "."};
    const int idx = *rc::gen::inRange<int>(0, static_cast<int>(kMalformed.size()));
    const std::string bad_path = kMalformed[static_cast<std::size_t>(idx)];

    const std::string doc = kBaseDoc;
    const conf::Config cfg = conf::Config::from_string(doc);
    const int cpp_code = cpp_error_code([&] { (void)cfg.get_int(bad_path); });

    const int handle = open_bridge_handle(doc);
    int out = 0;
    const int bridge_ret = conf_get_int_c(
        handle, bad_path.data(), static_cast<int>(bad_path.size()), &out);
    RC_ASSERT(conf_close_c(handle) == static_cast<int>(conf::Error_Code::Success));

    RC_ASSERT(cpp_code == static_cast<int>(conf::Error_Code::Invalid_Arg));
    RC_ASSERT(cpp_code == bridge_ret);
}

// ─── Property 6d: Parse_Error lockstep (load path) ───────────────────────────
// Feature: conf-config-parser, Property 6: Error-Code/Exception Lockstep
//
// Malformed YAML drives the load path. Config::from_string throws
// Conf_Error{Parse_Error} and conf_load_string_c returns 3 — they are equal.
// (Inputs are the documents known to trigger YAML::ParserException in the pinned
// yaml-cpp 0.8.0, mirroring test_malformed_yaml.cpp.)
//
// **Validates: Requirements 11.4, 19.2**

RC_GTEST_PROP(ErrorLockstepProperty6, ParseErrorLockstep, ()) {
    static const std::array<const char*, 6> kBad = {
        "key: [1, 2",
        "key: {a: 1",
        "root: {list: [1, 2, 3}",
        "parent:\n\tchild: value\n",
        "a:\n  - b\n c: oops\n",
        "key: value1: value2\n",
    };
    const int idx = *rc::gen::inRange<int>(0, static_cast<int>(kBad.size()));
    const std::string bad = kBad[static_cast<std::size_t>(idx)];

    const int cpp_code =
        cpp_error_code([&] { (void)conf::Config::from_string(bad); });

    int handle = 0;
    const int bridge_ret =
        conf_load_string_c(bad.data(), static_cast<int>(bad.size()), &handle);
    // A failed load must not register a handle; guard against a leak regardless.
    if (bridge_ret == static_cast<int>(conf::Error_Code::Success)) {
        conf_close_c(handle);
    }

    RC_ASSERT(cpp_code == static_cast<int>(conf::Error_Code::Parse_Error));
    RC_ASSERT(cpp_code == bridge_ret);
}

// ─── Property 6e: File_Not_Found lockstep (load path) ────────────────────────
// Feature: conf-config-parser, Property 6: Error-Code/Exception Lockstep
//
// A path that does not exist drives the file load path. Config::from_file throws
// Conf_Error{File_Not_Found} and conf_load_c returns 2 — they are equal.
//
// **Validates: Requirements 11.4, 19.2**

RC_GTEST_PROP(ErrorLockstepProperty6, FileNotFoundLockstep, ()) {
    // A path under a directory that cannot exist; the random suffix keeps the
    // generator broad without ever naming a real file.
    const std::string path =
        "/nonexistent_conf_dir_/does_not_exist_" + *letters() + ".yaml";

    const int cpp_code =
        cpp_error_code([&] { (void)conf::Config::from_file(path); });

    int handle = 0;
    const int bridge_ret =
        conf_load_c(path.data(), static_cast<int>(path.size()), &handle);
    if (bridge_ret == static_cast<int>(conf::Error_Code::Success)) {
        conf_close_c(handle);
    }

    RC_ASSERT(cpp_code == static_cast<int>(conf::Error_Code::File_Not_Found));
    RC_ASSERT(cpp_code == bridge_ret);
}

// ─── Property 6f: combined query-path lockstep (selects a mode each run) ─────
// Feature: conf-config-parser, Property 6: Error-Code/Exception Lockstep
//
// Each iteration selects among the three query-path failure modes and asserts
// the same generalized invariant — cpp_code == bridge_ret — over a single shared
// document/handle. This directly exercises the "generate/select among the
// failure modes each iteration" form of the property: whatever mode is chosen,
// the thrown Error_Code and the bridge's returned int stay in lockstep.
//
// **Validates: Requirements 11.4, 19.2**

RC_GTEST_PROP(ErrorLockstepProperty6, CombinedQueryPathLockstep, ()) {
    enum class Mode { Key_Not_Found, Type_Mismatch, Invalid_Arg };
    const int mode_idx = *rc::gen::inRange<int>(0, 3);
    const auto mode = static_cast<Mode>(mode_idx);

    // Build the document and the key under test according to the selected mode.
    std::string doc = kBaseDoc;
    std::string key;
    conf::Error_Code expected = conf::Error_Code::Success;

    if (mode == Mode::Key_Not_Found) {
        key = "no_such_key_" + *letters();
        expected = conf::Error_Code::Key_Not_Found;
    } else if (mode == Mode::Type_Mismatch) {
        const std::string value = "k" + *letters();
        doc = "text: " + value + "\n";
        key = "text";
        expected = conf::Error_Code::Type_Mismatch;
    } else {  // Invalid_Arg
        static const std::array<const char*, 5> kMalformed = {
            ".a", "a.", "a..b", "..", "."};
        const int idx =
            *rc::gen::inRange<int>(0, static_cast<int>(kMalformed.size()));
        key = kMalformed[static_cast<std::size_t>(idx)];
        expected = conf::Error_Code::Invalid_Arg;
    }

    const conf::Config cfg = conf::Config::from_string(doc);
    const int cpp_code = cpp_error_code([&] { (void)cfg.get_int(key); });

    const int handle = open_bridge_handle(doc);
    int out = 0;
    const int bridge_ret =
        conf_get_int_c(handle, key.data(), static_cast<int>(key.size()), &out);
    RC_ASSERT(conf_close_c(handle) == static_cast<int>(conf::Error_Code::Success));

    // The selected mode produced its expected code, and the two surfaces agree.
    RC_ASSERT(cpp_code == static_cast<int>(expected));
    RC_ASSERT(cpp_code == bridge_ret);
}
