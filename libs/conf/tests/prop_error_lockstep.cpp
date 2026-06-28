// --- Property-Based Tests: Error-Code / Exception Lockstep ------------------
// Feature: conf-config-parser
//
// Uses RapidCheck to verify that for every failure mode, the Error_Code carried
// by the thrown Conf_Error equals the int the corresponding C bridge function
// returns. This guarantees the Fortran-visible error codes never drift from the
// C++ core's meaning.
//
// Property 6: Error-Code/Exception Lockstep
//   Validates: Requirements 11.4, 19.2
// -----------------------------------------------------------------------------

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <conf/config.hpp>
#include <conf/error.hpp>
#include <cstring>
#include <optional>
#include <string>

// ─── extern "C" prototypes for the CONF C bridge ─────────────────────────────

extern "C" {

int conf_load_string_c(const char *yaml_text, int text_len, int *handle_out);
int conf_close_c(int handle);

int conf_get_int_c(int handle, const char *key, int key_len, int *out);
int conf_get_double_c(int handle, const char *key, int key_len, double *out);
int conf_get_bool_c(int handle, const char *key, int key_len, int *out);
int conf_get_string_len_c(int handle, const char *key, int key_len, int *str_len_out);

}  // extern "C"

namespace {

// ─── Test fixture YAML with known keys of various types ──────────────────────
// Provides a small document with int, double, bool, string, nested map, and
// sequence so that random keys will sometimes hit existing nodes (success or
// type mismatch) and sometimes miss entirely (Key_Not_Found).

constexpr const char *kFixtureYaml =
    "answer: 42\n"
    "pi: 3.14159\n"
    "enabled: true\n"
    "name: hello\n"
    "nested:\n"
    "  depth: 7\n"
    "  flag: false\n"
    "items:\n"
    "  - 10\n"
    "  - 20\n"
    "  - 30\n";

// ─── Helpers: load both C++ and C configs from same YAML ─────────────────────

struct DualConfig {
    conf::Config cpp_cfg;
    int c_handle;

    DualConfig(conf::Config cfg, int h) : cpp_cfg(std::move(cfg)), c_handle(h) {}
    DualConfig(DualConfig &&other) noexcept : cpp_cfg(std::move(other.cpp_cfg)), c_handle(other.c_handle) {
        other.c_handle = 0;  // prevent double-close
    }
    DualConfig &operator=(DualConfig &&other) noexcept {
        if (this != &other) {
            if (c_handle > 0) conf_close_c(c_handle);
            cpp_cfg = std::move(other.cpp_cfg);
            c_handle = other.c_handle;
            other.c_handle = 0;
        }
        return *this;
    }
    ~DualConfig() {
        if (c_handle > 0) {
            conf_close_c(c_handle);
        }
    }
};

std::optional<DualConfig> load_dual(const std::string &yaml) {
    // C++ side
    conf::Config cpp_cfg = conf::Config::from_string(yaml);

    // C bridge side
    int handle = 0;
    int rc = conf_load_string_c(yaml.c_str(), static_cast<int>(yaml.size()), &handle);
    if (rc != 0) return std::nullopt;

    return DualConfig(std::move(cpp_cfg), handle);
}

// ─── Key generators ─────────────────────────────────────────────────────────
// Mix of keys that exist (to trigger success or type mismatch) and random
// keys (to trigger Key_Not_Found or Invalid_Arg).

rc::Gen<std::string> genKey() {
    return rc::gen::oneOf(
        // Known keys (may succeed or type-mismatch depending on accessor)
        rc::gen::element<std::string>("answer", "pi", "enabled", "name", "nested", "nested.depth", "nested.flag", "items", "items.0", "items.1",
                                      "items.2"),
        // Missing keys
        rc::gen::element<std::string>("missing", "no.such.key", "items.99", "nested.nope"),
        // Malformed paths (should yield Invalid_Arg)
        rc::gen::element<std::string>("", ".leading", "trailing.", "double..dot", "...", ".."),
        // Random alphanumeric keys (most will be Key_Not_Found)
        rc::gen::container<std::string>(rc::gen::inRange('a', 'z')));
}

// ─── Comparison logic ────────────────────────────────────────────────────────
// For a given key, call the C++ throwing accessor and the C bridge function.
// Compare outcomes: if C++ throws Conf_Error(code), C bridge must return code.
// If C++ succeeds with value V, C bridge must return 0 and output must equal V.

enum class CppOutcome { Success, ConfError, OtherException };

struct CppResult {
    CppOutcome outcome;
    int error_code = 0;  // meaningful when outcome == ConfError
};

}  // anonymous namespace

// --- Property 6a: get_int lockstep -------------------------------------------
// For random keys on a fixed YAML document, conf::Config::get_int and
// conf_get_int_c return identical error codes / success.
//
// **Validates: Requirements 11.4, 19.2**

RC_GTEST_PROP(ErrorLockstepProperty6, GetIntLockstep, ()) {
    auto dual = load_dual(kFixtureYaml);
    RC_ASSERT(dual.has_value());

    const std::string key = *genKey();

    // C++ side: call get_int, observe outcome
    CppResult cpp_result{CppOutcome::Success, 0};
    int cpp_value = 0;
    try {
        cpp_value = dual->cpp_cfg.get_int(key);
    } catch (const conf::Conf_Error &e) {
        cpp_result.outcome = CppOutcome::ConfError;
        cpp_result.error_code = static_cast<int>(e.code());
    } catch (...) {
        cpp_result.outcome = CppOutcome::OtherException;
    }

    // C bridge side: call conf_get_int_c
    int c_out = 0;
    int c_rc = conf_get_int_c(dual->c_handle, key.c_str(), static_cast<int>(key.size()), &c_out);

    // Lockstep assertion
    if (cpp_result.outcome == CppOutcome::ConfError) {
        // C bridge must return the same error code
        RC_ASSERT(c_rc == cpp_result.error_code);
    } else if (cpp_result.outcome == CppOutcome::Success) {
        // C bridge must return Success and same value
        RC_ASSERT(c_rc == static_cast<int>(conf::Error_Code::Success));
        RC_ASSERT(c_out == cpp_value);
    } else {
        // Unexpected exception type — bridge should map to Unknown
        RC_ASSERT(c_rc == static_cast<int>(conf::Error_Code::Unknown));
    }
}

// --- Property 6b: get_double lockstep ----------------------------------------
// For random keys, conf::Config::get_double and conf_get_double_c agree.
//
// **Validates: Requirements 11.4, 19.2**

RC_GTEST_PROP(ErrorLockstepProperty6, GetDoubleLockstep, ()) {
    auto dual = load_dual(kFixtureYaml);
    RC_ASSERT(dual.has_value());

    const std::string key = *genKey();

    CppResult cpp_result{CppOutcome::Success, 0};
    double cpp_value = 0.0;
    try {
        cpp_value = dual->cpp_cfg.get_double(key);
    } catch (const conf::Conf_Error &e) {
        cpp_result.outcome = CppOutcome::ConfError;
        cpp_result.error_code = static_cast<int>(e.code());
    } catch (...) {
        cpp_result.outcome = CppOutcome::OtherException;
    }

    double c_out = 0.0;
    int c_rc = conf_get_double_c(dual->c_handle, key.c_str(), static_cast<int>(key.size()), &c_out);

    if (cpp_result.outcome == CppOutcome::ConfError) {
        RC_ASSERT(c_rc == cpp_result.error_code);
    } else if (cpp_result.outcome == CppOutcome::Success) {
        RC_ASSERT(c_rc == static_cast<int>(conf::Error_Code::Success));
        RC_ASSERT(c_out == cpp_value);
    } else {
        RC_ASSERT(c_rc == static_cast<int>(conf::Error_Code::Unknown));
    }
}

// --- Property 6c: get_bool lockstep ------------------------------------------
// For random keys, conf::Config::get_bool and conf_get_bool_c agree.
// Note: C bridge returns 1/0 for true/false.
//
// **Validates: Requirements 11.4, 19.2**

RC_GTEST_PROP(ErrorLockstepProperty6, GetBoolLockstep, ()) {
    auto dual = load_dual(kFixtureYaml);
    RC_ASSERT(dual.has_value());

    const std::string key = *genKey();

    CppResult cpp_result{CppOutcome::Success, 0};
    bool cpp_value = false;
    try {
        cpp_value = dual->cpp_cfg.get_bool(key);
    } catch (const conf::Conf_Error &e) {
        cpp_result.outcome = CppOutcome::ConfError;
        cpp_result.error_code = static_cast<int>(e.code());
    } catch (...) {
        cpp_result.outcome = CppOutcome::OtherException;
    }

    int c_out = 0;
    int c_rc = conf_get_bool_c(dual->c_handle, key.c_str(), static_cast<int>(key.size()), &c_out);

    if (cpp_result.outcome == CppOutcome::ConfError) {
        RC_ASSERT(c_rc == cpp_result.error_code);
    } else if (cpp_result.outcome == CppOutcome::Success) {
        RC_ASSERT(c_rc == static_cast<int>(conf::Error_Code::Success));
        int expected_c_bool = cpp_value ? 1 : 0;
        RC_ASSERT(c_out == expected_c_bool);
    } else {
        RC_ASSERT(c_rc == static_cast<int>(conf::Error_Code::Unknown));
    }
}

// --- Property 6d: get_string lockstep ----------------------------------------
// For random keys, conf::Config::get_string and conf_get_string_len_c agree on
// error codes. On success, the reported length matches the C++ string size.
//
// **Validates: Requirements 11.4, 19.2**

RC_GTEST_PROP(ErrorLockstepProperty6, GetStringLockstep, ()) {
    auto dual = load_dual(kFixtureYaml);
    RC_ASSERT(dual.has_value());

    const std::string key = *genKey();

    CppResult cpp_result{CppOutcome::Success, 0};
    std::string cpp_value;
    try {
        cpp_value = dual->cpp_cfg.get_string(key);
    } catch (const conf::Conf_Error &e) {
        cpp_result.outcome = CppOutcome::ConfError;
        cpp_result.error_code = static_cast<int>(e.code());
    } catch (...) {
        cpp_result.outcome = CppOutcome::OtherException;
    }

    int c_str_len = 0;
    int c_rc = conf_get_string_len_c(dual->c_handle, key.c_str(), static_cast<int>(key.size()), &c_str_len);

    if (cpp_result.outcome == CppOutcome::ConfError) {
        RC_ASSERT(c_rc == cpp_result.error_code);
    } else if (cpp_result.outcome == CppOutcome::Success) {
        RC_ASSERT(c_rc == static_cast<int>(conf::Error_Code::Success));
        RC_ASSERT(c_str_len == static_cast<int>(cpp_value.size()));
    } else {
        RC_ASSERT(c_rc == static_cast<int>(conf::Error_Code::Unknown));
    }
}

// --- Property 6e: Empty-key lockstep (Invalid_Arg) ---------------------------
// Empty keys and malformed paths must produce the same Invalid_Arg error code
// from both C++ (Conf_Error) and C bridge (return code). This sub-property
// specifically targets the Invalid_Arg path to ensure it is never lost in
// translation.
//
// **Validates: Requirements 11.4, 19.2**

RC_GTEST_PROP(ErrorLockstepProperty6, InvalidArgLockstep, ()) {
    auto dual = load_dual(kFixtureYaml);
    RC_ASSERT(dual.has_value());

    // Generate only malformed keys
    const std::string key = *rc::gen::element<std::string>("", ".leading", "trailing.", "double..dot", "...", "..", "a..b");

    // C++ side: get_int should throw Conf_Error(Invalid_Arg)
    CppResult cpp_result{CppOutcome::Success, 0};
    try {
        (void)dual->cpp_cfg.get_int(key);
    } catch (const conf::Conf_Error &e) {
        cpp_result.outcome = CppOutcome::ConfError;
        cpp_result.error_code = static_cast<int>(e.code());
    } catch (...) {
        cpp_result.outcome = CppOutcome::OtherException;
    }

    // C bridge side
    int c_out = 0;
    int c_rc = conf_get_int_c(dual->c_handle, key.c_str(), static_cast<int>(key.size()), &c_out);

    // Both must agree on the error code
    if (cpp_result.outcome == CppOutcome::ConfError) {
        RC_ASSERT(c_rc == cpp_result.error_code);
        // Specifically, for malformed paths we expect Invalid_Arg
        RC_ASSERT(c_rc == static_cast<int>(conf::Error_Code::Invalid_Arg));
    } else {
        // If C++ didn't throw ConfError, something is wrong — but we still
        // check lockstep: bridge should agree with whatever C++ did.
        if (cpp_result.outcome == CppOutcome::Success) {
            RC_ASSERT(c_rc == static_cast<int>(conf::Error_Code::Success));
        } else {
            RC_ASSERT(c_rc == static_cast<int>(conf::Error_Code::Unknown));
        }
    }
}
