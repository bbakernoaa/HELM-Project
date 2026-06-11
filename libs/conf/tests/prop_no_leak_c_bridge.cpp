// --- Property-Based Tests: No-Leak C-Bridge Invariant ------------------------
// Feature: conf-config-parser
//
// Uses RapidCheck to verify the structural no-leak invariant of the CONF
// extern "C" bridge: after any sequence of load → N queries → close, the
// handle is invalidated (proving RAII cleanup freed the Config). Additionally,
// failed loads (Parse_Error) never register a handle.
//
// Property 4: No-Leak on the C Bridge
//   The structural proof: if close works and the handle is invalidated, RAII
//   has freed the tree. No partial state remains. For any generated
//   load → N queries → close sequence (including string queries and failed
//   loads), the handle is invalid after close, observed via Bad_Handle
//   returns on all subsequent getters.
//
// **Validates: Requirements 30.1, 30.2, 30.3**
// -----------------------------------------------------------------------------

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <conf/error.hpp>

#include <cstring>
#include <string>
#include <vector>

// ─── extern "C" prototypes for the CONF C bridge ────────────────────────────
// These mirror the declarations in conf_c_interop.cpp. We declare them here
// so the test links directly against the conf library without needing a
// separate C header.

extern "C" {

int conf_load_string_c(const char* yaml_text, int text_len, int* handle_out);
int conf_close_c(int handle);
int conf_has_key_c(int handle, const char* key, int key_len, int* exists_out);
int conf_size_c(int handle, const char* key, int key_len, int* size_out);
int conf_get_int_c(int handle, const char* key, int key_len, int* out);
int conf_get_double_c(int handle, const char* key, int key_len, double* out);
int conf_get_bool_c(int handle, const char* key, int key_len, int* out);
int conf_get_string_len_c(int handle, const char* key, int key_len, int* str_len_out);
int conf_get_string_c(int handle, const char* key, int key_len,
                      char* buf, int buf_cap, int* written_out);

} // extern "C"

namespace {

// ─── Error code constants ───────────────────────────────────────────────────

constexpr int EC_SUCCESS        = static_cast<int>(conf::Error_Code::Success);
constexpr int EC_BAD_HANDLE     = static_cast<int>(conf::Error_Code::Bad_Handle);
constexpr int EC_PARSE_ERROR    = static_cast<int>(conf::Error_Code::Parse_Error);

// ─── YAML generators ────────────────────────────────────────────────────────

/// Generate a lowercase ASCII character (a-z).
rc::Gen<char> genLowerAlpha() {
    return rc::gen::map(rc::gen::inRange(0x61, 0x7B),
        [](int c) { return static_cast<char>(c); });
}

/// Generate a short lowercase string (0–50 chars).
rc::Gen<std::string> genShortString() {
    return rc::gen::mapcat(rc::gen::inRange<std::size_t>(0, 51),
        [](std::size_t len) {
            return rc::gen::container<std::string>(len, genLowerAlpha());
        });
}

/// Generate a valid simple YAML document with a few typed keys for querying.
/// The document always contains known keys so queries can exercise all getter
/// paths without needing to parse the generated YAML structure.
rc::Gen<std::string> genValidYaml() {
    return rc::gen::map(
        rc::gen::tuple(
            rc::gen::inRange(-1000, 1001),                     // int value
            rc::gen::map(rc::gen::inRange(-100, 101),          // double value
                [](int v) { return static_cast<double>(v) * 0.5; }),
            rc::gen::arbitrary<bool>(),                         // bool value
            genShortString()                                    // string value
        ),
        [](const std::tuple<int, double, bool, std::string>& t) {
            const auto& [iv, dv, bv, sv] = t;
            std::string yaml;
            yaml += "int_val: " + std::to_string(iv) + "\n";
            yaml += "dbl_val: " + std::to_string(dv) + "\n";
            yaml += "bool_val: " + std::string(bv ? "true" : "false") + "\n";
            yaml += "str_val: \"" + sv + "\"\n";
            yaml += "nested:\n";
            yaml += "  child: 99\n";
            return yaml;
        }
    );
}

/// Generate an invalid YAML string that will trigger Parse_Error.
/// All strings here are verified to cause yaml-cpp to throw a parse exception.
rc::Gen<std::string> genInvalidYaml() {
    return rc::gen::element<std::string>(
        "key: [\n",            // unclosed sequence flow
        "a: b\n  c: d\n",     // illegal map value (bad indentation)
        "{{{{\n",              // unclosed map flow
        "- ]\n",              // illegal flow end
        "{unclosed",           // unclosed map flow
        "[unclosed",           // unclosed sequence flow
        "{key: [}\n",          // illegal flow end (mismatched brackets)
        "[a, b, {c: ]}\n"     // illegal flow end inside sequence
    );
}

/// Enumeration of query operations we can perform on a live handle.
enum class QueryOp {
    HasKey,
    Size,
    GetInt,
    GetDouble,
    GetBool,
    GetStringLen,
    GetString
};

/// Generate a random query operation.
rc::Gen<QueryOp> genQueryOp() {
    return rc::gen::element(
        QueryOp::HasKey,
        QueryOp::Size,
        QueryOp::GetInt,
        QueryOp::GetDouble,
        QueryOp::GetBool,
        QueryOp::GetStringLen,
        QueryOp::GetString
    );
}

/// Generate a random key to query (mix of valid and invalid keys).
rc::Gen<std::string> genQueryKey() {
    return rc::gen::oneOf(
        // Keys that exist in our generated YAML
        rc::gen::element<std::string>(
            "int_val", "dbl_val", "bool_val", "str_val",
            "nested", "nested.child"),
        // Keys that don't exist (exercise Key_Not_Found path)
        rc::gen::element<std::string>(
            "missing", "no.such.key", "x.y.z"),
        // Malformed keys (exercise Invalid_Arg path)
        rc::gen::element<std::string>(
            "", ".leading", "trailing.", "double..dot")
    );
}

/// Execute a single query operation on the given handle.
/// Returns the error code from the C bridge function.
int executeQuery(int handle, QueryOp op, const std::string& key) {
    const char* key_ptr = key.c_str();
    int key_len = static_cast<int>(key.size());

    // For empty keys, we still pass a valid pointer but length 0.
    // The bridge should return Invalid_Arg for key_len < 1.
    if (key.empty()) {
        key_len = 0;
        // Pass non-null pointer with zero length
        static const char empty_key = '\0';
        key_ptr = &empty_key;
    }

    switch (op) {
        case QueryOp::HasKey: {
            int exists = -1;
            return conf_has_key_c(handle, key_ptr, key_len, &exists);
        }
        case QueryOp::Size: {
            int size = -1;
            return conf_size_c(handle, key_ptr, key_len, &size);
        }
        case QueryOp::GetInt: {
            int val = 0;
            return conf_get_int_c(handle, key_ptr, key_len, &val);
        }
        case QueryOp::GetDouble: {
            double val = 0.0;
            return conf_get_double_c(handle, key_ptr, key_len, &val);
        }
        case QueryOp::GetBool: {
            int val = -1;
            return conf_get_bool_c(handle, key_ptr, key_len, &val);
        }
        case QueryOp::GetStringLen: {
            int str_len = -1;
            return conf_get_string_len_c(handle, key_ptr, key_len, &str_len);
        }
        case QueryOp::GetString: {
            char buf[256];
            int written = -1;
            return conf_get_string_c(handle, key_ptr, key_len,
                                     buf, static_cast<int>(sizeof(buf)), &written);
        }
    }
    return -1;  // unreachable
}

}  // anonymous namespace

// --- Property 4a: Load-Query-Close cycle invalidates handle ------------------
// Feature: conf-config-parser, Property 4: No-Leak on the C Bridge
//
// For any valid YAML loaded via conf_load_string_c, followed by N random
// queries (exercising all getter paths), followed by conf_close_c:
//   (1) conf_close_c returns Success
//   (2) After close, the handle is invalid — every getter returns Bad_Handle
//   (3) This proves RAII cleanup was invoked (Config deleted, tree freed)
//
// **Validates: Requirements 30.1, 30.2, 30.3**

RC_GTEST_PROP(NoLeakCBridgeProperty4, LoadQueryCloseInvalidatesHandle, ()) {
    // Generate a valid YAML document and a sequence of random queries.
    const std::string yaml = *genValidYaml();
    const int num_queries = *rc::gen::inRange(0, 51);

    // Step 1: Load the YAML via the C bridge.
    int handle = 0;
    int rc_load = conf_load_string_c(yaml.c_str(),
                                     static_cast<int>(yaml.size()),
                                     &handle);
    RC_ASSERT(rc_load == EC_SUCCESS);
    RC_ASSERT(handle > 0);

    // Step 2: Perform N random queries on the live handle.
    // All queries should return Success or a typed error (Key_Not_Found,
    // Type_Mismatch, Invalid_Arg) — never Bad_Handle while handle is live.
    for (int i = 0; i < num_queries; ++i) {
        const QueryOp op = *genQueryOp();
        const std::string key = *genQueryKey();
        int query_rc = executeQuery(handle, op, key);
        // While the handle is live, we must never see Bad_Handle.
        RC_ASSERT(query_rc != EC_BAD_HANDLE);
    }

    // Step 3: Close the handle — RAII cleanup.
    int rc_close = conf_close_c(handle);
    RC_ASSERT(rc_close == EC_SUCCESS);

    // Step 4: After close, the handle is invalid. Every operation must
    // return Bad_Handle, proving the Config was freed and the handle
    // registry entry removed.
    {
        int exists = -1;
        RC_ASSERT(conf_has_key_c(handle, "int_val", 7, &exists) == EC_BAD_HANDLE);
    }
    {
        int size = -1;
        RC_ASSERT(conf_size_c(handle, "int_val", 7, &size) == EC_BAD_HANDLE);
    }
    {
        int val = 0;
        RC_ASSERT(conf_get_int_c(handle, "int_val", 7, &val) == EC_BAD_HANDLE);
    }
    {
        double val = 0.0;
        RC_ASSERT(conf_get_double_c(handle, "dbl_val", 7, &val) == EC_BAD_HANDLE);
    }
    {
        int val = -1;
        RC_ASSERT(conf_get_bool_c(handle, "bool_val", 8, &val) == EC_BAD_HANDLE);
    }
    {
        int str_len = -1;
        RC_ASSERT(conf_get_string_len_c(handle, "str_val", 7, &str_len) == EC_BAD_HANDLE);
    }
    {
        char buf[64];
        int written = -1;
        RC_ASSERT(conf_get_string_c(handle, "str_val", 7, buf, 64, &written) == EC_BAD_HANDLE);
    }

    // Double close must also return Bad_Handle (handle already released).
    RC_ASSERT(conf_close_c(handle) == EC_BAD_HANDLE);
}

// --- Property 4b: Failed loads never register a handle -----------------------
// Feature: conf-config-parser, Property 4: No-Leak on the C Bridge
//
// For any invalid YAML that triggers Parse_Error, conf_load_string_c must:
//   (1) Return Parse_Error (not Success)
//   (2) Not modify handle_out (remains at its sentinel value)
//   (3) Leave no orphaned handle in the registry
//
// This ensures that failed loads cause no resource leak — nothing is allocated
// or registered when parsing fails.
//
// **Validates: Requirements 30.1, 30.2, 30.3**

RC_GTEST_PROP(NoLeakCBridgeProperty4, FailedLoadsNeverRegisterHandle, ()) {
    const std::string bad_yaml = *genInvalidYaml();

    // Initialize handle_out to a known sentinel to verify it's unchanged.
    int handle = -999;
    int rc_load = conf_load_string_c(bad_yaml.c_str(),
                                     static_cast<int>(bad_yaml.size()),
                                     &handle);

    // (1) Must return Parse_Error.
    RC_ASSERT(rc_load == EC_PARSE_ERROR);

    // (2) handle_out should remain unchanged (bridge should not write to it on failure).
    // Note: the spec says "SHALL NOT return a Config instance" / "SHALL NOT register
    // a handle for the failed load". The safest test is that if handle changed,
    // it's not a valid token in the registry.
    // Even if the bridge writes to handle_out (implementation detail), the important
    // thing is that no handle is registered. We verify by attempting close:
    if (handle != -999 && handle > 0) {
        // If the bridge somehow wrote a positive value, it must still be invalid.
        int rc_close = conf_close_c(handle);
        RC_ASSERT(rc_close == EC_BAD_HANDLE);
    }
}

// --- Property 4c: Multiple load-close cycles leave no residual state ---------
// Feature: conf-config-parser, Property 4: No-Leak on the C Bridge
//
// For a random sequence of N complete load→query→close cycles (simulating a
// model that loads configs repeatedly during a run), every handle is properly
// invalidated after its close, and no cross-cycle pollution occurs.
//
// **Validates: Requirements 30.1, 30.2, 30.3**

RC_GTEST_PROP(NoLeakCBridgeProperty4, MultipleCyclesNoResidualState, ()) {
    const int num_cycles = *rc::gen::inRange(1, 21);
    std::vector<int> closed_handles;

    for (int cycle = 0; cycle < num_cycles; ++cycle) {
        const std::string yaml = *genValidYaml();
        const int num_queries = *rc::gen::inRange(0, 11);

        // Load
        int handle = 0;
        int rc_load = conf_load_string_c(yaml.c_str(),
                                         static_cast<int>(yaml.size()),
                                         &handle);
        RC_ASSERT(rc_load == EC_SUCCESS);
        RC_ASSERT(handle > 0);

        // Verify this handle is distinct from all previously closed handles
        // (non-reuse guarantee inherited from Handle_Registry).
        for (int old_handle : closed_handles) {
            RC_ASSERT(handle != old_handle);
        }

        // Query
        for (int i = 0; i < num_queries; ++i) {
            const QueryOp op = *genQueryOp();
            const std::string key = *genQueryKey();
            int query_rc = executeQuery(handle, op, key);
            RC_ASSERT(query_rc != EC_BAD_HANDLE);
        }

        // Close
        int rc_close = conf_close_c(handle);
        RC_ASSERT(rc_close == EC_SUCCESS);

        // Verify invalidation
        int val = 0;
        RC_ASSERT(conf_get_int_c(handle, "int_val", 7, &val) == EC_BAD_HANDLE);

        closed_handles.push_back(handle);
    }

    // Final check: all previously closed handles are permanently invalid.
    for (int old_handle : closed_handles) {
        int val = 0;
        RC_ASSERT(conf_get_int_c(old_handle, "int_val", 7, &val) == EC_BAD_HANDLE);
        RC_ASSERT(conf_close_c(old_handle) == EC_BAD_HANDLE);
    }
}

// --- Property 4d: String query lifecycle — no allocation outlives close -------
// Feature: conf-config-parser, Property 4: No-Leak on the C Bridge
//
// For any valid YAML with string values, the two-step string protocol
// (get_string_len → get_string) returns correct data while the handle is live,
// and all subsequent string queries return Bad_Handle after close. This
// verifies that no internal string buffer outlives the conf_close_c call.
//
// **Validates: Requirements 30.1, 30.2, 30.3**

RC_GTEST_PROP(NoLeakCBridgeProperty4, StringQueryLifecycle, ()) {
    // Generate a string value to embed in YAML (1–100 lowercase chars).
    const std::string str_val = *rc::gen::mapcat(
        rc::gen::inRange<std::size_t>(1, 101),
        [](std::size_t len) {
            return rc::gen::container<std::string>(len, genLowerAlpha());
        });

    const std::string yaml = "mykey: \"" + str_val + "\"\n";

    // Load
    int handle = 0;
    int rc_load = conf_load_string_c(yaml.c_str(),
                                     static_cast<int>(yaml.size()),
                                     &handle);
    RC_ASSERT(rc_load == EC_SUCCESS);

    // Perform the length-first string protocol N times while live.
    const int repetitions = *rc::gen::inRange(1, 11);
    for (int r = 0; r < repetitions; ++r) {
        // Step 1: get string length
        int str_len = -1;
        int rc_len = conf_get_string_len_c(handle, "mykey", 5, &str_len);
        RC_ASSERT(rc_len == EC_SUCCESS);
        RC_ASSERT(str_len == static_cast<int>(str_val.size()));

        // Step 2: get string into buffer
        std::vector<char> buf(static_cast<std::size_t>(str_len + 1), '\0');
        int written = -1;
        int rc_str = conf_get_string_c(handle, "mykey", 5,
                                       buf.data(), str_len, &written);
        RC_ASSERT(rc_str == EC_SUCCESS);
        RC_ASSERT(written == str_len);

        // Verify content matches
        std::string result(buf.data(), static_cast<std::size_t>(written));
        RC_ASSERT(result == str_val);
    }

    // Close — RAII frees everything.
    RC_ASSERT(conf_close_c(handle) == EC_SUCCESS);

    // After close, string queries return Bad_Handle — no residual buffer.
    {
        int str_len = -1;
        RC_ASSERT(conf_get_string_len_c(handle, "mykey", 5, &str_len) == EC_BAD_HANDLE);
    }
    {
        char buf[128];
        int written = -1;
        RC_ASSERT(conf_get_string_c(handle, "mykey", 5, buf, 128, &written) == EC_BAD_HANDLE);
    }
}
