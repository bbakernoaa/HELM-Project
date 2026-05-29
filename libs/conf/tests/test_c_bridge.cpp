// ─── CONF extern "C" Bridge Unit Tests ───────────────────────────────────────
// Exercises the Fortran-facing extern "C" bridge implemented in
// src/fortran/conf_c_interop.cpp. The bridge has no public C header (the only
// consumer is conf_mod.f90 via iso_c_binding), so the prototypes are declared
// here inside an extern "C" block, matching the exact signatures compiled into
// the conf library.
//
// Coverage:
//   * Lifecycle: conf_load_string_c returns Success + a handle > 0; conf_close_c
//     frees it and returns Success.
//   * Each error path returns its exact Error_Code:
//       Invalid_Arg, Bad_Handle, Key_Not_Found, Type_Mismatch, Parse_Error,
//       File_Not_Found.
//   * conf_has_key_c: a missing key is NOT an error (exists=0, Success).
//   * Length-first string round trip: conf_get_string_len_c reports N and
//     conf_get_string_c writes exactly N bytes (no null terminator), sets
//     written=N, returns Success, and the bytes match.
//   * Buffer_Too_Small: a buffer shorter than N writes nothing (sentinel bytes
//     unchanged) and returns Buffer_Too_Small.
//   * A closed handle reused on any handle-taking function returns Bad_Handle.
//
// Feature: conf-config-parser
// Requirements: 17.4, 17.5, 17.6, 20.1, 20.2, 20.3, 21.4, 32.5, 32.6
// ─────────────────────────────────────────────────────────────────────────────

#include <conf/error.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>

// ─── Bridge prototypes (no public C header exists; declare them here) ────────
// These MUST match the definitions in src/fortran/conf_c_interop.cpp exactly.
extern "C" {
int conf_load_c(const char* path, int path_len, int* handle_out);
int conf_load_string_c(const char* yaml_text, int text_len, int* handle_out);
int conf_close_c(int handle);
int conf_has_key_c(int handle, const char* key, int key_len, int* exists_out);
int conf_size_c(int handle, const char* key, int key_len, int* size_out);
int conf_get_int_c(int handle, const char* key, int key_len, int* out);
int conf_get_double_c(int handle, const char* key, int key_len, double* out);
int conf_get_bool_c(int handle, const char* key, int key_len, int* out);
int conf_get_string_len_c(int handle, const char* key, int key_len, int* str_len_out);
int conf_get_string_c(int handle, const char* key, int key_len, char* buf,
                      int buf_cap, int* written_out);
}

namespace {

using conf::Error_Code;

// Convenience: the int value of an Error_Code enumerator (the bridge returns ints).
constexpr int code(Error_Code c) { return static_cast<int>(c); }

// Pass a string literal / char array as the (const char*, int len) pair the
// bridge expects. The trailing NUL is intentionally excluded from the length.
// NOTE: the operand MUST be an array (a string literal or `char[]`); a
// `const char*` would make sizeof return the pointer size, not the length.
#define STR_ARG(s) (s), static_cast<int>(sizeof(s) - 1)

// A representative model-config document. "name" is a string scalar of known
// length; "label" is the non-numeric string used to provoke Type_Mismatch.
// Declared as an array (not a pointer) so STR_ARG's sizeof yields its length.
constexpr char kYaml[] =
    "model:\n"
    "  layers: 42\n"
    "  name: spherical\n"      // 9 bytes: s p h e r i c a l
    "  label: not_a_number\n"
    "  ratio: 3.5\n"
    "  enabled: true\n"
    "grid:\n"
    "  res: 7\n";

// RAII guard that loads kYaml via conf_load_string_c and closes the handle on
// destruction (unless the test already closed it). Keeps each test independent
// of the global Handle_Registry's monotonic token state.
class Loaded_Config {
public:
    Loaded_Config() {
        const int rc = conf_load_string_c(STR_ARG(kYaml), &handle_);
        EXPECT_EQ(rc, code(Error_Code::Success));
        EXPECT_GT(handle_, 0);
    }

    Loaded_Config(const Loaded_Config&)            = delete;
    Loaded_Config& operator=(const Loaded_Config&) = delete;

    ~Loaded_Config() {
        if (handle_ > 0) {
            conf_close_c(handle_);  // best-effort; ignore status during teardown
        }
    }

    [[nodiscard]] int handle() const { return handle_; }

    // Close now and remember that the handle is gone (so the dtor won't reclose).
    int close() {
        const int h = handle_;
        handle_ = 0;
        return conf_close_c(h);
    }

private:
    int handle_{0};
};

} // namespace

// ─── Lifecycle: load string + close (Req 17.1, 21.2) ─────────────────────────

TEST(CBridgeLifecycle, LoadStringReturnsSuccessAndPositiveHandle) {
    int handle = -1;
    const int rc = conf_load_string_c(STR_ARG(kYaml), &handle);

    EXPECT_EQ(rc, code(Error_Code::Success));
    EXPECT_GT(handle, 0);

    EXPECT_EQ(conf_close_c(handle), code(Error_Code::Success));
}

// ─── Invalid_Arg: null / non-positive length arguments (Req 17.5, 16.3) ──────

TEST(CBridgeInvalidArg, LoadRejectsNullPath) {
    int handle = 123;  // sentinel: must stay unchanged on a non-zero return
    EXPECT_EQ(conf_load_c(nullptr, 4, &handle), code(Error_Code::Invalid_Arg));
    EXPECT_EQ(handle, 123);
}

TEST(CBridgeInvalidArg, LoadRejectsNullHandleOut) {
    EXPECT_EQ(conf_load_c("a.yaml", 6, nullptr), code(Error_Code::Invalid_Arg));
}

TEST(CBridgeInvalidArg, LoadRejectsNonPositivePathLen) {
    int handle = 7;
    EXPECT_EQ(conf_load_c("x", 0, &handle), code(Error_Code::Invalid_Arg));
    EXPECT_EQ(conf_load_c("x", -1, &handle), code(Error_Code::Invalid_Arg));
    EXPECT_EQ(handle, 7);
}

TEST(CBridgeInvalidArg, GettersRejectNullKey) {
    Loaded_Config cfg;
    int out = -999;
    EXPECT_EQ(conf_get_int_c(cfg.handle(), nullptr, 5, &out),
              code(Error_Code::Invalid_Arg));
    EXPECT_EQ(out, -999);  // unchanged on non-zero return (Req 17.8)
}

TEST(CBridgeInvalidArg, GettersRejectNullOut) {
    Loaded_Config cfg;
    EXPECT_EQ(conf_get_int_c(cfg.handle(), STR_ARG("model.layers"), nullptr),
              code(Error_Code::Invalid_Arg));
}

TEST(CBridgeInvalidArg, GettersRejectNonPositiveKeyLen) {
    Loaded_Config cfg;
    int out = 55;
    EXPECT_EQ(conf_get_int_c(cfg.handle(), "model.layers", 0, &out),
              code(Error_Code::Invalid_Arg));
    EXPECT_EQ(conf_get_int_c(cfg.handle(), "model.layers", -3, &out),
              code(Error_Code::Invalid_Arg));
    EXPECT_EQ(out, 55);
}

// ─── Bad_Handle: token 0, never-issued, and released tokens (Req 17.7, 21.1) ─

TEST(CBridgeBadHandle, ZeroHandleRejected) {
    int out = 0;
    EXPECT_EQ(conf_get_int_c(0, STR_ARG("model.layers"), &out),
              code(Error_Code::Bad_Handle));
    EXPECT_EQ(conf_size_c(0, STR_ARG("model"), &out),
              code(Error_Code::Bad_Handle));
    EXPECT_EQ(conf_close_c(0), code(Error_Code::Bad_Handle));
}

TEST(CBridgeBadHandle, NeverIssuedHandleRejected) {
    // A very large token that the monotonic registry will not have issued.
    constexpr int kBogus = 2000000000;
    int out = 0;
    EXPECT_EQ(conf_get_int_c(kBogus, STR_ARG("model.layers"), &out),
              code(Error_Code::Bad_Handle));
    EXPECT_EQ(conf_size_c(kBogus, STR_ARG("model"), &out),
              code(Error_Code::Bad_Handle));
    EXPECT_EQ(conf_close_c(kBogus), code(Error_Code::Bad_Handle));
}

// ─── Bad_Handle: a closed handle is invalid forever on reuse (Req 21.4) ──────

TEST(CBridgeBadHandle, ClosedHandleReusedReturnsBadHandle) {
    Loaded_Config cfg;
    const int handle = cfg.handle();

    // First close succeeds.
    EXPECT_EQ(cfg.close(), code(Error_Code::Success));

    // Every subsequent handle-taking call on the same token is Bad_Handle.
    int out = -1;
    EXPECT_EQ(conf_get_int_c(handle, STR_ARG("model.layers"), &out),
              code(Error_Code::Bad_Handle));
    EXPECT_EQ(conf_size_c(handle, STR_ARG("model"), &out),
              code(Error_Code::Bad_Handle));
    int exists = -1;
    EXPECT_EQ(conf_has_key_c(handle, STR_ARG("model.layers"), &exists),
              code(Error_Code::Bad_Handle));
    // Re-closing an already-closed handle is also Bad_Handle.
    EXPECT_EQ(conf_close_c(handle), code(Error_Code::Bad_Handle));
}

// ─── Key_Not_Found: getter on a missing key (Req 13.x via getter) ────────────

TEST(CBridgeKeyNotFound, GetIntOnMissingKey) {
    Loaded_Config cfg;
    int out = 4242;
    EXPECT_EQ(conf_get_int_c(cfg.handle(), STR_ARG("model.does_not_exist"), &out),
              code(Error_Code::Key_Not_Found));
    EXPECT_EQ(out, 4242);  // unchanged on non-zero return (Req 17.8)
}

// ─── Type_Mismatch: integer requested from a non-numeric string (Req 15.x) ───

TEST(CBridgeTypeMismatch, GetIntOnNonNumericString) {
    Loaded_Config cfg;
    int out = 9;
    EXPECT_EQ(conf_get_int_c(cfg.handle(), STR_ARG("model.label"), &out),
              code(Error_Code::Type_Mismatch));
    EXPECT_EQ(out, 9);  // unchanged on non-zero return (Req 17.8)
}

// ─── Parse_Error: malformed YAML at load (Req 14.3) ──────────────────────────

TEST(CBridgeParseError, LoadStringWithMalformedYaml) {
    constexpr char kBad[] = "key: [1, 2";  // unclosed flow sequence
    int handle = 77;
    EXPECT_EQ(conf_load_string_c(STR_ARG(kBad), &handle),
              code(Error_Code::Parse_Error));
    EXPECT_EQ(handle, 77);  // no handle registered on a failed load (Req 14.3)
}

// ─── File_Not_Found: load a nonexistent path (Req 1.4 via bridge) ────────────

TEST(CBridgeFileNotFound, LoadNonexistentPath) {
    const auto missing =
        (std::filesystem::temp_directory_path() /
         "conf_c_bridge_missing_9a3f.yaml")
            .string();
    ASSERT_FALSE(std::filesystem::exists(missing));

    int handle = 88;
    EXPECT_EQ(conf_load_c(missing.c_str(), static_cast<int>(missing.size()),
                          &handle),
              code(Error_Code::File_Not_Found));
    EXPECT_EQ(handle, 88);  // unchanged on a failed load
}

// ─── conf_has_key_c: present vs. missing key (Req 17.6) ──────────────────────

TEST(CBridgeHasKey, ExistingKeySetsOneAndSucceeds) {
    Loaded_Config cfg;
    int exists = -1;
    EXPECT_EQ(conf_has_key_c(cfg.handle(), STR_ARG("model.layers"), &exists),
              code(Error_Code::Success));
    EXPECT_EQ(exists, 1);
}

TEST(CBridgeHasKey, MissingKeySetsZeroAndSucceeds) {
    Loaded_Config cfg;
    int exists = -1;
    EXPECT_EQ(conf_has_key_c(cfg.handle(), STR_ARG("model.nope"), &exists),
              code(Error_Code::Success));
    EXPECT_EQ(exists, 0);  // a missing key is NOT an error here
}

// ─── Length-first string round trip (Req 20.1, 20.2, 32.5) ───────────────────

TEST(CBridgeStringRoundTrip, LenThenCopyWritesExactlyNBytesNoTerminator) {
    Loaded_Config cfg;
    const std::string expected = "spherical";  // value of model.name
    const int n = static_cast<int>(expected.size());

    // Step 1: query the exact byte count.
    int reported_len = -1;
    EXPECT_EQ(conf_get_string_len_c(cfg.handle(), STR_ARG("model.name"),
                                    &reported_len),
              code(Error_Code::Success));
    EXPECT_EQ(reported_len, n);

    // Step 2: copy into a buffer that is exactly N bytes. The bridge must write
    // exactly N bytes and must NOT null-terminate. Use a guard byte after the
    // N-byte region to prove no terminator (or overrun) is written.
    std::array<char, 64> buf{};
    constexpr char kFill = '\x7f';
    buf.fill(kFill);

    int written = -1;
    EXPECT_EQ(conf_get_string_c(cfg.handle(), STR_ARG("model.name"), buf.data(),
                                n, &written),
              code(Error_Code::Success));
    EXPECT_EQ(written, n);

    // The N value bytes match exactly.
    EXPECT_EQ(std::string(buf.data(), static_cast<std::size_t>(n)), expected);

    // The byte immediately past the value was NOT touched (no NUL terminator).
    EXPECT_EQ(buf[static_cast<std::size_t>(n)], kFill);
}

TEST(CBridgeStringRoundTrip, LargerBufferStillWritesExactlyNBytes) {
    Loaded_Config cfg;
    const std::string expected = "spherical";
    const int n = static_cast<int>(expected.size());

    std::array<char, 64> buf{};
    constexpr char kFill = '\x7f';
    buf.fill(kFill);

    int written = -1;
    EXPECT_EQ(conf_get_string_c(cfg.handle(), STR_ARG("model.name"), buf.data(),
                                static_cast<int>(buf.size()), &written),
              code(Error_Code::Success));
    EXPECT_EQ(written, n);
    EXPECT_EQ(std::string(buf.data(), static_cast<std::size_t>(n)), expected);
    EXPECT_EQ(buf[static_cast<std::size_t>(n)], kFill);  // no terminator
}

// ─── Buffer_Too_Small: short buffer writes nothing (Req 20.3, 32.6) ──────────

TEST(CBridgeBufferTooSmall, ShortBufferWritesNothingAndLeavesWrittenUnchanged) {
    Loaded_Config cfg;
    const std::string expected = "spherical";
    const int n = static_cast<int>(expected.size());

    // Pre-fill the buffer with a sentinel pattern; it must remain untouched.
    std::array<char, 64> buf{};
    constexpr char kSentinel = '\x5a';
    buf.fill(kSentinel);

    int written = -12345;  // sentinel: must remain unchanged (Req 20.3)
    EXPECT_EQ(conf_get_string_c(cfg.handle(), STR_ARG("model.name"), buf.data(),
                                n - 1, &written),
              code(Error_Code::Buffer_Too_Small));

    // written_out left unchanged.
    EXPECT_EQ(written, -12345);

    // The entire buffer is still the sentinel pattern: nothing was written.
    for (char c : buf) {
        EXPECT_EQ(c, kSentinel);
    }
}
