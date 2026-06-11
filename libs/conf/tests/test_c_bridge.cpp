// ─── Unit Tests: C Bridge (extern "C" interop) ──────────────────────────────
// Validates the extern "C" API surface that exposes CONF to Fortran via opaque
// integer handles and int error codes.
//
// Requirements: 17.4, 17.5, 17.6, 20.1, 20.2, 20.3, 21.4, 32.5, 32.6
// ──────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <conf/error.hpp>

#include <cstring>
#include <string>
#include <vector>

// ─── Declare the extern "C" function prototypes ─────────────────────────────
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
int conf_get_string_c(int handle, const char* key, int key_len,
                      char* buf, int buf_cap, int* written_out);
}

// ─── Convenience aliases for Error_Code integer values ──────────────────────
static constexpr int SUCCESS        = static_cast<int>(conf::Error_Code::Success);
static constexpr int INVALID_ARG    = static_cast<int>(conf::Error_Code::Invalid_Arg);
static constexpr int FILE_NOT_FOUND = static_cast<int>(conf::Error_Code::File_Not_Found);
static constexpr int PARSE_ERROR    = static_cast<int>(conf::Error_Code::Parse_Error);
static constexpr int KEY_NOT_FOUND  = static_cast<int>(conf::Error_Code::Key_Not_Found);
static constexpr int TYPE_MISMATCH  = static_cast<int>(conf::Error_Code::Type_Mismatch);
static constexpr int BAD_HANDLE     = static_cast<int>(conf::Error_Code::Bad_Handle);
static constexpr int BUFFER_TOO_SMALL = static_cast<int>(conf::Error_Code::Buffer_Too_Small);

// ─── Valid YAML for use in tests ────────────────────────────────────────────
static const std::string VALID_YAML =
    "model:\n"
    "  name: test_model\n"
    "  layers: 42\n"
    "  rate: 3.14\n"
    "  enabled: true\n"
    "  disabled: false\n";

// ─── Test 1: conf_load_string_c with valid YAML returns Success ─────────────
// Requirement 17.5: conf_load_string_c returns Success and a positive handle.
TEST(CBridge, LoadStringValidYamlReturnsSuccess) {
    int handle = 0;
    int rc = conf_load_string_c(VALID_YAML.c_str(),
                                static_cast<int>(VALID_YAML.size()),
                                &handle);
    EXPECT_EQ(rc, SUCCESS);
    EXPECT_GT(handle, 0);

    // Clean up
    conf_close_c(handle);
}

// ─── Test 2: conf_load_c with non-existent path returns File_Not_Found ──────
// Requirement 20.1: non-existent file path returns File_Not_Found.
TEST(CBridge, LoadFileNotFoundReturnsFileNotFound) {
    const std::string path = "/no/such/path/config.yaml";
    int handle = 0;
    int rc = conf_load_c(path.c_str(), static_cast<int>(path.size()), &handle);
    EXPECT_EQ(rc, FILE_NOT_FOUND);
}

// ─── Test 3: conf_load_string_c with malformed YAML returns Parse_Error ─────
// Requirement 20.2: malformed YAML returns Parse_Error.
TEST(CBridge, LoadStringMalformedYamlReturnsParseError) {
    const std::string bad_yaml = "key: [unclosed bracket";
    int handle = 0;
    int rc = conf_load_string_c(bad_yaml.c_str(),
                                static_cast<int>(bad_yaml.size()),
                                &handle);
    EXPECT_EQ(rc, PARSE_ERROR);
}

// ─── Test 4: Null path pointer returns Invalid_Arg ──────────────────────────
// Requirement 17.5: null required pointer returns Invalid_Arg.
TEST(CBridge, NullPathReturnsInvalidArg) {
    int handle = 0;
    int rc = conf_load_c(nullptr, 5, &handle);
    EXPECT_EQ(rc, INVALID_ARG);
}

// ─── Test 5: Null handle_out pointer returns Invalid_Arg ────────────────────
// Requirement 17.5: null handle_out returns Invalid_Arg.
TEST(CBridge, NullHandleOutReturnsInvalidArg) {
    const std::string path = "/some/path.yaml";
    int rc = conf_load_c(path.c_str(), static_cast<int>(path.size()), nullptr);
    EXPECT_EQ(rc, INVALID_ARG);
}

// ─── Test 6: path_len < 1 returns Invalid_Arg ───────────────────────────────
// Requirement 17.5: path_len < 1 returns Invalid_Arg.
TEST(CBridge, ZeroPathLenReturnsInvalidArg) {
    const std::string path = "anything";
    int handle = 0;
    int rc = conf_load_c(path.c_str(), 0, &handle);
    EXPECT_EQ(rc, INVALID_ARG);
}

// ─── Test 7: Handle 0 returns Bad_Handle for any getter ─────────────────────
// Requirement 17.6, 20.3: token 0 (CONF_HANDLE_INVALID) returns Bad_Handle.
TEST(CBridge, HandleZeroReturnsBadHandle) {
    const std::string key = "model.layers";
    int int_out = 0;
    double dbl_out = 0.0;
    int bool_out = 0;
    int str_len = 0;

    EXPECT_EQ(conf_get_int_c(0, key.c_str(), static_cast<int>(key.size()), &int_out),
              BAD_HANDLE);
    EXPECT_EQ(conf_get_double_c(0, key.c_str(), static_cast<int>(key.size()), &dbl_out),
              BAD_HANDLE);
    EXPECT_EQ(conf_get_bool_c(0, key.c_str(), static_cast<int>(key.size()), &bool_out),
              BAD_HANDLE);
    EXPECT_EQ(conf_get_string_len_c(0, key.c_str(), static_cast<int>(key.size()), &str_len),
              BAD_HANDLE);
}

// ─── Test 8: Released handle returns Bad_Handle ─────────────────────────────
// Requirement 21.4, 32.6: a released handle returns Bad_Handle on reuse.
TEST(CBridge, ReleasedHandleReturnsBadHandle) {
    int handle = 0;
    int rc = conf_load_string_c(VALID_YAML.c_str(),
                                static_cast<int>(VALID_YAML.size()),
                                &handle);
    ASSERT_EQ(rc, SUCCESS);
    ASSERT_GT(handle, 0);

    // Close the handle
    rc = conf_close_c(handle);
    ASSERT_EQ(rc, SUCCESS);

    // Now all getters should return Bad_Handle
    const std::string key = "model.layers";
    int int_out = 0;
    EXPECT_EQ(conf_get_int_c(handle, key.c_str(), static_cast<int>(key.size()), &int_out),
              BAD_HANDLE);

    int exists = 0;
    EXPECT_EQ(conf_has_key_c(handle, key.c_str(), static_cast<int>(key.size()), &exists),
              BAD_HANDLE);
}

// ─── Test 9: conf_get_int_c on a valid key returns the value ────────────────
// Requirement 17.4, 32.5: successful scalar getter returns the correct value.
TEST(CBridge, GetIntValidKeyReturnsValue) {
    int handle = 0;
    int rc = conf_load_string_c(VALID_YAML.c_str(),
                                static_cast<int>(VALID_YAML.size()),
                                &handle);
    ASSERT_EQ(rc, SUCCESS);

    const std::string key = "model.layers";
    int out = 0;
    rc = conf_get_int_c(handle, key.c_str(), static_cast<int>(key.size()), &out);
    EXPECT_EQ(rc, SUCCESS);
    EXPECT_EQ(out, 42);

    conf_close_c(handle);
}

// ─── Test 10: conf_get_int_c on a missing key returns Key_Not_Found ─────────
// Requirement 17.4, 32.5: missing key returns Key_Not_Found.
TEST(CBridge, GetIntMissingKeyReturnsKeyNotFound) {
    int handle = 0;
    int rc = conf_load_string_c(VALID_YAML.c_str(),
                                static_cast<int>(VALID_YAML.size()),
                                &handle);
    ASSERT_EQ(rc, SUCCESS);

    const std::string key = "model.nonexistent";
    int out = 0;
    rc = conf_get_int_c(handle, key.c_str(), static_cast<int>(key.size()), &out);
    EXPECT_EQ(rc, KEY_NOT_FOUND);

    conf_close_c(handle);
}

// ─── Test 11: conf_get_int_c on a non-numeric value returns Type_Mismatch ───
// Requirement 17.4, 32.5: type mismatch returns Type_Mismatch.
TEST(CBridge, GetIntNonNumericReturnsTypeMismatch) {
    int handle = 0;
    int rc = conf_load_string_c(VALID_YAML.c_str(),
                                static_cast<int>(VALID_YAML.size()),
                                &handle);
    ASSERT_EQ(rc, SUCCESS);

    // "model.name" is "test_model" — not an integer
    const std::string key = "model.name";
    int out = 0;
    rc = conf_get_int_c(handle, key.c_str(), static_cast<int>(key.size()), &out);
    EXPECT_EQ(rc, TYPE_MISMATCH);

    conf_close_c(handle);
}

// ─── Test 12: conf_get_bool_c returns 1 for true, 0 for false ───────────────
// Requirement 17.4: conf_get_bool_c writes 1 for true and 0 for false.
TEST(CBridge, GetBoolReturnsTrueAndFalse) {
    int handle = 0;
    int rc = conf_load_string_c(VALID_YAML.c_str(),
                                static_cast<int>(VALID_YAML.size()),
                                &handle);
    ASSERT_EQ(rc, SUCCESS);

    // Test true
    const std::string key_true = "model.enabled";
    int bool_out = -1;
    rc = conf_get_bool_c(handle, key_true.c_str(),
                         static_cast<int>(key_true.size()), &bool_out);
    EXPECT_EQ(rc, SUCCESS);
    EXPECT_EQ(bool_out, 1);

    // Test false
    const std::string key_false = "model.disabled";
    bool_out = -1;
    rc = conf_get_bool_c(handle, key_false.c_str(),
                         static_cast<int>(key_false.size()), &bool_out);
    EXPECT_EQ(rc, SUCCESS);
    EXPECT_EQ(bool_out, 0);

    conf_close_c(handle);
}

// ─── Test 13: String length-first protocol round trip ───────────────────────
// Requirement 17.4, 32.5: conf_get_string_len_c reports N, then
// conf_get_string_c with buf_cap=N writes exactly N bytes and returns Success.
TEST(CBridge, StringLengthFirstProtocolRoundTrip) {
    int handle = 0;
    int rc = conf_load_string_c(VALID_YAML.c_str(),
                                static_cast<int>(VALID_YAML.size()),
                                &handle);
    ASSERT_EQ(rc, SUCCESS);

    const std::string key = "model.name";
    const std::string expected = "test_model";

    // Step 1: query length
    int str_len = 0;
    rc = conf_get_string_len_c(handle, key.c_str(),
                               static_cast<int>(key.size()), &str_len);
    EXPECT_EQ(rc, SUCCESS);
    EXPECT_EQ(str_len, static_cast<int>(expected.size()));

    // Step 2: fetch string with exactly enough capacity
    std::vector<char> buf(static_cast<std::size_t>(str_len), '\0');
    int written = 0;
    rc = conf_get_string_c(handle, key.c_str(), static_cast<int>(key.size()),
                           buf.data(), str_len, &written);
    EXPECT_EQ(rc, SUCCESS);
    EXPECT_EQ(written, str_len);
    EXPECT_EQ(std::string(buf.data(), static_cast<std::size_t>(written)), expected);

    conf_close_c(handle);
}

// ─── Test 14: Buffer too small writes nothing and returns Buffer_Too_Small ──
// Requirement 17.4, 32.5: buf_cap < N returns Buffer_Too_Small, writes nothing.
TEST(CBridge, BufferTooSmallReturnsBufferTooSmall) {
    int handle = 0;
    int rc = conf_load_string_c(VALID_YAML.c_str(),
                                static_cast<int>(VALID_YAML.size()),
                                &handle);
    ASSERT_EQ(rc, SUCCESS);

    const std::string key = "model.name";  // "test_model" = 10 chars

    // Query length first to know the required size
    int str_len = 0;
    rc = conf_get_string_len_c(handle, key.c_str(),
                               static_cast<int>(key.size()), &str_len);
    ASSERT_EQ(rc, SUCCESS);
    ASSERT_GT(str_len, 0);

    // Provide a buffer that is too small (one byte less than needed)
    int small_cap = str_len - 1;
    std::vector<char> buf(static_cast<std::size_t>(small_cap), 'X');
    int written = -1;  // sentinel
    rc = conf_get_string_c(handle, key.c_str(), static_cast<int>(key.size()),
                           buf.data(), small_cap, &written);
    EXPECT_EQ(rc, BUFFER_TOO_SMALL);

    // Buffer should be untouched (all 'X')
    for (int i = 0; i < small_cap; ++i) {
        EXPECT_EQ(buf[static_cast<std::size_t>(i)], 'X')
            << "Buffer byte " << i << " was modified despite Buffer_Too_Small";
    }

    conf_close_c(handle);
}

// ─── Test 15: conf_close_c returns Success, then handle is Bad_Handle ───────
// Requirement 21.4, 32.6: close returns Success; reuse returns Bad_Handle.
TEST(CBridge, CloseReturnSuccessThenBadHandleOnReuse) {
    int handle = 0;
    int rc = conf_load_string_c(VALID_YAML.c_str(),
                                static_cast<int>(VALID_YAML.size()),
                                &handle);
    ASSERT_EQ(rc, SUCCESS);
    ASSERT_GT(handle, 0);

    // Close should succeed
    rc = conf_close_c(handle);
    EXPECT_EQ(rc, SUCCESS);

    // Second close returns Bad_Handle
    rc = conf_close_c(handle);
    EXPECT_EQ(rc, BAD_HANDLE);

    // Getter on closed handle returns Bad_Handle
    const std::string key = "model.layers";
    int out = 0;
    rc = conf_get_int_c(handle, key.c_str(), static_cast<int>(key.size()), &out);
    EXPECT_EQ(rc, BAD_HANDLE);
}
