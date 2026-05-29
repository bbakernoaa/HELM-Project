/// @file conf_c_interop.cpp
/// @brief Extern "C" interop bridge for Fortran iso_c_binding consumption.
///
/// This file implements the C-API bridge that connects the Fortran conf_mod
/// module to the C++ CONF core library. It mirrors HALO's halo_c_interop.cpp
/// pattern exactly:
///   - Every function returns an int whose value is a conf::Error_Code
///     (0 == Success, non-zero == a specific failure).
///   - Input strings are passed as a (const char*, int len) pair because
///     Fortran character data is not null-terminated.
///   - Output values are returned through caller-supplied pointers.
///   - Every C++ exception is caught at the boundary via CONF_C_TRY so that no
///     exception ever propagates into Fortran (which would be undefined
///     behavior).
///   - conf::Config objects are referenced from Fortran via opaque int handle
///     tokens managed by the thread-safe Handle_Registry singleton.
///
/// String values use the length-first ("two-call") marshalling protocol: the
/// caller first queries the exact byte count via conf_get_string_len_c, then
/// allocates its own buffer and receives a copy via conf_get_string_c. CONF
/// never returns ownership of heap memory across the boundary and retains no
/// allocation that outlives a single bridge call.
///
/// Requirements: 12.3, 17.1-17.8, 19.1-19.5, 20.1-20.7, 21.1-21.4, 30.1-30.3

#include "handle_registry.hpp"

#include <conf/config.hpp>
#include <conf/error.hpp>

#include <cstddef>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>

/// @brief Wrap a C-bridge function body, translating any C++ exception into an
///        int Error_Code.
///
/// Guarantees that no exception ever crosses into Fortran. The catch order is
/// significant and mirrors the design:
///   1. conf::Conf_Error      -> the precise Error_Code it carries.
///   2. std::invalid_argument -> Invalid_Arg.
///   3. std::bad_alloc        -> Unknown.
///   4. std::exception        -> Unknown.
///   5. ... (anything else)   -> Unknown.
///
/// Catching conf::Conf_Error first lets the exact failure mode (Bad_Handle,
/// Key_Not_Found, Type_Mismatch, Buffer_Too_Small, ...) reach the Fortran
/// caller unchanged. This mirrors HALO_C_TRY with the Conf_Error enhancement.
#define CONF_C_TRY(body)                                              \
    try {                                                             \
        body;                                                         \
        return static_cast<int>(conf::Error_Code::Success);           \
    } catch (const conf::Conf_Error& e) {                             \
        return static_cast<int>(e.code());                            \
    } catch (const std::invalid_argument&) {                          \
        return static_cast<int>(conf::Error_Code::Invalid_Arg);       \
    } catch (const std::bad_alloc&) {                                 \
        return static_cast<int>(conf::Error_Code::Unknown);           \
    } catch (const std::exception&) {                                 \
        return static_cast<int>(conf::Error_Code::Unknown);           \
    } catch (...) {                                                   \
        return static_cast<int>(conf::Error_Code::Unknown);           \
    }

namespace {

/// @brief Look up a registered Config* for a handle, or throw Bad_Handle.
///
/// Returns the associated conf::Config pointer for a currently registered
/// token. For token 0 (the invalid sentinel), a never-issued token, or an
/// already-released token, the registry returns nullptr and this helper throws
/// Conf_Error{Bad_Handle} WITHOUT dereferencing any Config object.
[[nodiscard]] conf::Config* config_for(int handle) {
    auto* cfg = static_cast<conf::Config*>(
        conf::fortran::Handle_Registry::instance().lookup(handle));
    if (!cfg) {
        throw conf::Conf_Error(conf::Error_Code::Bad_Handle,
                               "invalid or released config handle");
    }
    return cfg;
}

} // anonymous namespace

extern "C" {

// ── Lifecycle ────────────────────────────────────────────────────────────────

/// @brief Parse a YAML file and register a Config, returning its handle.
///
/// On success, a heap-allocated conf::Config is registered and its opaque token
/// is written through handle_out. A failed load (File_Not_Found, Parse_Error,
/// Invalid_Arg) registers nothing, allocates nothing that outlives the call,
/// and leaves handle_out unchanged.
///
/// @param path       Pointer to path bytes (not null-terminated).
/// @param path_len   Number of bytes in path; must be >= 1.
/// @param handle_out Output: opaque handle token for the loaded Config.
/// @return Success, or Invalid_Arg / File_Not_Found / Parse_Error / Unknown.
int conf_load_c(const char* path, int path_len, int* handle_out) {
    CONF_C_TRY(
        if (!path || !handle_out || path_len < 1)
            throw conf::Conf_Error(conf::Error_Code::Invalid_Arg,
                                   "null/empty path argument");
        std::string p(path, static_cast<std::size_t>(path_len));
        // from_file is evaluated before the allocation, so a failed parse never
        // leaks a Config and never registers a handle.
        auto* cfg = new conf::Config(conf::Config::from_file(p));
        *handle_out = conf::fortran::Handle_Registry::instance()
                          .register_handle(static_cast<void*>(cfg));
    )
}

/// @brief Parse YAML from an in-memory string and register a Config.
///
/// @param yaml_text  Pointer to YAML bytes (not null-terminated).
/// @param text_len   Number of bytes in yaml_text; must be >= 1.
/// @param handle_out Output: opaque handle token for the loaded Config.
/// @return Success, or Invalid_Arg / Parse_Error / Unknown.
int conf_load_string_c(const char* yaml_text, int text_len, int* handle_out) {
    CONF_C_TRY(
        if (!yaml_text || !handle_out || text_len < 1)
            throw conf::Conf_Error(conf::Error_Code::Invalid_Arg,
                                   "null/empty yaml_text argument");
        std::string text(yaml_text, static_cast<std::size_t>(text_len));
        auto* cfg = new conf::Config(conf::Config::from_string(text));
        *handle_out = conf::fortran::Handle_Registry::instance()
                          .register_handle(static_cast<void*>(cfg));
    )
}

/// @brief Destroy a Config and permanently invalidate its handle.
///
/// Releases the token from the registry and deletes the underlying Config,
/// whose RAII destructor frees the parsed node tree. A token that is not
/// currently registered (0, never issued, or already released) yields
/// Bad_Handle and leaves the registry unchanged.
///
/// @param handle Opaque handle of the Config to destroy.
/// @return Success, or Bad_Handle.
int conf_close_c(int handle) {
    CONF_C_TRY(
        auto* ptr = conf::fortran::Handle_Registry::instance().release(handle);
        if (!ptr)
            throw conf::Conf_Error(conf::Error_Code::Bad_Handle,
                                   "invalid or released config handle");
        delete static_cast<conf::Config*>(ptr);  // RAII frees the node tree
    )
}

// ── Existence / structure ────────────────────────────────────────────────────

/// @brief Set exists_out to 1 if the dotted-path key resolves, else 0.
///
/// A missing key is NOT an error here: the function reports 0 and returns
/// Success. Errors are limited to Invalid_Arg and Bad_Handle.
///
/// @param handle     Opaque handle of the Config.
/// @param key        Pointer to key bytes (not null-terminated).
/// @param key_len    Number of bytes in key; must be >= 1.
/// @param exists_out Output: 1 if the key resolves, 0 otherwise.
/// @return Success, or Invalid_Arg / Bad_Handle.
int conf_has_key_c(int handle, const char* key, int key_len, int* exists_out) {
    CONF_C_TRY(
        if (!key || !exists_out || key_len < 1)
            throw conf::Conf_Error(conf::Error_Code::Invalid_Arg,
                                   "null/empty key argument");
        const conf::Config* cfg = config_for(handle);
        std::string_view k(key, static_cast<std::size_t>(key_len));
        *exists_out = cfg->has(k) ? 1 : 0;
    )
}

/// @brief Set size_out to the child count of a map/sequence node.
///
/// A key that resolves to a scalar or null node, or that does not resolve at
/// all, yields a size of 0 and returns Success (a missing key is not an error
/// for this introspection call). Errors are limited to Invalid_Arg and
/// Bad_Handle.
///
/// @param handle   Opaque handle of the Config.
/// @param key      Pointer to key bytes (not null-terminated).
/// @param key_len  Number of bytes in key; must be >= 1.
/// @param size_out Output: child count, or 0 for scalar/null/missing.
/// @return Success, or Invalid_Arg / Bad_Handle.
int conf_size_c(int handle, const char* key, int key_len, int* size_out) {
    CONF_C_TRY(
        if (!key || !size_out || key_len < 1)
            throw conf::Conf_Error(conf::Error_Code::Invalid_Arg,
                                   "null/empty key argument");
        const conf::Config* cfg = config_for(handle);
        std::string_view k(key, static_cast<std::size_t>(key_len));
        *size_out = static_cast<int>(cfg->size(k));
    )
}

// ── Scalar getters (numeric / bool) ──────────────────────────────────────────

/// @brief Read an integer value at a dotted-path key.
///
/// @param handle  Opaque handle of the Config.
/// @param key     Pointer to key bytes (not null-terminated).
/// @param key_len Number of bytes in key; must be >= 1.
/// @param out     Output: the integer value (written only on Success).
/// @return Success, or Invalid_Arg / Bad_Handle / Key_Not_Found / Type_Mismatch.
int conf_get_int_c(int handle, const char* key, int key_len, int* out) {
    CONF_C_TRY(
        if (!key || !out || key_len < 1)
            throw conf::Conf_Error(conf::Error_Code::Invalid_Arg,
                                   "null/empty key argument");
        const conf::Config* cfg = config_for(handle);
        std::string_view k(key, static_cast<std::size_t>(key_len));
        // get_int throws Key_Not_Found / Type_Mismatch; on throw, *out is
        // left unchanged because the assignment never executes.
        *out = cfg->get_int(k);
    )
}

/// @brief Read a double value at a dotted-path key.
///
/// @param handle  Opaque handle of the Config.
/// @param key     Pointer to key bytes (not null-terminated).
/// @param key_len Number of bytes in key; must be >= 1.
/// @param out     Output: the double value (written only on Success).
/// @return Success, or Invalid_Arg / Bad_Handle / Key_Not_Found / Type_Mismatch.
int conf_get_double_c(int handle, const char* key, int key_len, double* out) {
    CONF_C_TRY(
        if (!key || !out || key_len < 1)
            throw conf::Conf_Error(conf::Error_Code::Invalid_Arg,
                                   "null/empty key argument");
        const conf::Config* cfg = config_for(handle);
        std::string_view k(key, static_cast<std::size_t>(key_len));
        *out = cfg->get_double(k);
    )
}

/// @brief Read a boolean value at a dotted-path key, writing 1 (true) or 0 (false).
///
/// @param handle  Opaque handle of the Config.
/// @param key     Pointer to key bytes (not null-terminated).
/// @param key_len Number of bytes in key; must be >= 1.
/// @param out     Output: 1 for true, 0 for false (written only on Success).
/// @return Success, or Invalid_Arg / Bad_Handle / Key_Not_Found / Type_Mismatch.
int conf_get_bool_c(int handle, const char* key, int key_len, int* out) {
    CONF_C_TRY(
        if (!key || !out || key_len < 1)
            throw conf::Conf_Error(conf::Error_Code::Invalid_Arg,
                                   "null/empty key argument");
        const conf::Config* cfg = config_for(handle);
        std::string_view k(key, static_cast<std::size_t>(key_len));
        *out = cfg->get_bool(k) ? 1 : 0;
    )
}

// ── String getter (two-call, length-first protocol) ──────────────────────────

/// @brief STEP 1: report the exact byte count the string value would occupy.
///
/// Reports through str_len_out the number of bytes (excluding any terminator)
/// the value at key occupies. This count equals the byte count
/// conf_get_string_c enforces for the same handle, key, and unchanged document.
///
/// @param handle      Opaque handle of the Config.
/// @param key         Pointer to key bytes (not null-terminated).
/// @param key_len     Number of bytes in key; must be >= 1.
/// @param str_len_out Output: exact byte count (written only on Success).
/// @return Success, or Invalid_Arg / Bad_Handle / Key_Not_Found / Type_Mismatch.
int conf_get_string_len_c(int handle, const char* key, int key_len,
                          int* str_len_out) {
    CONF_C_TRY(
        if (!key || !str_len_out || key_len < 1)
            throw conf::Conf_Error(conf::Error_Code::Invalid_Arg,
                                   "null/empty key argument");
        const conf::Config* cfg = config_for(handle);
        std::string_view k(key, static_cast<std::size_t>(key_len));
        std::string v = cfg->get_string(k);  // throws if missing / not convertible
        *str_len_out = static_cast<int>(v.size());
    )
}

/// @brief STEP 2: copy the string value into the caller-allocated buffer.
///
/// Copies the value's bytes into the caller-owned buffer. The string is NOT
/// null-terminated: written_out is the authoritative logical length. CONF
/// allocates nothing that outlives this call and retains no static buffer.
///
/// Behavior:
///   - If the required byte count exceeds buf_cap, nothing is written, the
///     buffer and written_out are left untouched, and Buffer_Too_Small is
///     returned.
///   - The 0-length value case (buf_cap >= 0) writes nothing, reports 0, and
///     returns Success — even when buf is null and buf_cap is 0.
///   - buf is a required (non-null) pointer only when buf_cap > 0; a buffer
///     capacity of 0 by itself is not Invalid_Arg.
///
/// @param handle      Opaque handle of the Config.
/// @param key         Pointer to key bytes (not null-terminated).
/// @param key_len     Number of bytes in key; must be >= 1.
/// @param buf         Caller-owned destination buffer (required when buf_cap > 0).
/// @param buf_cap     Capacity of buf in bytes; must be >= 0.
/// @param written_out Output: bytes written (set only on a Success copy).
/// @return Success, or Invalid_Arg / Bad_Handle / Key_Not_Found / Type_Mismatch
///         / Buffer_Too_Small.
int conf_get_string_c(int handle, const char* key, int key_len,
                      char* buf, int buf_cap, int* written_out) {
    CONF_C_TRY(
        // key and written_out are always required. buf is required only when
        // buf_cap > 0. A buf_cap of 0 alone is NOT Invalid_Arg; a negative
        // capacity is malformed.
        if (!key || !written_out || key_len < 1 || buf_cap < 0 ||
            (buf_cap > 0 && !buf))
            throw conf::Conf_Error(conf::Error_Code::Invalid_Arg,
                                   "null/empty string arguments");
        const conf::Config* cfg = config_for(handle);
        std::string_view k(key, static_cast<std::size_t>(key_len));
        std::string v = cfg->get_string(k);  // throws if missing / not convertible

        const int required = static_cast<int>(v.size());
        if (required > buf_cap)
            // Write nothing; leave buf and *written_out unchanged.
            throw conf::Conf_Error(conf::Error_Code::Buffer_Too_Small,
                                   "caller buffer too small for string value");

        if (required > 0)
            // buf is guaranteed non-null here: buf_cap >= required > 0.
            std::memcpy(buf, v.data(), static_cast<std::size_t>(required));

        *written_out = required;  // not null-terminated; this is the logical length
    )
}

} // extern "C"
