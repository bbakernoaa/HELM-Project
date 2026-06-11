// CONF — src/fortran/conf_c_interop.cpp
// The extern "C" bridge exposing CONF to Fortran via opaque handles and int error codes.
//
// Every function returns an int Error_Code. The CONF_C_TRY macro guarantees no
// exception ever crosses into Fortran (which would be undefined behavior).
// Mirrors HALO's proven HALO_C_TRY pattern exactly.

#include <conf/config.hpp>
#include <conf/error.hpp>
#include "fortran/handle_registry.hpp"

#include <cstring>   // std::memcpy
#include <new>       // std::bad_alloc
#include <stdexcept> // std::invalid_argument, std::exception
#include <string>

// ─── Exception-to-Error-Code Macro ──────────────────────────────────────────
// Wraps a C-bridge function body, translating any C++ exception into an int
// error code. Guarantees no exception ever crosses into Fortran.

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

// ─── Helper: validate handle and return Config pointer ──────────────────────
// Returns nullptr and sets rc to Bad_Handle if the handle is invalid.
static conf::Config* lookup_config(int handle) {
    if (handle == conf::fortran::CONF_HANDLE_INVALID) {
        return nullptr;
    }
    auto& reg = conf::fortran::Handle_Registry::instance();
    return static_cast<conf::Config*>(reg.lookup(handle));
}

// ─── extern "C" API ─────────────────────────────────────────────────────────

extern "C" {

// ── Lifecycle ────────────────────────────────────────────────────────────────

int conf_load_c(const char* path, int path_len, int* handle_out) {
    // Validate required pointers and length BEFORE CONF_C_TRY
    if (!path || !handle_out || path_len < 1) {
        return static_cast<int>(conf::Error_Code::Invalid_Arg);
    }

    CONF_C_TRY(
        std::string p(path, static_cast<std::size_t>(path_len));
        auto* cfg = new conf::Config(conf::Config::from_file(p));
        *handle_out = conf::fortran::Handle_Registry::instance()
                          .register_handle(static_cast<void*>(cfg));
    )
}

int conf_load_string_c(const char* yaml_text, int text_len, int* handle_out) {
    // Validate required pointers and length BEFORE CONF_C_TRY
    if (!yaml_text || !handle_out || text_len < 1) {
        return static_cast<int>(conf::Error_Code::Invalid_Arg);
    }

    CONF_C_TRY(
        std::string text(yaml_text, static_cast<std::size_t>(text_len));
        auto* cfg = new conf::Config(conf::Config::from_string(text));
        *handle_out = conf::fortran::Handle_Registry::instance()
                          .register_handle(static_cast<void*>(cfg));
    )
}

int conf_close_c(int handle) {
    // Validate handle BEFORE any Config access
    if (handle == conf::fortran::CONF_HANDLE_INVALID) {
        return static_cast<int>(conf::Error_Code::Bad_Handle);
    }

    auto& reg = conf::fortran::Handle_Registry::instance();
    void* ptr = reg.release(handle);
    if (!ptr) {
        return static_cast<int>(conf::Error_Code::Bad_Handle);
    }

    delete static_cast<conf::Config*>(ptr);
    return static_cast<int>(conf::Error_Code::Success);
}

// ── Existence / structure ────────────────────────────────────────────────────

int conf_has_key_c(int handle, const char* key, int key_len, int* exists_out) {
    // Validate required pointers and length
    if (!key || !exists_out || key_len < 1) {
        return static_cast<int>(conf::Error_Code::Invalid_Arg);
    }

    // Validate handle
    conf::Config* cfg = lookup_config(handle);
    if (!cfg) {
        return static_cast<int>(conf::Error_Code::Bad_Handle);
    }

    CONF_C_TRY(
        std::string k(key, static_cast<std::size_t>(key_len));
        *exists_out = cfg->has(k) ? 1 : 0;
    )
}

int conf_size_c(int handle, const char* key, int key_len, int* size_out) {
    // Validate required pointers and length
    if (!key || !size_out || key_len < 1) {
        return static_cast<int>(conf::Error_Code::Invalid_Arg);
    }

    // Validate handle
    conf::Config* cfg = lookup_config(handle);
    if (!cfg) {
        return static_cast<int>(conf::Error_Code::Bad_Handle);
    }

    CONF_C_TRY(
        std::string k(key, static_cast<std::size_t>(key_len));
        // size() returns 0 for scalar/null/missing — this is not an error
        *size_out = static_cast<int>(cfg->size(k));
    )
}

// ── Scalar getters (numeric / bool) ─────────────────────────────────────────

int conf_get_int_c(int handle, const char* key, int key_len, int* out) {
    // Validate required pointers and length
    if (!key || !out || key_len < 1) {
        return static_cast<int>(conf::Error_Code::Invalid_Arg);
    }

    // Validate handle
    conf::Config* cfg = lookup_config(handle);
    if (!cfg) {
        return static_cast<int>(conf::Error_Code::Bad_Handle);
    }

    CONF_C_TRY(
        std::string k(key, static_cast<std::size_t>(key_len));
        *out = cfg->get_int(k);
    )
}

int conf_get_double_c(int handle, const char* key, int key_len, double* out) {
    // Validate required pointers and length
    if (!key || !out || key_len < 1) {
        return static_cast<int>(conf::Error_Code::Invalid_Arg);
    }

    // Validate handle
    conf::Config* cfg = lookup_config(handle);
    if (!cfg) {
        return static_cast<int>(conf::Error_Code::Bad_Handle);
    }

    CONF_C_TRY(
        std::string k(key, static_cast<std::size_t>(key_len));
        *out = cfg->get_double(k);
    )
}

int conf_get_bool_c(int handle, const char* key, int key_len, int* out) {
    // Validate required pointers and length
    if (!key || !out || key_len < 1) {
        return static_cast<int>(conf::Error_Code::Invalid_Arg);
    }

    // Validate handle
    conf::Config* cfg = lookup_config(handle);
    if (!cfg) {
        return static_cast<int>(conf::Error_Code::Bad_Handle);
    }

    CONF_C_TRY(
        std::string k(key, static_cast<std::size_t>(key_len));
        *out = cfg->get_bool(k) ? 1 : 0;
    )
}

// ── String getter (two-call, length-first protocol) ──────────────────────────

int conf_get_string_len_c(int handle, const char* key, int key_len, int* str_len_out) {
    // Validate required pointers and length
    if (!key || !str_len_out || key_len < 1) {
        return static_cast<int>(conf::Error_Code::Invalid_Arg);
    }

    // Validate handle
    conf::Config* cfg = lookup_config(handle);
    if (!cfg) {
        return static_cast<int>(conf::Error_Code::Bad_Handle);
    }

    CONF_C_TRY(
        std::string k(key, static_cast<std::size_t>(key_len));
        std::string v = cfg->get_string(k);  // throws Key_Not_Found / Type_Mismatch
        *str_len_out = static_cast<int>(v.size());
    )
}

int conf_get_string_c(int handle, const char* key, int key_len,
                      char* buf, int buf_cap, int* written_out) {
    // Validate required pointers and length
    // Note: buf_cap of 0 alone is NOT Invalid_Arg (it may be valid for empty strings)
    // but a null buf pointer IS Invalid_Arg, and a null written_out IS Invalid_Arg
    if (!key || !written_out || key_len < 1) {
        return static_cast<int>(conf::Error_Code::Invalid_Arg);
    }
    // buf must be non-null (even if buf_cap is 0, the pointer must be valid for the
    // case where the string is empty and we write 0 bytes)
    if (!buf) {
        return static_cast<int>(conf::Error_Code::Invalid_Arg);
    }

    // Validate handle
    conf::Config* cfg = lookup_config(handle);
    if (!cfg) {
        return static_cast<int>(conf::Error_Code::Bad_Handle);
    }

    CONF_C_TRY(
        std::string k(key, static_cast<std::size_t>(key_len));
        std::string v = cfg->get_string(k);  // throws Key_Not_Found / Type_Mismatch

        int need = static_cast<int>(v.size());

        // If the string is non-empty and buf_cap is insufficient, return
        // Buffer_Too_Small and write nothing.
        if (need > buf_cap) {
            return static_cast<int>(conf::Error_Code::Buffer_Too_Small);
        }

        // Copy into the caller buffer. For empty strings (need == 0), this
        // is a no-op. Never null-terminate.
        if (need > 0) {
            std::memcpy(buf, v.data(), static_cast<std::size_t>(need));
        }
        *written_out = need;
    )
}

} // extern "C"
