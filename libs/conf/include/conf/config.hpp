#ifndef CONF_CONFIG_HPP
#define CONF_CONFIG_HPP

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace conf {

class Value;  // forward declaration — defined in conf/value.hpp

/// RAII owner of a parsed YAML configuration document.
/// Movable, non-copyable. Each instance owns an independent node tree.
class Config {
public:
    // ── Factory constructors (named, to disambiguate file vs. string) ──────────
    [[nodiscard]] static Config from_file(const std::string& path);
    [[nodiscard]] static Config from_string(const std::string& yaml_text);

    // ── RAII: move-only ───────────────────────────────────────────────────────
    Config(Config&&) noexcept;
    Config& operator=(Config&&) noexcept;
    Config(const Config&)            = delete;
    Config& operator=(const Config&) = delete;
    ~Config();

    // ── Existence / structure introspection (non-throwing) ────────────────────
    [[nodiscard]] bool        has(std::string_view dotted_path) const noexcept;
    [[nodiscard]] bool        is_map(std::string_view dotted_path) const noexcept;
    [[nodiscard]] bool        is_sequence(std::string_view dotted_path) const noexcept;
    [[nodiscard]] std::size_t size(std::string_view dotted_path) const noexcept;

    // ── Typed scalar accessors (throwing flavor) ─────────────────────────────
    // Throw Conf_Error{Key_Not_Found} or Conf_Error{Type_Mismatch} on failure.
    [[nodiscard]] int         get_int(std::string_view dotted_path) const;
    [[nodiscard]] double      get_double(std::string_view dotted_path) const;
    [[nodiscard]] bool        get_bool(std::string_view dotted_path) const;
    [[nodiscard]] std::string get_string(std::string_view dotted_path) const;

    // ── Typed scalar accessors (non-throwing flavor) ─────────────────────────
    // Return std::nullopt on missing key OR type mismatch.
    [[nodiscard]] std::optional<int>         try_int(std::string_view path) const noexcept;
    [[nodiscard]] std::optional<double>      try_double(std::string_view path) const noexcept;
    [[nodiscard]] std::optional<bool>        try_bool(std::string_view path) const noexcept;
    [[nodiscard]] std::optional<std::string> try_string(std::string_view path) const noexcept;

    // ── Defaulted accessor (never throws; returns fallback on any failure) ───
    template <typename T>
    [[nodiscard]] T get_or(std::string_view dotted_path, T fallback) const noexcept;

    // ── Generic node access (returns a lightweight Value view) ───────────────
    [[nodiscard]] Value at(std::string_view dotted_path) const;

    // ── List access ──────────────────────────────────────────────────────────
    [[nodiscard]] std::vector<int>         get_int_list(std::string_view path) const;
    [[nodiscard]] std::vector<double>      get_double_list(std::string_view path) const;
    [[nodiscard]] std::vector<std::string> get_string_list(std::string_view path) const;

private:
    Config() noexcept;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ── Template definition (must be visible in the header) ──────────────────────

template <typename T>
T Config::get_or(std::string_view dotted_path, T fallback) const noexcept {
    try {
        if constexpr (std::is_same_v<T, int>) {
            return get_int(dotted_path);
        } else if constexpr (std::is_same_v<T, double>) {
            return get_double(dotted_path);
        } else if constexpr (std::is_same_v<T, bool>) {
            return get_bool(dotted_path);
        } else if constexpr (std::is_same_v<T, std::string>) {
            return get_string(dotted_path);
        } else {
            return fallback;
        }
    } catch (...) {
        return fallback;
    }
}

} // namespace conf

#endif // CONF_CONFIG_HPP
