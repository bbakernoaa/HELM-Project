#ifndef CONF_CONFIG_HPP
#define CONF_CONFIG_HPP

/// @file config.hpp
/// @brief Public RAII owner of a parsed YAML configuration document.
///
/// This header declares conf::Config, the entry point for all configuration
/// queries. A Config owns an independent, parsed node tree and exposes typed,
/// dotted-path accessors over it. Ownership is RAII-managed through a pimpl
/// (std::unique_ptr<Impl>): destruction frees the entire backend node tree and
/// there is no raw new/delete in the conf namespace.
///
/// This is a public CONF header. It includes only standard-library headers and
/// CONF's own public headers; it pulls in no yaml-cpp type and no other HELM
/// header, upholding the Tier-1 No-Circular-Dependency law and keeping the YAML
/// backend an invisible private implementation detail.

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "conf/error.hpp"

namespace conf {

class Value;  // forward declaration; defined in conf/value.hpp

/// @brief RAII owner of a parsed YAML configuration document.
///
/// Movable, non-copyable. Each instance owns an independent node tree captured
/// at construction time; destroying or moving one Config has no effect on any
/// other. All yaml-cpp types are hidden behind the pimpl (Impl), so this public
/// interface exposes only conf:: and standard-library types.
class Config {
public:
    // ── Factory constructors (named, to disambiguate file vs. string) ──

    /// @brief Parse a YAML document from a file path.
    /// @param path Filesystem path to a readable file containing valid YAML 1.2.
    /// @return A Config owning the parsed node tree.
    /// @throws Conf_Error{File_Not_Found} if the path cannot be opened.
    /// @throws Conf_Error{Parse_Error} if the file is not valid YAML.
    [[nodiscard]] static Config from_file(const std::string& path);

    /// @brief Parse a YAML document from an in-memory string.
    /// @param yaml_text Text containing valid YAML 1.2.
    /// @return A Config owning the parsed node tree.
    /// @throws Conf_Error{Parse_Error} if the text is not valid YAML.
    [[nodiscard]] static Config from_string(const std::string& yaml_text);

    // ── RAII: move-only ──

    Config(Config&&) noexcept;
    Config& operator=(Config&&) noexcept;
    Config(const Config&)            = delete;
    Config& operator=(const Config&) = delete;
    ~Config();  ///< Frees the backend node tree.

    // ── Existence / structure introspection (non-throwing) ──

    /// @brief Report whether a dotted path resolves to a defined node.
    /// @return true if the path resolves; false on a missing or malformed path.
    [[nodiscard]] bool has(std::string_view dotted_path) const noexcept;

    /// @brief Report whether a dotted path resolves to a map node.
    /// @return true only for a resolved map node; false otherwise.
    [[nodiscard]] bool is_map(std::string_view dotted_path) const noexcept;

    /// @brief Report whether a dotted path resolves to a sequence node.
    /// @return true only for a resolved sequence node; false otherwise.
    [[nodiscard]] bool is_sequence(std::string_view dotted_path) const noexcept;

    /// @brief Number of children for a map/sequence node.
    /// @return Child count for a map or sequence node; 0 if scalar, null,
    ///         missing, or the path is malformed.
    [[nodiscard]] std::size_t size(std::string_view dotted_path) const noexcept;

    // ── Typed scalar accessors (throwing flavor) ──

    /// @throws Conf_Error{Invalid_Arg} on a malformed dotted path.
    /// @throws Conf_Error{Key_Not_Found} if the path does not resolve.
    /// @throws Conf_Error{Type_Mismatch} if the node cannot convert to the type.
    [[nodiscard]] int         get_int(std::string_view dotted_path) const;
    [[nodiscard]] double      get_double(std::string_view dotted_path) const;
    [[nodiscard]] bool        get_bool(std::string_view dotted_path) const;
    [[nodiscard]] std::string get_string(std::string_view dotted_path) const;

    // ── Typed scalar accessors (non-throwing flavor) ──
    // Return std::nullopt on a missing key, type mismatch, or malformed path.

    [[nodiscard]] std::optional<int>         try_int(std::string_view path) const noexcept;
    [[nodiscard]] std::optional<double>      try_double(std::string_view path) const noexcept;
    [[nodiscard]] std::optional<bool>        try_bool(std::string_view path) const noexcept;
    [[nodiscard]] std::optional<std::string> try_string(std::string_view path) const noexcept;

    // ── Defaulted accessor (never throws; returns fallback on any failure) ──

    /// @brief Return the resolved/converted value, or the fallback on any failure.
    /// @tparam T One of int, double, bool, or std::string.
    template <typename T>
    [[nodiscard]] T get_or(std::string_view dotted_path, T fallback) const noexcept;

    // ── Generic node access (returns a lightweight Value view) ──

    /// @brief Resolve a dotted path to a lightweight, non-owning Value view.
    /// @throws Conf_Error{Invalid_Arg} on a malformed dotted path.
    /// @throws Conf_Error{Key_Not_Found} if the path does not resolve.
    [[nodiscard]] Value at(std::string_view dotted_path) const;

    // ── List access ──

    /// @throws Conf_Error{Invalid_Arg} on a malformed dotted path.
    /// @throws Conf_Error{Key_Not_Found} if the path does not resolve.
    /// @throws Conf_Error{Type_Mismatch} if the node is not a sequence or any
    ///         element cannot convert to the requested element type.
    [[nodiscard]] std::vector<int>         get_int_list(std::string_view path) const;
    [[nodiscard]] std::vector<double>      get_double_list(std::string_view path) const;
    [[nodiscard]] std::vector<std::string> get_string_list(std::string_view path) const;

private:
    Config() noexcept;            ///< Used by the factory methods.
    struct Impl;                  ///< pimpl: hides detail::Yaml_Tree + yaml-cpp.
    std::unique_ptr<Impl> impl_;  ///< RAII ownership of the parsed tree.
};

} // namespace conf

#endif // CONF_CONFIG_HPP
