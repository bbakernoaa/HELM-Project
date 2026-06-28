#ifndef CONF_DETAIL_YAML_TREE_HPP
#define CONF_DETAIL_YAML_TREE_HPP

/// @file yaml_tree.hpp
/// @brief Private YAML backend wrapper — the ONLY header that includes yaml-cpp.
///
/// This header lives in src/detail/ and is NEVER placed under include/ or
/// installed. It is compiled into the conf library but kept completely private.
/// Downstream consumers of HELM::CONF never see yaml-cpp types.

#include <yaml-cpp/yaml.h>

#include <string>
#include <string_view>

#include "conf/error.hpp"

namespace conf {

// Forward-declare Node_Kind so Yaml_Tree can report it without depending on
// the public value.hpp header (which itself must not include yaml-cpp).
enum class Node_Kind;

namespace detail {

/// @brief Private wrapper around a yaml-cpp node tree.
///
/// Yaml_Tree is the sole point of contact with the yaml-cpp library.
/// It owns the parsed YAML::Node root and provides:
///   - Factory construction from file path or in-memory string.
///   - Dotted-path resolution (throwing Conf_Error on failure).
///   - Typed conversion of resolved nodes.
///   - Node kind introspection.
///
/// This class is move-only. A moved-from instance is left in an empty state
/// where `empty()` returns true.
class Yaml_Tree {
   public:
    // ── Factory constructors ─────────────────────────────────────────────────

    /// @brief Parse a YAML file from disk.
    /// @param path Filesystem path to the YAML file.
    /// @throws Conf_Error with File_Not_Found if the path cannot be opened.
    /// @throws Conf_Error with Parse_Error if the file contains malformed YAML.
    [[nodiscard]] static Yaml_Tree from_file(const std::string &path);

    /// @brief Parse YAML from an in-memory string.
    /// @param text The YAML text to parse.
    /// @throws Conf_Error with Parse_Error if the text contains malformed YAML.
    [[nodiscard]] static Yaml_Tree from_string(const std::string &text);

    // ── Path resolution ──────────────────────────────────────────────────────

    /// @brief Resolve a dotted path against the root node tree.
    ///
    /// Walks the parsed tree one segment at a time (splitting on '.').
    /// - Raises Invalid_Arg for empty paths or empty segments (leading/trailing/
    ///   consecutive dots).
    /// - Raises Key_Not_Found for absent map keys, out-of-range or non-integer
    ///   sequence indices, or descending past a scalar/null node.
    /// - Never returns an undefined node; the returned node is always defined.
    ///
    /// @param dotted_path A non-empty, '.'-delimited key string.
    /// @return The resolved YAML::Node (always defined on success).
    /// @throws Conf_Error with Invalid_Arg or Key_Not_Found.
    [[nodiscard]] YAML::Node resolve(std::string_view dotted_path) const;

    // ── Typed conversion ─────────────────────────────────────────────────────

    /// @brief Convert a resolved node to type T.
    ///
    /// Delegates to yaml-cpp's as<T>() after verifying the node is a scalar
    /// (any node is valid for T = std::string). Translates conversion failures
    /// into Conf_Error with Type_Mismatch.
    ///
    /// Explicit instantiations are provided in yaml_tree.cpp for:
    ///   int, double, bool, std::string
    ///
    /// @tparam T Target type (int, double, bool, or std::string).
    /// @param node A defined, resolved YAML::Node.
    /// @return The converted value of type T.
    /// @throws Conf_Error with Type_Mismatch if conversion is not possible.
    template <typename T>
    [[nodiscard]] T convert(const YAML::Node &node) const;

    // ── Node introspection ───────────────────────────────────────────────────

    /// @brief Determine the kind of a YAML node.
    ///
    /// Maps yaml-cpp's internal node type to the public conf::Node_Kind enum.
    /// @param node The YAML::Node to inspect.
    /// @return The corresponding Node_Kind value.
    [[nodiscard]] static Node_Kind node_kind(const YAML::Node &node) noexcept;

    // ── State queries ────────────────────────────────────────────────────────

    /// @brief Access the root node of the parsed tree.
    /// @return A const reference to the root YAML::Node.
    [[nodiscard]] const YAML::Node &root() const noexcept {
        return root_;
    }

    /// @brief Check whether this tree is in an empty (moved-from) state.
    /// @return true if the tree has no valid root (moved-from or default).
    [[nodiscard]] bool empty() const noexcept {
        return !root_.IsDefined();
    }

    // ── Special members ──────────────────────────────────────────────────────

    Yaml_Tree(Yaml_Tree &&other) noexcept;
    Yaml_Tree &operator=(Yaml_Tree &&other) noexcept;

    Yaml_Tree(const Yaml_Tree &) = delete;
    Yaml_Tree &operator=(const Yaml_Tree &) = delete;

    ~Yaml_Tree() = default;

   private:
    /// @brief Construct from a pre-parsed YAML root node.
    explicit Yaml_Tree(YAML::Node root) noexcept;

    YAML::Node root_;  ///< Owns the parsed YAML node tree.
};

}  // namespace detail
}  // namespace conf

#endif  // CONF_DETAIL_YAML_TREE_HPP
