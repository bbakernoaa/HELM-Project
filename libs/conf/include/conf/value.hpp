#ifndef CONF_VALUE_HPP
#define CONF_VALUE_HPP

/// @file value.hpp
/// @brief Node_Kind enumeration and the conf::Value resolved-node view.
///
/// This header declares conf::Value, a lightweight, non-owning typed view over a
/// single resolved node returned by conf::Config::at, together with the Node_Kind
/// enumeration describing the kind of a resolved node. Value lets callers
/// introspect a node's kind and convert it to a scalar type without re-resolving
/// the dotted path.
///
/// This is a public CONF header. It includes only standard-library headers plus
/// CONF's own conf/error.hpp; it pulls in no yaml-cpp type and no other HELM
/// header, upholding the Tier-1 No-Circular-Dependency law. The node handle is
/// type-erased behind a `const void*` so the yaml-cpp backend never leaks into
/// the public API.

#include <cstddef>
#include <optional>
#include <string>

#include "conf/error.hpp"

namespace conf {

/// @brief The kind of a resolved configuration node.
///
/// Every node of a parsed YAML document is classified as exactly one Node_Kind.
/// `Null` denotes a key present with an explicitly null or empty value, while
/// `Undefined` denotes a key that does not resolve; both are distinct from a
/// `Scalar` node holding an empty string.
enum class Node_Kind { Undefined, Null, Scalar, Sequence, Map };

/// @brief Non-owning typed view over a single resolved node.
///
/// Returned by conf::Config::at. A Value views a node inside the parent Config's
/// tree; it neither owns nor copies that tree and remains valid only while the
/// parent Config is alive. Its validity is tied to the parent Config's lifetime
/// and does not extend it.
class Config;

class Value {
public:
    /// @brief Report the kind of the viewed node.
    /// @return Exactly one of Undefined, Null, Scalar, Sequence, or Map.
    [[nodiscard]] Node_Kind kind() const noexcept;

    /// @brief Report whether the viewed node is a defined node.
    /// @return true unless the node is Undefined.
    [[nodiscard]] bool is_defined() const noexcept;

    /// @brief Report the child count of the viewed node.
    /// @return The number of children for a Map or Sequence node; 0 for a
    ///         Scalar, Null, or Undefined node.
    [[nodiscard]] std::size_t size() const noexcept;

    // ── Throwing conversions (raise Conf_Error{Type_Mismatch} on failure) ──
    /// @brief Convert the viewed scalar to int.
    /// @throws Conf_Error with Error_Code::Type_Mismatch if the node is not a
    ///         scalar or its text is not a valid int.
    [[nodiscard]] int         as_int() const;
    /// @brief Convert the viewed scalar to double.
    /// @throws Conf_Error with Error_Code::Type_Mismatch if the node is not a
    ///         scalar or its text is not a valid double.
    [[nodiscard]] double      as_double() const;
    /// @brief Convert the viewed scalar to bool.
    /// @throws Conf_Error with Error_Code::Type_Mismatch if the node is not a
    ///         scalar or its text is not a valid bool.
    [[nodiscard]] bool        as_bool() const;
    /// @brief Convert the viewed scalar to string.
    /// @throws Conf_Error with Error_Code::Type_Mismatch if the node is not a
    ///         scalar (every scalar is a valid representation for as_string).
    [[nodiscard]] std::string as_string() const;

    // ── Non-throwing conversions (noexcept; std::nullopt on failure) ──
    /// @brief Convert the viewed scalar to int without throwing.
    /// @return An engaged optional on success; std::nullopt on failure.
    [[nodiscard]] std::optional<int>         try_int() const noexcept;
    /// @brief Convert the viewed scalar to double without throwing.
    /// @return An engaged optional on success; std::nullopt on failure.
    [[nodiscard]] std::optional<double>      try_double() const noexcept;
    /// @brief Convert the viewed scalar to bool without throwing.
    /// @return An engaged optional on success; std::nullopt on failure.
    [[nodiscard]] std::optional<bool>        try_bool() const noexcept;
    /// @brief Convert the viewed scalar to string without throwing.
    /// @return An engaged optional on success; std::nullopt on failure.
    [[nodiscard]] std::optional<std::string> try_string() const noexcept;

private:
    friend class Config;

    /// @brief Construct a Value viewing a type-erased detail node handle.
    /// @param node_ptr A non-owning pointer into the parent Config's tree.
    explicit Value(const void* node_ptr) noexcept;

    const void* node_;  ///< Points into the parent Config's tree; non-owning.
};

} // namespace conf

#endif // CONF_VALUE_HPP
