#ifndef CONF_VALUE_HPP
#define CONF_VALUE_HPP

#include <cstddef>
#include <optional>
#include <string>

namespace conf {

/// Describes the kind of a resolved YAML node.
enum class Node_Kind { Undefined, Null, Scalar, Sequence, Map };

// Forward declaration — Value is constructed only by Config.
class Config;

/// Non-owning typed view over a single resolved YAML node.
/// Validity is tied to the parent Config's lifetime (does not extend it).
class Value {
   public:
    /// Reports the node's kind as exactly one of the Node_Kind enumerators.
    [[nodiscard]] Node_Kind kind() const noexcept;

    /// Returns true if the viewed node is defined (kind != Undefined).
    [[nodiscard]] bool is_defined() const noexcept;

    /// Returns the child count for map/sequence nodes; 0 for scalar/null/undefined.
    [[nodiscard]] std::size_t size() const noexcept;

    // ── Throwing conversions ──
    // Raise Conf_Error{Type_Mismatch} when the node is not a scalar or
    // the scalar text cannot be parsed as the requested type.

    [[nodiscard]] int as_int() const;
    [[nodiscard]] double as_double() const;
    [[nodiscard]] bool as_bool() const;
    [[nodiscard]] std::string as_string() const;

    // ── Non-throwing conversions ──
    // Return an engaged optional on success; empty optional on any failure.

    [[nodiscard]] std::optional<int> try_int() const noexcept;
    [[nodiscard]] std::optional<double> try_double() const noexcept;
    [[nodiscard]] std::optional<bool> try_bool() const noexcept;
    [[nodiscard]] std::optional<std::string> try_string() const noexcept;

   private:
    friend class Config;
    explicit Value(const void *node_ptr) noexcept;

    /// Type-erased pointer into the parent Config's node tree.
    /// Points to a yaml-cpp node internally; the type is erased here so that
    /// no yaml-cpp header is required by consumers of this public header.
    const void *node_;
};

}  // namespace conf

#endif  // CONF_VALUE_HPP
