#ifndef CONF_VALUE_HPP
#define CONF_VALUE_HPP

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

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

    // ── Child access (sequence / map navigation) ──
    // These allow iteration over structured YAML without building dotted-path
    // strings. They return lightweight Value views into the same tree.

    /// Access a sequence element by index. Returns an undefined Value if out of range.
    [[nodiscard]] Value operator[](std::size_t index) const noexcept;

    /// Access a map child by key. Returns an undefined Value if not found.
    [[nodiscard]] Value operator[](const std::string &key) const noexcept;

    /// Get the list of keys for a map node. Empty vector for non-map nodes.
    [[nodiscard]] std::vector<std::string> keys() const;

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

    // ── Defaulted scalar access ──
    // Returns fallback on any failure (missing, wrong type, undefined node).

    [[nodiscard]] std::string string_or(const std::string &fallback) const noexcept;
    [[nodiscard]] int int_or(int fallback) const noexcept;
    [[nodiscard]] double double_or(double fallback) const noexcept;
    [[nodiscard]] bool bool_or(bool fallback) const noexcept;

   private:
    friend class Config;
    explicit Value(std::shared_ptr<void> node_ptr) noexcept;

    /// Type-erased shared pointer into the parent Config's node tree.
    /// Points to a yaml-cpp node internally; the type is erased here so that
    /// no yaml-cpp header is required by consumers of this public header.
    std::shared_ptr<void> node_;
};

}  // namespace conf

#endif  // CONF_VALUE_HPP
