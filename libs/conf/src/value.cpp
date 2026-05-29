/// @file value.cpp
/// @brief Implementation of conf::Value, the non-owning resolved-node view.
///
/// A conf::Value is a lightweight, non-owning view over a single resolved node.
/// Its private `const void* node_` is a TYPE-ERASED pointer to a YAML::Node that
/// is owned and kept alive by conf::Config::Impl for the parent Config's
/// lifetime. This translation unit is the only place in the public Value API
/// that re-interprets that pointer back into a yaml-cpp node:
///
///     const auto* node = reinterpret_cast<const YAML::Node*>(node_);
///
/// Because value.cpp lives under src/ (never installed) it may include the
/// private backend header detail/yaml_tree.hpp, which in turn pulls in
/// <yaml-cpp/yaml.h>. No yaml-cpp type ever appears in a public CONF header, so
/// the Tier-1 No-Circular-Dependency and backend-encapsulation contracts hold.
///
/// SHARED INTEROP CONTRACT (must match src/config.cpp): Config::at constructs a
/// Value from a pointer to a YAML::Node living inside Config::Impl. Value neither
/// owns nor copies that tree — it only reads through the pointer and is valid
/// only while the parent Config is alive. Value has no public default
/// constructor, so for a Value produced by Config::at the pointer is a valid
/// `const YAML::Node*`. As defence in depth we still treat a null pointer as an
/// Undefined node so nothing ever dereferences null.

#include <conf/value.hpp>

#include <optional>
#include <string>
#include <type_traits>

#include "detail/yaml_tree.hpp"

namespace conf {

namespace {

/// @brief Human-readable name for a supported conversion target type.
/// Used only to build diagnostic messages for Type_Mismatch errors. Mirrors the
/// helper of the same purpose in detail/yaml_tree.cpp.
template <typename T>
constexpr const char* type_name() {
    if constexpr (std::is_same_v<T, int>) {
        return "int";
    } else if constexpr (std::is_same_v<T, double>) {
        return "double";
    } else if constexpr (std::is_same_v<T, bool>) {
        return "bool";
    } else if constexpr (std::is_same_v<T, std::string>) {
        return "string";
    } else {
        return "value";
    }
}

/// @brief Re-interpret the type-erased handle as a yaml-cpp node pointer.
/// @return The viewed node, or nullptr for a (defensively handled) null handle.
const YAML::Node* as_node(const void* node_ptr) noexcept {
    return reinterpret_cast<const YAML::Node*>(node_ptr);
}

/// @brief Convert the viewed node to T, mirroring detail::Yaml_Tree::convert<T>.
///
/// A non-scalar node can never convert to a numeric/bool type, so it is rejected
/// up-front with Type_Mismatch. For T == std::string the backend itself rejects
/// non-scalar nodes, which the catch translates to Type_Mismatch. A null handle
/// (only possible for a default-constructed Value, which has no public ctor) is
/// treated as a non-convertible Undefined node.
///
/// @throws conf::Conf_Error with Error_Code::Type_Mismatch on any failure.
template <typename T>
T convert_as(const void* node_ptr) {
    const YAML::Node* node = as_node(node_ptr);

    // Defensive: a null handle behaves like an Undefined node — never converts.
    if (node == nullptr) {
        throw Conf_Error(Error_Code::Type_Mismatch, "node is not a scalar");
    }

    if (!node->IsScalar() && !std::is_same_v<T, std::string>) {
        throw Conf_Error(Error_Code::Type_Mismatch, "node is not a scalar");
    }

    try {
        return node->as<T>();
    } catch (const YAML::Exception&) {
        throw Conf_Error(Error_Code::Type_Mismatch,
                         std::string("cannot convert scalar to ") + type_name<T>());
    }
}

} // namespace

// ─── Construction ────────────────────────────────────────────────────────────

Value::Value(const void* node_ptr) noexcept : node_(node_ptr) {}

// ─── Kind / structure introspection ──────────────────────────────────────────

Node_Kind Value::kind() const noexcept {
    const YAML::Node* node = as_node(node_);
    // Map the yaml-cpp node type onto exactly one Node_Kind. A null handle or an
    // undefined node is reported as Undefined so nothing dereferences null.
    if (node == nullptr || !node->IsDefined()) {
        return Node_Kind::Undefined;
    }
    if (node->IsNull()) {
        return Node_Kind::Null;
    }
    if (node->IsScalar()) {
        return Node_Kind::Scalar;
    }
    if (node->IsSequence()) {
        return Node_Kind::Sequence;
    }
    if (node->IsMap()) {
        return Node_Kind::Map;
    }
    return Node_Kind::Undefined;  // fallback — unreachable for a well-formed node
}

bool Value::is_defined() const noexcept {
    return kind() != Node_Kind::Undefined;
}

std::size_t Value::size() const noexcept {
    // Child count for a Map or Sequence; 0 for Scalar, Null, or Undefined.
    const Node_Kind k = kind();
    if (k == Node_Kind::Map || k == Node_Kind::Sequence) {
        const YAML::Node* node = as_node(node_);
        return node->size();
    }
    return 0;
}

// ─── Throwing conversions (Conf_Error{Type_Mismatch} on failure) ─────────────

int Value::as_int() const {
    return convert_as<int>(node_);
}

double Value::as_double() const {
    return convert_as<double>(node_);
}

bool Value::as_bool() const {
    return convert_as<bool>(node_);
}

std::string Value::as_string() const {
    return convert_as<std::string>(node_);
}

// ─── Non-throwing conversions (noexcept; std::nullopt on failure) ────────────

std::optional<int> Value::try_int() const noexcept {
    try {
        return as_int();
    } catch (const Conf_Error&) {
        return std::nullopt;
    }
}

std::optional<double> Value::try_double() const noexcept {
    try {
        return as_double();
    } catch (const Conf_Error&) {
        return std::nullopt;
    }
}

std::optional<bool> Value::try_bool() const noexcept {
    try {
        return as_bool();
    } catch (const Conf_Error&) {
        return std::nullopt;
    }
}

std::optional<std::string> Value::try_string() const noexcept {
    try {
        return as_string();
    } catch (const Conf_Error&) {
        return std::nullopt;
    }
}

} // namespace conf
