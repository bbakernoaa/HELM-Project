// CONF — src/value.cpp
// Implementation of conf::Value (non-owning typed view of a resolved node).

#include "conf/value.hpp"

#include <yaml-cpp/yaml.h>

#include "conf/error.hpp"
#include "detail/yaml_tree.hpp"

namespace conf {

// ── Construction ─────────────────────────────────────────────────────────────

Value::Value(const void *node_ptr) noexcept : node_(node_ptr) {}

// ── Introspection ────────────────────────────────────────────────────────────

Node_Kind Value::kind() const noexcept {
    if (!node_) return Node_Kind::Undefined;
    const auto *n = static_cast<const YAML::Node *>(node_);
    return detail::Yaml_Tree::node_kind(*n);
}

bool Value::is_defined() const noexcept {
    return kind() != Node_Kind::Undefined;
}

std::size_t Value::size() const noexcept {
    if (!node_) return 0;
    const auto *n = static_cast<const YAML::Node *>(node_);
    if (n->IsMap() || n->IsSequence()) return n->size();
    return 0;
}

// ── Throwing conversions ─────────────────────────────────────────────────────

int Value::as_int() const {
    if (!node_) throw Conf_Error(Error_Code::Type_Mismatch, "undefined node");
    const auto *n = static_cast<const YAML::Node *>(node_);
    if (!n->IsScalar()) throw Conf_Error(Error_Code::Type_Mismatch, "node is not a scalar");
    try {
        return n->as<int>();
    } catch (const YAML::BadConversion &) {
        throw Conf_Error(Error_Code::Type_Mismatch, "cannot convert scalar to int");
    }
}

double Value::as_double() const {
    if (!node_) throw Conf_Error(Error_Code::Type_Mismatch, "undefined node");
    const auto *n = static_cast<const YAML::Node *>(node_);
    if (!n->IsScalar()) throw Conf_Error(Error_Code::Type_Mismatch, "node is not a scalar");
    try {
        return n->as<double>();
    } catch (const YAML::BadConversion &) {
        throw Conf_Error(Error_Code::Type_Mismatch, "cannot convert scalar to double");
    }
}

bool Value::as_bool() const {
    if (!node_) throw Conf_Error(Error_Code::Type_Mismatch, "undefined node");
    const auto *n = static_cast<const YAML::Node *>(node_);
    if (!n->IsScalar()) throw Conf_Error(Error_Code::Type_Mismatch, "node is not a scalar");
    try {
        return n->as<bool>();
    } catch (const YAML::BadConversion &) {
        throw Conf_Error(Error_Code::Type_Mismatch, "cannot convert scalar to bool");
    }
}

std::string Value::as_string() const {
    if (!node_) throw Conf_Error(Error_Code::Type_Mismatch, "undefined node");
    const auto *n = static_cast<const YAML::Node *>(node_);
    if (!n->IsScalar()) throw Conf_Error(Error_Code::Type_Mismatch, "node is not a scalar");
    try {
        return n->as<std::string>();
    } catch (const YAML::BadConversion &) {
        throw Conf_Error(Error_Code::Type_Mismatch, "cannot convert node to string");
    }
}

// ── Non-throwing conversions ─────────────────────────────────────────────────

std::optional<int> Value::try_int() const noexcept {
    try {
        return as_int();
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<double> Value::try_double() const noexcept {
    try {
        return as_double();
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<bool> Value::try_bool() const noexcept {
    try {
        return as_bool();
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<std::string> Value::try_string() const noexcept {
    try {
        return as_string();
    } catch (...) {
        return std::nullopt;
    }
}

}  // namespace conf
