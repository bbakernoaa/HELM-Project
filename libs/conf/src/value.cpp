// CONF — src/value.cpp
// Implementation of conf::Value (non-owning typed view of a resolved node).

#include "conf/value.hpp"

#include <yaml-cpp/yaml.h>

#include <memory>
#include <vector>

#include "conf/error.hpp"
#include "detail/yaml_tree.hpp"

namespace conf {

// ── Construction ─────────────────────────────────────────────────────────────

Value::Value(std::shared_ptr<void> node_ptr) noexcept : node_(std::move(node_ptr)) {}

Value Value::from_raw(const void* node_ptr) noexcept {
    // Create a shared_ptr with a no-op deleter (caller owns the pointed-to node)
    return Value(std::shared_ptr<void>(const_cast<void*>(node_ptr), [](void*) {}));
}

// ── Introspection ────────────────────────────────────────────────────────────

Node_Kind Value::kind() const noexcept {
    if (!node_) return Node_Kind::Undefined;
    const auto *n = static_cast<const YAML::Node *>(node_.get());
    return detail::Yaml_Tree::node_kind(*n);
}

bool Value::is_defined() const noexcept {
    return kind() != Node_Kind::Undefined;
}

Value::operator bool() const noexcept {
    return is_defined();
}

std::size_t Value::size() const noexcept {
    if (!node_) return 0;
    const auto *n = static_cast<const YAML::Node *>(node_.get());
    if (n->IsMap() || n->IsSequence()) return n->size();
    return 0;
}

// ── Child access ─────────────────────────────────────────────────────────────

Value Value::operator[](std::size_t index) const noexcept {
    if (!node_) return Value(nullptr);
    const auto *n = static_cast<const YAML::Node *>(node_.get());
    if (!n->IsSequence() || index >= n->size()) return Value(nullptr);
    return Value(std::make_shared<YAML::Node>((*n)[index]));
}

Value Value::operator[](const std::string &key) const noexcept {
    if (!node_) return Value(nullptr);
    const auto *n = static_cast<const YAML::Node *>(node_.get());
    if (!n->IsMap()) return Value(nullptr);
    YAML::Node child = (*n)[key];
    if (!child.IsDefined()) return Value(nullptr);
    return Value(std::make_shared<YAML::Node>(child));
}

std::vector<std::string> Value::keys() const {
    std::vector<std::string> result;
    if (!node_) return result;
    const auto *n = static_cast<const YAML::Node *>(node_.get());
    if (!n->IsMap()) return result;
    result.reserve(n->size());
    for (auto it = n->begin(); it != n->end(); ++it) {
        result.push_back(it->first.as<std::string>());
    }
    return result;
}

std::vector<double> Value::as_double_list() const {
    std::vector<double> result;
    if (!node_) return result;
    const auto *n = static_cast<const YAML::Node *>(node_.get());
    if (!n->IsSequence()) return result;
    result.reserve(n->size());
    for (std::size_t i = 0; i < n->size(); ++i) {
        result.push_back((*n)[i].as<double>());
    }
    return result;
}

std::vector<std::string> Value::as_string_list() const {
    std::vector<std::string> result;
    if (!node_) return result;
    const auto *n = static_cast<const YAML::Node *>(node_.get());
    if (!n->IsSequence()) return result;
    result.reserve(n->size());
    for (std::size_t i = 0; i < n->size(); ++i) {
        result.push_back((*n)[i].as<std::string>());
    }
    return result;
}

// ── Throwing conversions ─────────────────────────────────────────────────────

int Value::as_int() const {
    if (!node_) throw Conf_Error(Error_Code::Type_Mismatch, "undefined node");
    const auto *n = static_cast<const YAML::Node *>(node_.get());
    if (!n->IsScalar()) throw Conf_Error(Error_Code::Type_Mismatch, "node is not a scalar");
    try {
        return n->as<int>();
    } catch (const YAML::BadConversion &) {
        throw Conf_Error(Error_Code::Type_Mismatch, "cannot convert scalar to int");
    }
}

double Value::as_double() const {
    if (!node_) throw Conf_Error(Error_Code::Type_Mismatch, "undefined node");
    const auto *n = static_cast<const YAML::Node *>(node_.get());
    if (!n->IsScalar()) throw Conf_Error(Error_Code::Type_Mismatch, "node is not a scalar");
    try {
        return n->as<double>();
    } catch (const YAML::BadConversion &) {
        throw Conf_Error(Error_Code::Type_Mismatch, "cannot convert scalar to double");
    }
}

bool Value::as_bool() const {
    if (!node_) throw Conf_Error(Error_Code::Type_Mismatch, "undefined node");
    const auto *n = static_cast<const YAML::Node *>(node_.get());
    if (!n->IsScalar()) throw Conf_Error(Error_Code::Type_Mismatch, "node is not a scalar");
    try {
        return n->as<bool>();
    } catch (const YAML::BadConversion &) {
        throw Conf_Error(Error_Code::Type_Mismatch, "cannot convert scalar to bool");
    }
}

std::string Value::as_string() const {
    if (!node_) throw Conf_Error(Error_Code::Type_Mismatch, "undefined node");
    const auto *n = static_cast<const YAML::Node *>(node_.get());
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

// ── Defaulted scalar access ──────────────────────────────────────────────────

std::string Value::string_or(const std::string &fallback) const noexcept {
    auto v = try_string();
    return v.has_value() ? *v : fallback;
}

int Value::int_or(int fallback) const noexcept {
    auto v = try_int();
    return v.has_value() ? *v : fallback;
}

double Value::double_or(double fallback) const noexcept {
    auto v = try_double();
    return v.has_value() ? *v : fallback;
}

bool Value::bool_or(bool fallback) const noexcept {
    auto v = try_bool();
    return v.has_value() ? *v : fallback;
}

}  // namespace conf
