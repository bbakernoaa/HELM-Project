// CONF — src/config.cpp
// Implementation of conf::Config (RAII owner of a parsed YAML document).

#include "conf/config.hpp"

#include <yaml-cpp/yaml.h>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "conf/error.hpp"
#include "conf/value.hpp"
#include "detail/yaml_tree.hpp"

namespace conf {

// ── Impl (pimpl) ─────────────────────────────────────────────────────────────

struct Config::Impl {
    detail::Yaml_Tree tree;

    /// Resolved nodes kept alive for Value views returned by at().
    /// Each Value holds a non-owning pointer into this container.
    mutable std::vector<std::unique_ptr<YAML::Node>> resolved_nodes;

    explicit Impl(detail::Yaml_Tree t) : tree(std::move(t)) {}
};

// ── Private default constructor (used by factories) ──────────────────────────

Config::Config() noexcept : impl_(nullptr) {}

// ── Factory constructors ─────────────────────────────────────────────────────

Config Config::from_file(const std::string &path) {
    Config cfg;
    cfg.impl_ = std::make_unique<Impl>(detail::Yaml_Tree::from_file(path));
    return cfg;
}

Config Config::from_string(const std::string &yaml_text) {
    Config cfg;
    cfg.impl_ = std::make_unique<Impl>(detail::Yaml_Tree::from_string(yaml_text));
    return cfg;
}

// ── Move operations ──────────────────────────────────────────────────────────

Config::Config(Config &&) noexcept = default;

Config &Config::operator=(Config &&other) noexcept {
    if (this != &other) {
        impl_ = std::move(other.impl_);  // releases prior tree, nullifies other
    }
    return *this;
}

Config::~Config() = default;

// ── Introspection (noexcept) ─────────────────────────────────────────────────

bool Config::has(std::string_view dotted_path) const noexcept {
    if (!impl_) return false;
    try {
        (void)impl_->tree.resolve(dotted_path);
        return true;
    } catch (...) {
        return false;
    }
}

bool Config::is_map(std::string_view dotted_path) const noexcept {
    if (!impl_) return false;
    try {
        auto node = impl_->tree.resolve(dotted_path);
        return node.IsMap();
    } catch (...) {
        return false;
    }
}

bool Config::is_sequence(std::string_view dotted_path) const noexcept {
    if (!impl_) return false;
    try {
        auto node = impl_->tree.resolve(dotted_path);
        return node.IsSequence();
    } catch (...) {
        return false;
    }
}

std::size_t Config::size(std::string_view dotted_path) const noexcept {
    if (!impl_) return 0;
    try {
        auto node = impl_->tree.resolve(dotted_path);
        if (node.IsMap() || node.IsSequence()) {
            return node.size();
        }
        return 0;
    } catch (...) {
        return 0;
    }
}

// ── Throwing scalar accessors ────────────────────────────────────────────────

int Config::get_int(std::string_view dotted_path) const {
    if (!impl_) {
        throw Conf_Error(Error_Code::Key_Not_Found, "empty config");
    }
    auto node = impl_->tree.resolve(dotted_path);
    return impl_->tree.convert<int>(node);
}

double Config::get_double(std::string_view dotted_path) const {
    if (!impl_) {
        throw Conf_Error(Error_Code::Key_Not_Found, "empty config");
    }
    auto node = impl_->tree.resolve(dotted_path);
    return impl_->tree.convert<double>(node);
}

bool Config::get_bool(std::string_view dotted_path) const {
    if (!impl_) {
        throw Conf_Error(Error_Code::Key_Not_Found, "empty config");
    }
    auto node = impl_->tree.resolve(dotted_path);
    return impl_->tree.convert<bool>(node);
}

std::string Config::get_string(std::string_view dotted_path) const {
    if (!impl_) {
        throw Conf_Error(Error_Code::Key_Not_Found, "empty config");
    }
    auto node = impl_->tree.resolve(dotted_path);
    return impl_->tree.convert<std::string>(node);
}

// ── Non-throwing scalar accessors ────────────────────────────────────────────

std::optional<int> Config::try_int(std::string_view path) const noexcept {
    try {
        return get_int(path);
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<double> Config::try_double(std::string_view path) const noexcept {
    try {
        return get_double(path);
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<bool> Config::try_bool(std::string_view path) const noexcept {
    try {
        return get_bool(path);
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<std::string> Config::try_string(std::string_view path) const noexcept {
    try {
        return get_string(path);
    } catch (...) {
        return std::nullopt;
    }
}

// ── List accessors ───────────────────────────────────────────────────────────

std::vector<int> Config::get_int_list(std::string_view path) const {
    if (!impl_) {
        throw Conf_Error(Error_Code::Key_Not_Found, "empty config");
    }
    auto node = impl_->tree.resolve(path);
    if (!node.IsSequence()) {
        throw Conf_Error(Error_Code::Type_Mismatch, "node is not a sequence");
    }
    // Build into a temporary; only return if ALL elements convert successfully.
    std::vector<int> result;
    result.reserve(node.size());
    for (std::size_t i = 0; i < node.size(); ++i) {
        result.push_back(impl_->tree.convert<int>(node[i]));
    }
    return result;
}

std::vector<double> Config::get_double_list(std::string_view path) const {
    if (!impl_) {
        throw Conf_Error(Error_Code::Key_Not_Found, "empty config");
    }
    auto node = impl_->tree.resolve(path);
    if (!node.IsSequence()) {
        throw Conf_Error(Error_Code::Type_Mismatch, "node is not a sequence");
    }
    std::vector<double> result;
    result.reserve(node.size());
    for (std::size_t i = 0; i < node.size(); ++i) {
        result.push_back(impl_->tree.convert<double>(node[i]));
    }
    return result;
}

std::vector<std::string> Config::get_string_list(std::string_view path) const {
    if (!impl_) {
        throw Conf_Error(Error_Code::Key_Not_Found, "empty config");
    }
    auto node = impl_->tree.resolve(path);
    if (!node.IsSequence()) {
        throw Conf_Error(Error_Code::Type_Mismatch, "node is not a sequence");
    }
    std::vector<std::string> result;
    result.reserve(node.size());
    for (std::size_t i = 0; i < node.size(); ++i) {
        result.push_back(impl_->tree.convert<std::string>(node[i]));
    }
    return result;
}

// ── Value view access ────────────────────────────────────────────────────────

Value Config::at(std::string_view dotted_path) const {
    if (!impl_) {
        throw Conf_Error(Error_Code::Key_Not_Found, "empty config");
    }
    auto node = impl_->tree.resolve(dotted_path);
    // Store a heap-allocated copy of the resolved YAML::Node. Since yaml-cpp
    // nodes are reference-counted handles, the copy is cheap and remains valid
    // as long as the root tree (owned by Impl) exists. The unique_ptr in
    // resolved_nodes ensures the handle is freed when Config is destroyed.
    impl_->resolved_nodes.push_back(std::make_unique<YAML::Node>(node));
    return Value(static_cast<const void *>(impl_->resolved_nodes.back().get()));
}

}  // namespace conf
