/// @file config.cpp
/// @brief Implementation of conf::Config, the public RAII configuration owner.
///
/// Config owns a parsed YAML document through a pimpl (std::unique_ptr<Impl>).
/// Impl holds the private detail::Yaml_Tree backend plus a stable container of
/// resolved nodes that back the non-owning conf::Value views returned by
/// Config::at. Because config.cpp lives under src/ (never installed) it may
/// include the private backend header detail/yaml_tree.hpp, which pulls in
/// <yaml-cpp/yaml.h>. No yaml-cpp type ever appears in a public CONF header, so
/// the Tier-1 No-Circular-Dependency and backend-encapsulation contracts hold.
///
/// SHARED INTEROP CONTRACT (must match src/value.cpp exactly): conf::Value's
/// private `const void* node_` points to a YAML::Node OWNED and kept alive by
/// Config::Impl for the entire lifetime of the Config; value.cpp re-interprets
/// it with `reinterpret_cast<const YAML::Node*>(node_)`. Config::at therefore
/// resolves a path to a YAML::Node, stores it in Impl::at_nodes_ (a std::deque,
/// whose element addresses are stable across growth — unlike std::vector), and
/// constructs the Value from the address of that stored node. The container is
/// `mutable` because at() is a const method. Value is constructed through its
/// private constructor, which is accessible because Value declares
/// `friend class Config`.
///
/// RAII / move semantics: Config is move-only. A moved-from Config is left in a
/// valid empty state (impl_ == nullptr): introspection returns false/0 and frees
/// nothing on destruction; throwing accessors and at() treat a null impl like an
/// unresolvable document and raise Key_Not_Found, so the non-throwing flavors
/// still yield nullopt / fallback. Move-assignment releases any previously owned
/// tree and is safe under self-move-assignment. There is no raw new/delete in
/// the conf namespace; the unique_ptr frees Impl (and thus the tree).

#include <conf/config.hpp>

#include <cstddef>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <conf/value.hpp>

#include "detail/yaml_tree.hpp"

namespace conf {

// ─── Pimpl definition ────────────────────────────────────────────────────────

/// @brief Private implementation: owns the parsed tree and the Value-backing
///        node store.
///
/// `tree` is the sole owner of the parsed document. `at_nodes_` holds one
/// YAML::Node per successful Config::at call; a std::deque is used (never a
/// std::vector) because the addresses of its elements remain stable as it grows,
/// which is required since each conf::Value stores a raw pointer into this
/// container. It is `mutable` so the const Config::at can append to it.
struct Config::Impl {
    explicit Impl(detail::Yaml_Tree parsed_tree) : tree(std::move(parsed_tree)) {}

    detail::Yaml_Tree tree;                  ///< Owns the parsed node tree.
    mutable std::deque<YAML::Node> at_nodes_; ///< Stable storage backing Values.
};

// ─── Construction / RAII ─────────────────────────────────────────────────────

Config::Config() noexcept : impl_(nullptr) {}

Config::Config(Config&& other) noexcept = default;

Config& Config::operator=(Config&& other) noexcept {
    // Guard against self-move-assignment so we never release-then-use our own
    // tree. On a genuine move, assigning the unique_ptr releases (frees) any
    // previously owned tree before taking ownership of the moved-in one, and
    // leaves the source's impl_ null (a valid empty state).
    if (this != &other) {
        impl_ = std::move(other.impl_);
    }
    return *this;
}

Config::~Config() = default;  // unique_ptr frees Impl, which frees the tree.

// ─── Factory constructors ────────────────────────────────────────────────────

Config Config::from_file(const std::string& path) {
    // The backend translates an unopenable path to File_Not_Found and a syntax
    // error to Parse_Error; either propagates out and no Config is returned.
    Config cfg;
    cfg.impl_ = std::make_unique<Impl>(detail::Yaml_Tree::from_file(path));
    return cfg;
}

Config Config::from_string(const std::string& yaml_text) {
    Config cfg;
    cfg.impl_ = std::make_unique<Impl>(detail::Yaml_Tree::from_string(yaml_text));
    return cfg;
}

// ─── Existence / structure introspection (noexcept) ──────────────────────────

bool Config::has(std::string_view dotted_path) const noexcept {
    // A moved-from / null Config resolves nothing.
    if (!impl_) {
        return false;
    }
    try {
        // resolve() returns a defined node (Map/Sequence/Scalar/Null) on success
        // and raises on a malformed or non-resolving path. We only care whether
        // it succeeds, so the resolved node is intentionally discarded.
        static_cast<void>(impl_->tree.resolve(dotted_path));
        return true;
    } catch (const Conf_Error&) {
        return false;
    }
}

bool Config::is_map(std::string_view dotted_path) const noexcept {
    if (!impl_) {
        return false;
    }
    try {
        return impl_->tree.resolve(dotted_path).IsMap();
    } catch (const Conf_Error&) {
        return false;
    }
}

bool Config::is_sequence(std::string_view dotted_path) const noexcept {
    if (!impl_) {
        return false;
    }
    try {
        return impl_->tree.resolve(dotted_path).IsSequence();
    } catch (const Conf_Error&) {
        return false;
    }
}

std::size_t Config::size(std::string_view dotted_path) const noexcept {
    if (!impl_) {
        return 0;
    }
    try {
        const YAML::Node node = impl_->tree.resolve(dotted_path);
        // Child count for a map/sequence; 0 for a scalar or null node.
        if (node.IsMap() || node.IsSequence()) {
            return node.size();
        }
        return 0;
    } catch (const Conf_Error&) {
        return 0;
    }
}

// ─── Typed scalar accessors (throwing flavor) ────────────────────────────────
// Each resolves then converts. resolve() enforces Invalid_Arg (malformed path)
// before Key_Not_Found (non-resolving path); convert() raises Type_Mismatch on a
// non-scalar node or an unparseable scalar — giving the required error
// precedence Invalid_Arg → Key_Not_Found → Type_Mismatch. A null impl is treated
// like an unresolvable document (Key_Not_Found).

int Config::get_int(std::string_view dotted_path) const {
    if (!impl_) {
        throw Conf_Error(Error_Code::Key_Not_Found, "configuration is empty");
    }
    return impl_->tree.convert<int>(impl_->tree.resolve(dotted_path));
}

double Config::get_double(std::string_view dotted_path) const {
    if (!impl_) {
        throw Conf_Error(Error_Code::Key_Not_Found, "configuration is empty");
    }
    return impl_->tree.convert<double>(impl_->tree.resolve(dotted_path));
}

bool Config::get_bool(std::string_view dotted_path) const {
    if (!impl_) {
        throw Conf_Error(Error_Code::Key_Not_Found, "configuration is empty");
    }
    return impl_->tree.convert<bool>(impl_->tree.resolve(dotted_path));
}

std::string Config::get_string(std::string_view dotted_path) const {
    if (!impl_) {
        throw Conf_Error(Error_Code::Key_Not_Found, "configuration is empty");
    }
    return impl_->tree.convert<std::string>(impl_->tree.resolve(dotted_path));
}

// ─── Typed scalar accessors (non-throwing flavor; noexcept) ──────────────────
// Wrap the throwing resolve+convert logic and swallow every Conf_Error, yielding
// std::nullopt on a malformed path, a missing key, or a type mismatch.

std::optional<int> Config::try_int(std::string_view path) const noexcept {
    try {
        if (!impl_) {
            return std::nullopt;
        }
        return impl_->tree.convert<int>(impl_->tree.resolve(path));
    } catch (const Conf_Error&) {
        return std::nullopt;
    }
}

std::optional<double> Config::try_double(std::string_view path) const noexcept {
    try {
        if (!impl_) {
            return std::nullopt;
        }
        return impl_->tree.convert<double>(impl_->tree.resolve(path));
    } catch (const Conf_Error&) {
        return std::nullopt;
    }
}

std::optional<bool> Config::try_bool(std::string_view path) const noexcept {
    try {
        if (!impl_) {
            return std::nullopt;
        }
        return impl_->tree.convert<bool>(impl_->tree.resolve(path));
    } catch (const Conf_Error&) {
        return std::nullopt;
    }
}

std::optional<std::string> Config::try_string(std::string_view path) const noexcept {
    try {
        if (!impl_) {
            return std::nullopt;
        }
        return impl_->tree.convert<std::string>(impl_->tree.resolve(path));
    } catch (const Conf_Error&) {
        return std::nullopt;
    }
}

// ─── Defaulted accessor (noexcept; returns fallback on any failure) ──────────

template <typename T>
T Config::get_or(std::string_view dotted_path, T fallback) const noexcept {
    try {
        if (!impl_) {
            return fallback;
        }
        return impl_->tree.convert<T>(impl_->tree.resolve(dotted_path));
    } catch (const Conf_Error&) {
        return fallback;
    }
}

// Explicit instantiations for the supported fallback types. get_or<T> is
// declared without a body in the header, so these definitions must live here.
template int Config::get_or<int>(std::string_view, int) const noexcept;
template double Config::get_or<double>(std::string_view, double) const noexcept;
template bool Config::get_or<bool>(std::string_view, bool) const noexcept;
template std::string Config::get_or<std::string>(std::string_view, std::string) const noexcept;

// ─── Generic node access (Value view) ────────────────────────────────────────

Value Config::at(std::string_view dotted_path) const {
    if (!impl_) {
        throw Conf_Error(Error_Code::Key_Not_Found, "configuration is empty");
    }
    // resolve() raises Invalid_Arg / Key_Not_Found before we touch at_nodes_, so
    // a failed call leaves the Config (and its node store) unchanged. On success
    // we keep the resolved node alive inside the deque and hand the Value a
    // pointer to that stored node — the exact handle value.cpp expects.
    YAML::Node node = impl_->tree.resolve(dotted_path);
    impl_->at_nodes_.push_back(node);
    return Value(static_cast<const void*>(&impl_->at_nodes_.back()));
}

// ─── List accessors ──────────────────────────────────────────────────────────
// resolve() (Invalid_Arg / Key_Not_Found) → sequence check (Type_Mismatch) →
// per-element convert (Type_Mismatch). Elements are converted into a LOCAL
// vector that is only returned once every element succeeds, so a mid-sequence
// mismatch propagates without ever returning a partial vector. An empty sequence
// yields an empty vector.

std::vector<int> Config::get_int_list(std::string_view path) const {
    if (!impl_) {
        throw Conf_Error(Error_Code::Key_Not_Found, "configuration is empty");
    }
    const YAML::Node node = impl_->tree.resolve(path);
    if (!node.IsSequence()) {
        throw Conf_Error(Error_Code::Type_Mismatch, "node is not a sequence");
    }
    std::vector<int> result;
    result.reserve(node.size());
    for (auto&& element : node) {
        result.push_back(impl_->tree.convert<int>(element));
    }
    return result;
}

std::vector<double> Config::get_double_list(std::string_view path) const {
    if (!impl_) {
        throw Conf_Error(Error_Code::Key_Not_Found, "configuration is empty");
    }
    const YAML::Node node = impl_->tree.resolve(path);
    if (!node.IsSequence()) {
        throw Conf_Error(Error_Code::Type_Mismatch, "node is not a sequence");
    }
    std::vector<double> result;
    result.reserve(node.size());
    for (auto&& element : node) {
        result.push_back(impl_->tree.convert<double>(element));
    }
    return result;
}

std::vector<std::string> Config::get_string_list(std::string_view path) const {
    if (!impl_) {
        throw Conf_Error(Error_Code::Key_Not_Found, "configuration is empty");
    }
    const YAML::Node node = impl_->tree.resolve(path);
    if (!node.IsSequence()) {
        throw Conf_Error(Error_Code::Type_Mismatch, "node is not a sequence");
    }
    std::vector<std::string> result;
    result.reserve(node.size());
    for (auto&& element : node) {
        result.push_back(impl_->tree.convert<std::string>(element));
    }
    return result;
}

} // namespace conf
