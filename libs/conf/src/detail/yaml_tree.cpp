/// @file detail/yaml_tree.cpp
/// @brief Implementation of the private yaml-cpp backend for CONF.
///
/// This translation unit (together with detail/yaml_tree.hpp) is the ONLY place
/// in CONF that touches yaml-cpp. It implements:
///   * from_file / from_string — load a document, translating backend failures
///     into the CONF error taxonomy (File_Not_Found / Parse_Error) while
///     treating empty/whitespace/comment-only input as a valid empty document.
///   * resolve — the total dotted-path resolver described in the design
///     ("Dotted-Path Resolution Algorithm"). It never crashes, aborts, or leaks
///     on any input; every failure maps to a typed Conf_Error.
///   * convert<T> — the typed scalar conversion described in the design
///     ("Typed Conversion Algorithm"), explicitly instantiated for the four
///     supported scalar types (int, double, bool, std::string).
///
/// Implementation notes on yaml-cpp handle semantics:
///   * YAML::Node has reference (handle) semantics: copying a Node aliases the
///     same underlying tree memory and shares its ref-counted lifetime.
///   * Node::operator= assigns *content* (and would mutate the tree). To re-seat
///     a handle onto a child node without mutating anything we therefore use
///     Node::reset(child), never operator=.
///   * Reading a map child via operator[] returns an undefined Node when the key
///     is absent and does not insert into the map, so resolution leaves the tree
///     unmodified.

#include "detail/yaml_tree.hpp"

#include <charconv>
#include <cstddef>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <vector>

namespace conf::detail {

namespace {

/// @brief Human-readable name for a supported conversion target type.
/// Used only to build diagnostic messages for Type_Mismatch errors.
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

} // namespace

// ─── Construction / loading ──────────────────────────────────────────────────

Yaml_Tree::Yaml_Tree(YAML::Node root) : root_(root) {}

Yaml_Tree Yaml_Tree::from_file(const std::string& path) {
    try {
        // LoadFile throws YAML::BadFile if the path cannot be opened (missing
        // file, permission denial, ...) and YAML::ParserException on a syntax
        // error. The temporary is handed straight to the private constructor so
        // no Node::operator= is involved.
        return Yaml_Tree(YAML::LoadFile(path));
    } catch (const YAML::BadFile& e) {
        // Any inability to open the path maps to File_Not_Found without
        // distinguishing the underlying cause.
        throw Conf_Error(Error_Code::File_Not_Found, e.what());
    } catch (const YAML::Exception& e) {
        // Every other backend failure on a real file is a syntax/parse error.
        // The backend's diagnostic text (which carries the line/column mark) is
        // preserved in the message.
        throw Conf_Error(Error_Code::Parse_Error, e.what());
    }
}

Yaml_Tree Yaml_Tree::from_string(const std::string& text) {
    try {
        // Empty, whitespace-only, or comment-only input does not throw: yaml-cpp
        // returns a Null/Undefined root, which we keep as a valid empty document
        // rather than reporting Parse_Error.
        return Yaml_Tree(YAML::Load(text));
    } catch (const YAML::Exception& e) {
        throw Conf_Error(Error_Code::Parse_Error, e.what());
    }
}

// ─── Dotted-path resolution ──────────────────────────────────────────────────

YAML::Node Yaml_Tree::resolve(std::string_view dotted_path) const {
    // ── Precondition: reject an empty path before walking anything. ──
    if (dotted_path.empty()) {
        throw Conf_Error(Error_Code::Invalid_Arg, "empty dotted path");
    }

    // ── Split on '.' and reject any empty segment (leading/trailing/double dot)
    //    BEFORE walking a single node, so a malformed path never touches the
    //    tree and is always reported as Invalid_Arg. ──
    std::vector<std::string_view> segments;
    std::size_t start = 0;
    while (true) {
        const std::size_t dot = dotted_path.find('.', start);
        const std::string_view seg = (dot == std::string_view::npos)
                                          ? dotted_path.substr(start)
                                          : dotted_path.substr(start, dot - start);
        if (seg.empty()) {
            throw Conf_Error(Error_Code::Invalid_Arg, "empty path segment");
        }
        segments.push_back(seg);
        if (dot == std::string_view::npos) {
            break;
        }
        start = dot + 1;
    }

    // ── Walk the tree one segment at a time. ──
    // 'current' is a handle aliasing the same memory as root_; it is re-seated
    // with reset() (never operator=) so the tree is never mutated.
    YAML::Node current(root_);
    for (const std::string_view seg : segments) {
        if (current.IsMap()) {
            // Literal, byte-for-byte, case-sensitive key match. Reading an
            // absent key yields an undefined node without modifying the map.
            YAML::Node child = current[std::string(seg)];
            if (!child.IsDefined()) {
                throw Conf_Error(Error_Code::Key_Not_Found, std::string(dotted_path));
            }
            current.reset(child);
        } else if (current.IsSequence()) {
            // The segment must consist solely of ASCII decimal digits and denote
            // an in-range index. from_chars (base 10, unsigned target) rejects
            // signs, whitespace, and non-digit characters, and reports
            // result_out_of_range for an oversized digit run instead of
            // overflowing — guaranteeing totality on arbitrary-magnitude input.
            std::size_t index = 0;
            const char* first = seg.data();
            const char* last = seg.data() + seg.size();
            const auto [ptr, ec] = std::from_chars(first, last, index);
            if (ec != std::errc{} || ptr != last || index >= current.size()) {
                throw Conf_Error(Error_Code::Key_Not_Found, std::string(dotted_path));
            }
            current.reset(current[index]);
        } else {
            // Scalar, Null, or Undefined node: cannot descend further.
            throw Conf_Error(Error_Code::Key_Not_Found, std::string(dotted_path));
        }
    }

    // Postcondition: 'current' is a defined node reached by the full path
    // (Map / Sequence / Scalar / Null — never Undefined).
    return current;
}

// ─── Typed conversion ────────────────────────────────────────────────────────

template <typename T>
T Yaml_Tree::convert(const YAML::Node& n) const {
    // A non-scalar node can never convert to a numeric/bool type. (For T ==
    // std::string the backend itself rejects non-scalar nodes, which we map to
    // Type_Mismatch in the catch below.)
    if (!n.IsScalar() && !std::is_same_v<T, std::string>) {
        throw Conf_Error(Error_Code::Type_Mismatch, "node is not a scalar");
    }

    try {
        return n.as<T>();
    } catch (const YAML::Exception&) {
        throw Conf_Error(Error_Code::Type_Mismatch,
                         std::string("cannot convert scalar to ") + type_name<T>());
    }
}

// ─── Explicit instantiations for the supported scalar types ──────────────────
template int Yaml_Tree::convert<int>(const YAML::Node&) const;
template double Yaml_Tree::convert<double>(const YAML::Node&) const;
template bool Yaml_Tree::convert<bool>(const YAML::Node&) const;
template std::string Yaml_Tree::convert<std::string>(const YAML::Node&) const;

} // namespace conf::detail
