#ifndef CONF_DETAIL_YAML_TREE_HPP
#define CONF_DETAIL_YAML_TREE_HPP

/// @file detail/yaml_tree.hpp
/// @brief Private yaml-cpp backend for CONF (NOT installed, NOT public).
///
/// This is the ONLY translation-unit header in CONF that includes
/// <yaml-cpp/yaml.h>. It seals every yaml-cpp type inside the conf::detail
/// namespace so that no YAML::Node ever leaks into a public conf:: header or
/// across the Fortran C bridge. The backend swap option (yaml-cpp -> rapidyaml)
/// documented in the design remains open precisely because this dependency is
/// confined here.
///
/// LOCATION CONTRACT: this header lives under src/detail/ and is compiled into
/// the library only. It is never placed under include/ and is never installed
/// or exported. Public consumers of HELM::CONF see only conf:: and standard-
/// library types.
///
/// This header may include conf/error.hpp (for conf::Conf_Error / Error_Code);
/// it pulls in no other HELM header, upholding the Tier-1 No-Circular-Dependency
/// law.

#include <string>
#include <string_view>

#include <yaml-cpp/yaml.h>

#include <conf/error.hpp>

namespace conf::detail {

/// @brief RAII wrapper around a parsed yaml-cpp document tree.
///
/// Owns the root YAML::Node produced by parsing a file or string, implements
/// dotted-path resolution against the node graph, and performs typed scalar
/// conversions. Instances are created exclusively through the from_file /
/// from_string factories; the parsing constructor is private.
///
/// All members are declared here; their definitions live in
/// src/detail/yaml_tree.cpp (the template convert<T> is explicitly instantiated
/// there for the supported scalar types).
class Yaml_Tree {
public:
    /// @brief Parse a YAML document from a file path.
    /// @param path Filesystem path to a readable YAML file.
    /// @return A Yaml_Tree owning the parsed root node.
    /// @throws conf::Conf_Error with Error_Code::File_Not_Found if the path
    ///         cannot be opened, or Error_Code::Parse_Error on a syntax error.
    [[nodiscard]] static Yaml_Tree from_file(const std::string& path);

    /// @brief Parse a YAML document from an in-memory string.
    /// @param text YAML source text.
    /// @return A Yaml_Tree owning the parsed root node.
    /// @throws conf::Conf_Error with Error_Code::Parse_Error on a syntax error.
    [[nodiscard]] static Yaml_Tree from_string(const std::string& text);

    /// @brief Resolve a dotted-path key to a node within the tree.
    ///
    /// Walks the tree one '.'-delimited segment at a time: map nodes advance by
    /// literal (byte-for-byte, no trimming or case-folding) key match, sequence
    /// nodes advance by an in-range ASCII-decimal index. The resolver is total:
    /// it never crashes, aborts, or leaks on any input.
    ///
    /// @param dotted_path A '.'-delimited key string (for example "a.b.0.c").
    /// @return The node reached by the full path.
    /// @throws conf::Conf_Error with Error_Code::Invalid_Arg if the path is
    ///         empty or has any empty segment (checked before walking), or
    ///         Error_Code::Key_Not_Found if a segment does not resolve.
    [[nodiscard]] YAML::Node resolve(std::string_view dotted_path) const;

    /// @brief Convert a resolved scalar node to the requested type.
    ///
    /// Enforces that the node is a scalar (or that T is std::string) and
    /// delegates to yaml-cpp's YAML::Node::as<T>(). Conversion failures from the
    /// backend are translated into a typed Conf_Error.
    ///
    /// @tparam T The target type (int, double, bool, or std::string).
    /// @param n A resolved node (assumed defined; produced by resolve()).
    /// @return The node value converted to T.
    /// @throws conf::Conf_Error with Error_Code::Type_Mismatch if the node is
    ///         not convertible to T.
    template <typename T>
    [[nodiscard]] T convert(const YAML::Node& n) const;

private:
    /// @brief Construct from an already-parsed root node.
    /// @param root The parsed yaml-cpp root node to take ownership of.
    explicit Yaml_Tree(YAML::Node root);

    YAML::Node root_;  ///< Owned root of the parsed document tree.
};

} // namespace conf::detail

#endif // CONF_DETAIL_YAML_TREE_HPP
