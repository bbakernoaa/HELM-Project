#ifndef CONF_ERROR_HPP
#define CONF_ERROR_HPP

#include <stdexcept>
#include <string>

namespace conf {

/// Stable error taxonomy. Integer values are part of the C ABI and MUST match
/// the parameter constants declared in conf_mod.f90. Append-only: never renumber.
enum class Error_Code : int {
    Success          = 0,   ///< Operation completed successfully.
    Invalid_Arg      = 1,   ///< Null pointer or malformed argument from caller.
    File_Not_Found   = 2,   ///< Config file path does not exist / cannot be opened.
    Parse_Error      = 3,   ///< Malformed YAML (syntax error at parse time).
    Key_Not_Found    = 4,   ///< Dotted-path key does not resolve to a node.
    Type_Mismatch    = 5,   ///< Node exists but cannot convert to requested type.
    Bad_Handle       = 6,   ///< Invalid or released opaque handle token.
    Buffer_Too_Small = 7,   ///< Caller string buffer cannot hold the value.
    Unknown          = 99   ///< Unexpected / uncategorized error.
};

/// Exception type carrying an Error_Code plus a human-readable message.
/// Thrown by the C++ core; caught and translated at the C boundary.
class Conf_Error : public std::runtime_error {
public:
    Conf_Error(Error_Code code, const std::string& what_msg)
        : std::runtime_error(what_msg), code_(code) {}

    [[nodiscard]] Error_Code code() const noexcept { return code_; }

private:
    Error_Code code_;
};

} // namespace conf

#endif // CONF_ERROR_HPP
