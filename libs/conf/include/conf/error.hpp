#ifndef CONF_ERROR_HPP
#define CONF_ERROR_HPP

/// @file error.hpp
/// @brief Stable error taxonomy and exception type for CONF.
///
/// This header defines the single, stable failure-mode enumeration shared by the
/// C++ core (as a conf::Conf_Error payload) and the C bridge (as the returned
/// int error code). Keeping the Fortran-visible error codes and the C++ exception
/// taxonomy in lockstep mirrors HALO's error model.
///
/// This is a public CONF header. It includes only standard-library headers; it
/// pulls in no yaml-cpp type and no other HELM header, upholding the Tier-1
/// No-Circular-Dependency law.

#include <stdexcept>
#include <string>

namespace conf {

/// @brief Stable error taxonomy for all CONF operations.
///
/// The integer values are part of the C ABI and MUST match the parameter
/// constants declared in conf_mod.f90. The enumeration is append-only: existing
/// enumerators are never renumbered, reassigned, or removed; new failure modes
/// are added only as new enumerators with previously unused values.
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

/// @brief Exception type carrying an Error_Code plus a human-readable message.
///
/// Thrown by the CONF C++ core; caught and translated to an int error code at the
/// C boundary. Each instance carries exactly one Error_Code value and a non-empty
/// diagnostic message (accessible via the inherited what()).
class Conf_Error : public std::runtime_error {
public:
    /// @brief Construct a Conf_Error.
    /// @param code     The error code classifying this failure.
    /// @param what_msg A non-empty, human-readable diagnostic message.
    Conf_Error(Error_Code code, const std::string& what_msg)
        : std::runtime_error(what_msg), code_(code) {}

    /// @brief Return the Error_Code supplied at construction.
    /// @return The identical Error_Code value passed to the constructor.
    [[nodiscard]] Error_Code code() const noexcept { return code_; }

private:
    Error_Code code_;
};

} // namespace conf

#endif // CONF_ERROR_HPP
