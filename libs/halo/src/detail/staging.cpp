/// @file detail/staging.cpp
/// @brief Compilation unit for host-staging buffer utilities.
///
/// The stage_send() and stage_recv() functions are fully defined as
/// function templates in halo/detail/staging.hpp (header-only).
/// This translation unit ensures the header is self-contained and
/// compiles cleanly. No explicit template instantiations are needed
/// because the exchange function templates (which call stage_send/stage_recv)
/// are themselves instantiated at the call site.

#include <halo/detail/staging.hpp>

// Intentionally empty — all staging logic is header-only template code.
