// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_IO_ESMF_WEIGHT_IO_HPP
#define AXIS_IO_ESMF_WEIGHT_IO_HPP

#include <axis/solver/interpolation_matrix.hpp>
#include <string>

namespace axis::io {

/// @class EsmfWeightIO
/// @brief Handles reading and writing 1-based, ESMF-compliant NetCDF weight files.
///
/// This utility provides standard, zero-copy, compile-ready serialization for AXIS
/// InterpolationMatrix objects, supporting seamless integration with earth system models
/// expecting SCRIP/ESMF coordinate weights structures.
///
/// @note This class is only compiled when AXIS is built with NetCDF support.
/// @tparam MemorySpace The Kokkos memory space (e.g. HostSpace, CudaSpace).
template <typename MemorySpace>
class EsmfWeightIO {
public:
    /// @brief Write an InterpolationMatrix to an ESMF-compliant NetCDF weight file (1-based indices).
    /// @param filepath The destination NetCDF file path.
    /// @param matrix   The sparse InterpolationMatrix to write.
    /// @throws std::runtime_error if a NetCDF I/O error occurs.
    static void write_esmf(
        const std::string& filepath,
        const solver::InterpolationMatrix<MemorySpace>& matrix
    );

    /// @brief Read an ESMF-compliant NetCDF weight file into an InterpolationMatrix (translates 1-based to 0-based).
    /// @param filepath The source NetCDF weights file path.
    /// @return The deserialized InterpolationMatrix.
    /// @throws std::runtime_error if a NetCDF read or parse error occurs.
    static solver::InterpolationMatrix<MemorySpace> read_esmf(
        const std::string& filepath
    );
};

} // namespace axis::io

#endif // AXIS_IO_ESMF_WEIGHT_IO_HPP
