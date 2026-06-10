// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

/// @file src/fortran/axis_c_interop.cpp
/// @brief extern "C" interop functions for the Fortran iso_c_binding layer.
///
/// These functions provide a flat C ABI that the axis_mod.f90 Fortran module
/// calls via iso_c_binding. All C++ exceptions are caught and translated to
/// integer error codes (AXIS_SUCCESS / AXIS_ERROR). Opaque integer handles
/// expose AXIS objects (meshes, interpolation matrices) to Fortran without
/// revealing C++ class internals.
///
/// Convention:
///   - All functions return int (error code).
///   - Output handles are written via pointer arguments.
///   - Field data is passed as contiguous double* arrays (Fortran c_loc).
///   - Token 0 is INVALID (never returned by register_handle).

#include "handle_registry.hpp"

#include <axis/solver/apply.hpp>
#include <axis/solver/interpolation_matrix.hpp>
#include <axis/solver/regrid_config.hpp>
#include <axis/solver/weight_generator.hpp>
#include <axis/topology/mesh_factory.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>

#include <Kokkos_Core.hpp>

#include <memory>
#include <string>

// ─────────────────────────────────────────────────────────────────────────────
// Error codes (match the Fortran module constants)
// ─────────────────────────────────────────────────────────────────────────────

#define AXIS_SUCCESS  0
#define AXIS_ERROR   -1

// ─────────────────────────────────────────────────────────────────────────────
// Exception-to-error-code macro
// ─────────────────────────────────────────────────────────────────────────────

/// Wraps an expression in a try/catch block that returns AXIS_SUCCESS on
/// normal completion or AXIS_ERROR on any exception. This provides a uniform
/// exception barrier at the C/Fortran boundary.
#define AXIS_C_TRY(expr) \
    do { \
        try { \
            expr; \
            return AXIS_SUCCESS; \
        } catch (...) { \
            return AXIS_ERROR; \
        } \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// Type aliases for the host-space types we expose through the C layer
// ─────────────────────────────────────────────────────────────────────────────

using HostMesh   = axis::topology::UnstructuredMesh<Kokkos::HostSpace>;
using HostMatrix = axis::solver::InterpolationMatrix<Kokkos::HostSpace>;

// ─────────────────────────────────────────────────────────────────────────────
// extern "C" interop functions
// ─────────────────────────────────────────────────────────────────────────────

extern "C" {

/// Initialize the AXIS runtime (Kokkos). Safe to call multiple times —
/// Kokkos::is_initialized() guards re-initialization.
///
/// @return AXIS_SUCCESS on success, AXIS_ERROR on failure.
int axis_init_c() {
    AXIS_C_TRY(
        if (!Kokkos::is_initialized()) {
            Kokkos::initialize();
        }
    );
}

/// Build a mesh from a named-grid token (e.g. "O1280", "F128", "N320").
///
/// @param name         Null-terminated grid name string.
/// @param mesh_handle  [out] Integer token for the created mesh.
/// @return AXIS_SUCCESS on success, AXIS_ERROR on failure.
int axis_mesh_from_named_c(const char* name, int* mesh_handle) {
    AXIS_C_TRY(
        auto mesh = std::make_shared<HostMesh>(
            axis::topology::MeshFactory::from_named<Kokkos::HostSpace>(
                std::string(name)));
        auto& reg = axis::fortran::Handle_Registry::instance();
        *mesh_handle = reg.register_handle(std::move(mesh));
    );
}

/// Build a mesh from a GridDescriptor (passed as opaque pointer from Fortran).
/// NOTE: This is a placeholder for advanced use — typically Fortran callers
/// use axis_mesh_from_named_c. A full descriptor bridge would require
/// additional marshalling functions.
///
/// @param descriptor_ptr  Pointer to a populated GridDescriptor.
/// @param mesh_handle     [out] Integer token for the created mesh.
/// @return AXIS_SUCCESS on success, AXIS_ERROR on failure.
int axis_mesh_from_descriptor_c(const void* descriptor_ptr, int* mesh_handle) {
    AXIS_C_TRY(
        const auto* desc = static_cast<const axis::ingest::GridDescriptor*>(descriptor_ptr);
        auto mesh = std::make_shared<HostMesh>(
            axis::topology::MeshFactory::from_descriptor<Kokkos::HostSpace>(*desc));
        auto& reg = axis::fortran::Handle_Registry::instance();
        *mesh_handle = reg.register_handle(std::move(mesh));
    );
}

/// Generate interpolation weights between source and destination meshes.
///
/// @param src_handle     Integer token for the source mesh.
/// @param dst_handle     Integer token for the destination mesh.
/// @param method         Interpolation method (0=Bilinear, 1=NearestNeighbor, 2=Conservative1st).
/// @param matrix_handle  [out] Integer token for the created interpolation matrix.
/// @return AXIS_SUCCESS on success, AXIS_ERROR on failure.
int axis_generate_weights_c(int src_handle, int dst_handle, int method,
                            int* matrix_handle) {
    AXIS_C_TRY(
        auto& reg = axis::fortran::Handle_Registry::instance();

        auto src_ptr = reg.lookup(src_handle);
        auto dst_ptr = reg.lookup(dst_handle);
        if (!src_ptr || !dst_ptr) {
            return AXIS_ERROR;
        }

        auto& src_mesh = *std::static_pointer_cast<HostMesh>(src_ptr);
        auto& dst_mesh = *std::static_pointer_cast<HostMesh>(dst_ptr);

        // Map integer method code to InterpolationMethod enum
        axis::solver::RegridConfig config;
        switch (method) {
            case 0: config.method = axis::solver::InterpolationMethod::Bilinear; break;
            case 1: config.method = axis::solver::InterpolationMethod::NearestNeighbor; break;
            case 2: config.method = axis::solver::InterpolationMethod::Conservative1stOrder; break;
            default: return AXIS_ERROR;
        }

        auto matrix = std::make_shared<HostMatrix>(
            axis::solver::WeightGenerator::generate<Kokkos::HostSpace>(
                src_mesh, dst_mesh, config));

        *matrix_handle = reg.register_handle(std::move(matrix));
    );
}

/// Apply interpolation weights: dst = S · src.
///
/// Constructs non-owning field_view over the Fortran contiguous arrays
/// (passed via c_loc). No data copy occurs — HELM Law #1 zero-copy.
///
/// @param matrix_handle  Integer token for the interpolation matrix.
/// @param src            Pointer to source field array [n_src] (Fortran c_loc).
/// @param dst            Pointer to destination field array [n_dst] (Fortran c_loc).
/// @param n_src          Number of source field elements.
/// @param n_dst          Number of destination field elements.
/// @return AXIS_SUCCESS on success, AXIS_ERROR on failure.
int axis_apply_c(int matrix_handle, const double* src, double* dst,
                 int n_src, int n_dst) {
    try {
        auto& reg = axis::fortran::Handle_Registry::instance();

        auto mat_ptr = reg.lookup(matrix_handle);
        if (!mat_ptr) {
            return AXIS_ERROR;
        }

        auto& matrix = *std::static_pointer_cast<HostMatrix>(mat_ptr);

        // Construct non-owning field_view over Fortran contiguous arrays.
        // Using explicit type aliases to avoid commas in macro arguments.
        using src_view_t = axis::field_view<const double, 1>;
        using dst_view_t = axis::field_view<double, 1>;

        src_view_t src_view(src, static_cast<std::size_t>(n_src));
        dst_view_t dst_view(dst, static_cast<std::size_t>(n_dst));

        axis::solver::apply<Kokkos::HostSpace>(matrix, src_view, dst_view);
        return AXIS_SUCCESS;
    } catch (...) {
        return AXIS_ERROR;
    }
}

/// Destroy (release) a mesh handle.
///
/// @param handle  Integer token for the mesh to release.
/// @return AXIS_SUCCESS on success, AXIS_ERROR on failure.
int axis_destroy_mesh_c(int handle) {
    AXIS_C_TRY(
        axis::fortran::Handle_Registry::instance().release(handle);
    );
}

/// Destroy (release) an interpolation matrix handle.
///
/// @param handle  Integer token for the matrix to release.
/// @return AXIS_SUCCESS on success, AXIS_ERROR on failure.
int axis_destroy_matrix_c(int handle) {
    AXIS_C_TRY(
        axis::fortran::Handle_Registry::instance().release(handle);
    );
}

} // extern "C"
