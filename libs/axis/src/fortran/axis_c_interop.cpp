// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

/// @file src/fortran/axis_c_interop.cpp
/// @brief extern "C" interop functions for the Fortran iso_c_binding layer.
///
/// These functions provide a flat C ABI that the axis_mod.f90 Fortran module
/// calls via iso_c_binding. All C++ exceptions are caught and translated to
/// integer error codes (AXIS_SUCCESS / AXIS_ERROR). Opaque integer handles
/// expose AXIS objects (meshes, interpolation matrices) to Fortran memory.
///
/// AXIS uses on-device memory spaces, but these C bindings handle device-resident
/// data structures transparently by managing them within the Host Space handle register.

#include <Kokkos_Core.hpp>
#include <axis/ingest/grid_descriptor.hpp>
#include <axis/solver/apply.hpp>
#include <axis/solver/conservation.hpp>
#include <axis/solver/interpolation_matrix.hpp>
#include <axis/solver/regrid_config.hpp>
#include <axis/solver/vector_regridder.hpp>
#include <axis/solver/weight_cache.hpp>
#include <axis/solver/weight_generator.hpp>
#include <axis/topology/mesh_factory.hpp>
#include <axis/topology/named_grid_registry.hpp>
#include <axis/topology/projection_builder.hpp>
#include <axis/topology/structured_grid.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>
#include <memory>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// Error codes (match the Fortran module constants)
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Error code constant indicating successful completion.
#define AXIS_SUCCESS 0

/// @brief Error code constant indicating a generic runtime error or exception.
#define AXIS_ERROR -1

// ─────────────────────────────────────────────────────────────────────────────
// Exception-to-error-code macro
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Wraps an expression in a try/catch block that returns AXIS_SUCCESS on
/// normal completion or AXIS_ERROR on any exception. This provides a uniform
/// exception barrier at the C/Fortran boundary.
///
/// @param expr The C++ expression or statement block to execute safely.
#define AXIS_C_TRY(expr)         \
    do {                         \
        try {                    \
            expr;                \
            return AXIS_SUCCESS; \
        } catch (...) {          \
            return AXIS_ERROR;   \
        }                        \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// Type aliases for the host-space types we expose through the C layer
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Type alias for the unstructured mesh type residing in host memory space.
using HostMesh = axis::topology::UnstructuredMesh<Kokkos::HostSpace>;

/// @brief Type alias for the sparse interpolation matrix residing in host memory space.
using HostMatrix = axis::solver::InterpolationMatrix<Kokkos::HostSpace>;

// ─────────────────────────────────────────────────────────────────────────────
// Handle registry helpers (declared as extern "C" in handle_registry.hpp)
// ─────────────────────────────────────────────────────────────────────────────

#include "handle_registry.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// extern "C" interop functions
// ─────────────────────────────────────────────────────────────────────────────

extern "C" {

/// @brief Initialize the AXIS runtime (Kokkos). Safe to call multiple times.
/// @return @c AXIS_SUCCESS on success, @c AXIS_ERROR on failure.
int axis_init_c() {
    AXIS_C_TRY(if (!Kokkos::is_initialized()) { Kokkos::initialize(); });
}

/// @brief Build a mesh from a named-grid token (e.g. "O1280", "F128", "N320").
/// @param[in]  name         Null-terminated grid name string.
/// @param[out] mesh_handle  Pointer to integer token for the created mesh.
/// @return @c AXIS_SUCCESS on success, @c AXIS_ERROR on failure.
int axis_mesh_from_named_c(const char *name, int *mesh_handle) {
    AXIS_C_TRY(auto mesh = std::make_shared<HostMesh>(axis::topology::MeshFactory::from_named<Kokkos::HostSpace>(std::string(name)));
               auto &reg = axis::fortran::Handle_Registry::instance(); *mesh_handle = reg.register_handle(std::move(mesh)););
}

/// @brief Build a mesh from a GridDescriptor (passed as opaque pointer from Fortran).
/// @param[in]  descriptor_ptr  Pointer to a populated GridDescriptor.
/// @param[out] mesh_handle     Pointer to integer token for the created mesh.
/// @return @c AXIS_SUCCESS on success, @c AXIS_ERROR on failure.
int axis_mesh_from_descriptor_c(const void *descriptor_ptr, int *mesh_handle) {
    AXIS_C_TRY(const auto *desc = static_cast<const axis::ingest::GridDescriptor *>(descriptor_ptr);
               auto mesh = std::make_shared<HostMesh>(axis::topology::MeshFactory::from_descriptor<Kokkos::HostSpace>(*desc));
               auto &reg = axis::fortran::Handle_Registry::instance(); *mesh_handle = reg.register_handle(std::move(mesh)););
}

/// @brief Generate interpolation weights between source and destination meshes.
/// @param[in]  src_handle     Integer token for the source mesh.
/// @param[in]  dst_handle     Integer token for the destination mesh.
/// @param[in]  method         Interpolation method (0=Bilinear, 1=NearestNeighbor, 2=Conservative1st).
/// @param[out] matrix_handle  Pointer to integer token for the created interpolation matrix.
/// @return @c AXIS_SUCCESS on success, @c AXIS_ERROR on failure.
int axis_generate_weights_c(int src_handle, int dst_handle, int method, int *matrix_handle) {
    AXIS_C_TRY(
        auto &reg = axis::fortran::Handle_Registry::instance();

        auto src_ptr = reg.lookup(src_handle); auto dst_ptr = reg.lookup(dst_handle); if (!src_ptr || !dst_ptr) { return AXIS_ERROR; }

                                                                                      auto &src_mesh = *std::static_pointer_cast<HostMesh>(src_ptr);
        auto &dst_mesh = *std::static_pointer_cast<HostMesh>(dst_ptr);

        // Map integer method code to InterpolationMethod enum
        axis::solver::RegridConfig config;
        switch (method) {
            case 0:
                config.method = axis::solver::InterpolationMethod::Bilinear;
                break;
            case 1:
                config.method = axis::solver::InterpolationMethod::NearestNeighbor;
                break;
            case 2:
                config.method = axis::solver::InterpolationMethod::Conservative1stOrder;
                break;
            default:
                return AXIS_ERROR;
        }

        auto matrix = std::make_shared<HostMatrix>(axis::solver::WeightGenerator::generate<Kokkos::HostSpace>(src_mesh, dst_mesh, config));

        *matrix_handle = reg.register_handle(std::move(matrix)););
}

/// @brief Generate coupled vector interpolation weights between source and destination meshes.
/// @param[in]  src_handle      Integer token for the source mesh.
/// @param[in]  dst_handle      Integer token for the destination mesh.
/// @param[in]  src_alpha       Pointer to source grid cell rotation angles [n_src].
/// @param[in]  dst_alpha       Pointer to destination grid cell rotation angles [n_dst].
/// @param[in]  method          Interpolation method (0=Bilinear, 1=NearestNeighbor, 2=Conservative1st).
/// @param[out] matrix_u_handle Pointer to integer token for the created U-component interpolation matrix.
/// @param[out] matrix_v_handle Pointer to integer token for the created V-component interpolation matrix.
/// @return @c AXIS_SUCCESS on success, @c AXIS_ERROR on failure.
int axis_generate_vector_weights_c(int src_handle, int dst_handle, const double *src_alpha, const double *dst_alpha, int method, int *matrix_u_handle,
                                   int *matrix_v_handle) {
    try {
        auto &reg = axis::fortran::Handle_Registry::instance();

        auto src_ptr = reg.lookup(src_handle);
        auto dst_ptr = reg.lookup(dst_handle);
        if (!src_ptr || !dst_ptr) {
            return AXIS_ERROR;
        }

        auto &src_mesh = *std::static_pointer_cast<HostMesh>(src_ptr);
        auto &dst_mesh = *std::static_pointer_cast<HostMesh>(dst_ptr);

        // Map integer method code to InterpolationMethod enum
        axis::solver::RegridConfig config;
        switch (method) {
            case 0:
                config.method = axis::solver::InterpolationMethod::Bilinear;
                break;
            case 1:
                config.method = axis::solver::InterpolationMethod::NearestNeighbor;
                break;
            case 2:
                config.method = axis::solver::InterpolationMethod::Conservative1stOrder;
                break;
            default:
                return AXIS_ERROR;
        }

        // Construct unmanaged Views for GridRotation
        Kokkos::View<const double *, Kokkos::HostSpace> src_rot_view(src_alpha, src_mesh.n_cells());
        Kokkos::View<const double *, Kokkos::HostSpace> dst_rot_view(dst_alpha, dst_mesh.n_cells());

        axis::solver::GridRotation<Kokkos::HostSpace> src_rot{src_rot_view};
        axis::solver::GridRotation<Kokkos::HostSpace> dst_rot{dst_rot_view};

        auto [W_u, W_v] = axis::solver::VectorWeightGenerator<Kokkos::HostSpace>::generate(src_mesh, dst_mesh, src_rot, dst_rot, config);

        auto W_u_ptr = std::make_shared<HostMatrix>(std::move(W_u));
        auto W_v_ptr = std::make_shared<HostMatrix>(std::move(W_v));

        *matrix_u_handle = reg.register_handle(std::move(W_u_ptr));
        *matrix_v_handle = reg.register_handle(std::move(W_v_ptr));

        return AXIS_SUCCESS;
    } catch (...) {
        return AXIS_ERROR;
    }
}

/// @brief Apply interpolation weights: dst = S * src.
/// @details Constructs non-owning field_view over the Fortran contiguous arrays
/// (passed via c_loc). No data copy occurs — HELM Law #1 zero-copy.
///
/// @param[in]  matrix_handle  Integer token for the interpolation matrix.
/// @param[in]  src            Pointer to source field array [n_src] (Fortran c_loc).
/// @param[out] dst            Pointer to destination field array [n_dst] (Fortran c_loc).
/// @param[in]  n_src          Number of source field elements.
/// @param[in]  n_dst          Number of destination field elements.
/// @return @c AXIS_SUCCESS on success, @c AXIS_ERROR on failure.
int axis_apply_c(int matrix_handle, const double *src, double *dst, int n_src, int n_dst) {
    try {
        auto &reg = axis::fortran::Handle_Registry::instance();
        auto mat_ptr = reg.lookup(matrix_handle);
        if (!mat_ptr) {
            return AXIS_ERROR;
        }

        auto matrix_ptr = std::static_pointer_cast<HostMatrix>(mat_ptr);

        // Wrap raw C/Fortran array pointers into non-owning field_views (zero-copy)
        axis::field_view<const double, 1> src_view(src, static_cast<std::size_t>(n_src));
        axis::field_view<double, 1> dst_view(dst, static_cast<std::size_t>(n_dst));

        axis::solver::apply<Kokkos::HostSpace>(*matrix_ptr, src_view, dst_view);
        return AXIS_SUCCESS;
    } catch (...) {
        return AXIS_ERROR;
    }
}

/// @brief Destroy (release) a mesh handle inside C++.
/// @param[in]  handle Opaque integer handle of the mesh to destroy.
/// @return @c AXIS_SUCCESS on success, @c AXIS_ERROR on failure.
int axis_destroy_mesh_c(int handle) {
    AXIS_C_TRY(auto &reg = axis::fortran::Handle_Registry::instance(); reg.release(handle););
}

/// @brief Destroy (release) an interpolation matrix handle inside C++.
/// @param[in]  handle Opaque integer handle of the interpolation matrix to destroy.
/// @return @c AXIS_SUCCESS on success, @c AXIS_ERROR on failure.
int axis_destroy_matrix_c(int handle) {
    AXIS_C_TRY(auto &reg = axis::fortran::Handle_Registry::instance(); reg.release(handle););
}

}  // extern "C"
