// SPDX-License-Identifier: Apache-2.0
// AXIS Python bindings via nanobind
//
// Exposes the AXIS interpolation engine to Python for benchmarking and
// interactive use. Uses numpy arrays as the zero-copy buffer interface.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <Kokkos_Core.hpp>

#include <axis/types.hpp>
#include <axis/topology/structured_grid.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/topology/mesh_factory.hpp>
#include <axis/topology/named_grid_registry.hpp>
#include <axis/solver/regrid_config.hpp>
#include <axis/solver/interpolation_matrix.hpp>
#include <axis/solver/weight_generator.hpp>
#include <axis/solver/apply.hpp>
#include <axis/solver/conservation.hpp>

namespace nb = nanobind;
using namespace nb::literals;

using HostMesh   = axis::topology::UnstructuredMesh<Kokkos::HostSpace>;
using HostMatrix = axis::solver::InterpolationMatrix<Kokkos::HostSpace>;

// ─── Kokkos lifecycle ────────────────────────────────────────────────────────

static bool kokkos_initialized = false;

void ensure_kokkos() {
    if (!kokkos_initialized && !Kokkos::is_initialized()) {
        Kokkos::initialize();
        kokkos_initialized = true;
    }
}

// ─── Helper: Build a regular lat-lon mesh from numpy arrays ──────────────────

HostMesh make_regular_mesh(std::size_t ni, std::size_t nj,
                           double lon_start, double lat_start,
                           double dlon, double dlat) {
    ensure_kokkos();

    const std::size_t n_centers = ni * nj;
    const std::size_t n_corners = (ni + 1) * (nj + 1);

    Kokkos::View<double*, Kokkos::HostSpace> cx("cx", n_centers);
    Kokkos::View<double*, Kokkos::HostSpace> cy("cy", n_centers);
    Kokkos::View<double*, Kokkos::HostSpace> crx("crx", n_corners);
    Kokkos::View<double*, Kokkos::HostSpace> cry("cry", n_corners);

    for (std::size_t j = 0; j < nj; ++j) {
        for (std::size_t i = 0; i < ni; ++i) {
            cx(i + j * ni) = lon_start + (static_cast<double>(i) + 0.5) * dlon;
            cy(i + j * ni) = lat_start + (static_cast<double>(j) + 0.5) * dlat;
        }
    }
    for (std::size_t j = 0; j <= nj; ++j) {
        for (std::size_t i = 0; i <= ni; ++i) {
            crx(i + j * (ni + 1)) = lon_start + static_cast<double>(i) * dlon;
            cry(i + j * (ni + 1)) = lat_start + static_cast<double>(j) * dlat;
        }
    }

    axis::topology::StructuredGrid<Kokkos::HostSpace> grid(
        ni, nj, std::move(cx), std::move(cy),
        axis::topology::CoordinateSystem::SphericalDeg);
    grid.set_corners(std::move(crx), std::move(cry));

    return grid.to_unstructured();
}

// ─── Module definition ───────────────────────────────────────────────────────

NB_MODULE(axis_py, m) {
    m.doc() = "AXIS Python bindings — spatial interpolation for Earth-system fields";

    // Interpolation method enum
    nb::enum_<axis::solver::InterpolationMethod>(m, "Method")
        .value("Bilinear", axis::solver::InterpolationMethod::Bilinear)
        .value("NearestNeighbor", axis::solver::InterpolationMethod::NearestNeighbor)
        .value("Bicubic", axis::solver::InterpolationMethod::Bicubic)
        .value("Patch", axis::solver::InterpolationMethod::Patch)
        .value("Conservative", axis::solver::InterpolationMethod::Conservative1stOrder);

    // NormType enum
    nb::enum_<axis::solver::NormType>(m, "NormType")
        .value("DstArea", axis::solver::NormType::DstArea)
        .value("FracArea", axis::solver::NormType::FracArea);

    // Mesh wrapper class
    nb::class_<HostMesh>(m, "Mesh")
        .def_prop_ro("n_nodes", &HostMesh::n_nodes)
        .def_prop_ro("n_cells", &HostMesh::n_cells);

    // Matrix wrapper class
    nb::class_<HostMatrix>(m, "Matrix")
        .def_prop_ro("nnz", &HostMatrix::nnz)
        .def_prop_ro("n_src", &HostMatrix::n_src)
        .def_prop_ro("n_dst", &HostMatrix::n_dst);

    // Make a regular lat-lon mesh
    m.def("make_regular_mesh", &make_regular_mesh,
          "ni"_a, "nj"_a, "lon_start"_a, "lat_start"_a, "dlon"_a, "dlat"_a,
          "Create a regular lat-lon UnstructuredMesh");

    // Make a named grid
    m.def("make_named_mesh", [](const std::string& name) -> HostMesh {
        ensure_kokkos();
        return axis::topology::NamedGridRegistry::generate<Kokkos::HostSpace>(name);
    }, "name"_a, "Generate a named grid (e.g., 'O32', 'F64')");

    // Generate weights
    m.def("generate_weights", [](const HostMesh& src, const HostMesh& dst,
                                  axis::solver::InterpolationMethod method) -> HostMatrix {
        ensure_kokkos();
        axis::solver::RegridConfig cfg;
        cfg.method = method;
        cfg.unmapped = axis::solver::UnmappedAction::Ignore;
        return axis::solver::WeightGenerator::generate<Kokkos::HostSpace>(src, dst, cfg);
    }, "src_mesh"_a, "dst_mesh"_a, "method"_a,
       "Generate interpolation weights between src and dst meshes");

    // Apply weights to a numpy array
    m.def("apply_weights", [](const HostMatrix& matrix,
                               nb::ndarray<nb::numpy, double, nb::ndim<1>> src_arr)
          -> nb::ndarray<nb::numpy, double> {
        ensure_kokkos();

        const std::size_t n_src = src_arr.shape(0);
        const std::size_t n_dst = matrix.n_dst();

        if (n_src != matrix.n_src()) {
            throw std::invalid_argument(
                "src array size (" + std::to_string(n_src) +
                ") != matrix.n_src (" + std::to_string(matrix.n_src()) + ")");
        }

        // Wrap numpy source as field_view (zero-copy)
        const double* src_ptr = src_arr.data();
        axis::field_view<const double, 1> src_view(src_ptr, n_src);

        // Allocate destination
        double* dst_ptr = new double[n_dst]();
        axis::field_view<double, 1> dst_view(dst_ptr, n_dst);

        // Apply
        axis::solver::apply<Kokkos::HostSpace>(matrix, src_view, dst_view);

        // Return as numpy array (with ownership)
        nb::capsule owner(dst_ptr, [](void* p) noexcept { delete[] static_cast<double*>(p); });
        return nb::ndarray<nb::numpy, double>(dst_ptr, {n_dst}, std::move(owner));
    }, "matrix"_a, "src"_a,
       "Apply interpolation matrix to source field, return destination array");

    // Conservation check
    m.def("check_conservation", [](const HostMatrix& matrix,
                                    nb::ndarray<nb::numpy, double, nb::ndim<1>> src_arr,
                                    nb::ndarray<nb::numpy, double, nb::ndim<1>> dst_arr)
          -> nb::dict {
        ensure_kokkos();

        axis::field_view<const double, 1> src_view(src_arr.data(), src_arr.shape(0));
        axis::field_view<const double, 1> dst_view(dst_arr.data(), dst_arr.shape(0));

        auto report = axis::solver::check_conservation<Kokkos::HostSpace>(
            src_view, dst_view, matrix, axis::solver::NormType::DstArea);

        nb::dict result;
        result["src_integral"] = report.src_integral;
        result["dst_integral"] = report.dst_integral;
        result["absolute_error"] = report.absolute_error;
        result["relative_error"] = report.relative_error;
        return result;
    }, "matrix"_a, "src"_a, "dst"_a,
       "Check conservation between source and destination fields");
}
