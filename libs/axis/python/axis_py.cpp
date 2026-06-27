// SPDX-License-Identifier: Apache-2.0
// AXIS Python bindings via nanobind
//
// Exposes the AXIS interpolation engine to Python for benchmarking and
// interactive use. Uses numpy arrays as the zero-copy buffer interface.
//
// Bindings cover:
//   - Mesh construction (regular lat-lon, named grids, from_descriptor)
//   - Weight generation with full RegridConfig dictionary
//   - Single-field apply and multi-field batch_apply
//   - Weight cache serialize/deserialize (to_bytes / from_bytes)
//   - Conservation check
//
// Links only against AXIS + nanobind — no AMIO/HALO dependency (Req 12.8).

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <Kokkos_Core.hpp>

#include <axis/types.hpp>
#include <axis/topology/projection_builder.hpp>
#include <axis/ingest/grid_descriptor.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/topology/structured_grid.hpp>
#include <axis/topology/mesh_factory.hpp>
#include <axis/topology/named_grid_registry.hpp>
#include <axis/solver/regrid_config.hpp>
#include <axis/solver/interpolation_matrix.hpp>
#include <axis/solver/weight_generator.hpp>
#include <axis/solver/weight_cache.hpp>
#include <axis/solver/apply.hpp>
#include <axis/solver/conservation.hpp>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

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

// ─── Helper: Build a projected mesh (e.g. Lambert Conformal) via PROJ ───────

HostMesh make_projected_mesh(std::size_t ni, std::size_t nj,
                             const std::string& proj_string,
                             nb::ndarray<nb::numpy, double, nb::ndim<1>> center_x,
                             nb::ndarray<nb::numpy, double, nb::ndim<1>> center_y) {
    ensure_kokkos();

    axis::ingest::ProjectedParams params;
    params.proj_string = proj_string;

    axis::ingest::BufferViews buffers;
    buffers.ni = ni;
    buffers.nj = nj;
    buffers.center_x = axis::field_view<const double, 1>(center_x.data(), center_x.shape(0));
    buffers.center_y = axis::field_view<const double, 1>(center_y.data(), center_y.shape(0));

    auto grid = axis::topology::ProjectionBuilder::build<Kokkos::HostSpace>(params, buffers);
    return grid.to_unstructured();
}

// ─── Helper: Build an unstructured UGRID mesh from arrays ───────────────────

HostMesh make_ugrid_mesh(nb::ndarray<nb::numpy, double, nb::ndim<2>> node_coords,
                         nb::ndarray<nb::numpy, axis::index_t, nb::ndim<1>> conn_offsets,
                         nb::ndarray<nb::numpy, axis::index_t, nb::ndim<1>> conn_indices) {
    ensure_kokkos();

    axis::ingest::GridDescriptor desc;
    desc.kind = axis::ingest::ConventionKind::UGRID;
    desc.ugrid.topology_dimension = 2;
    desc.ugrid.start_index = 0;

    desc.buffers.node_coords = axis::field_view<const double, 2>(
        node_coords.data(), node_coords.shape(0), node_coords.shape(1));
    desc.buffers.conn_offsets = axis::field_view<const axis::index_t, 1>(
        conn_offsets.data(), conn_offsets.shape(0));
    desc.buffers.conn_indices = axis::field_view<const axis::index_t, 1>(
        conn_indices.data(), conn_indices.shape(0));

    return axis::topology::MeshFactory::from_descriptor<Kokkos::HostSpace>(desc);
}

// ─── Helper: Parse RegridConfig from Python dict ─────────────────────────────

axis::solver::RegridConfig parse_regrid_config(const nb::dict& config) {
    axis::solver::RegridConfig cfg;

    if (config.contains("method")) {
        auto method_obj = config["method"];
        // Accept enum directly or string
        if (nb::isinstance<axis::solver::InterpolationMethod>(method_obj)) {
            cfg.method = nb::cast<axis::solver::InterpolationMethod>(method_obj);
        } else if (nb::isinstance<nb::str>(method_obj)) {
            std::string m = nb::cast<std::string>(method_obj);
            if (m == "bilinear" || m == "Bilinear")
                cfg.method = axis::solver::InterpolationMethod::Bilinear;
            else if (m == "nearest_neighbor" || m == "NearestNeighbor")
                cfg.method = axis::solver::InterpolationMethod::NearestNeighbor;
            else if (m == "bicubic" || m == "Bicubic")
                cfg.method = axis::solver::InterpolationMethod::Bicubic;
            else if (m == "patch" || m == "Patch")
                cfg.method = axis::solver::InterpolationMethod::Patch;
            else if (m == "conservative" || m == "Conservative1stOrder")
                cfg.method = axis::solver::InterpolationMethod::Conservative1stOrder;
            else if (m == "conservative2nd" || m == "Conservative2ndOrder")
                cfg.method = axis::solver::InterpolationMethod::Conservative2ndOrder;
            else
                throw std::invalid_argument("Unknown method: " + m);
        }
    }

    if (config.contains("norm_type")) {
        auto nt = config["norm_type"];
        if (nb::isinstance<axis::solver::NormType>(nt)) {
            cfg.norm_type = nb::cast<axis::solver::NormType>(nt);
        } else if (nb::isinstance<nb::str>(nt)) {
            std::string s = nb::cast<std::string>(nt);
            if (s == "dstarea" || s == "DstArea")
                cfg.norm_type = axis::solver::NormType::DstArea;
            else if (s == "fracarea" || s == "FracArea")
                cfg.norm_type = axis::solver::NormType::FracArea;
            else
                throw std::invalid_argument("Unknown norm_type: " + s);
        }
    }

    if (config.contains("unmapped")) {
        auto um = config["unmapped"];
        if (nb::isinstance<axis::solver::UnmappedAction>(um)) {
            cfg.unmapped = nb::cast<axis::solver::UnmappedAction>(um);
        } else if (nb::isinstance<nb::str>(um)) {
            std::string s = nb::cast<std::string>(um);
            if (s == "error" || s == "Error")
                cfg.unmapped = axis::solver::UnmappedAction::Error;
            else if (s == "ignore" || s == "Ignore")
                cfg.unmapped = axis::solver::UnmappedAction::Ignore;
            else
                throw std::invalid_argument("Unknown unmapped action: " + s);
        }
    }

    if (config.contains("line_type")) {
        auto lt = config["line_type"];
        if (nb::isinstance<axis::solver::LineType>(lt)) {
            cfg.line_type = nb::cast<axis::solver::LineType>(lt);
        } else if (nb::isinstance<nb::str>(lt)) {
            std::string s = nb::cast<std::string>(lt);
            if (s == "cartesian" || s == "Cartesian")
                cfg.line_type = axis::solver::LineType::Cartesian;
            else if (s == "great_circle" || s == "GreatCircle")
                cfg.line_type = axis::solver::LineType::GreatCircle;
            else
                throw std::invalid_argument("Unknown line_type: " + s);
        }
    }

    if (config.contains("use_limiter")) {
        cfg.use_limiter = nb::cast<bool>(config["use_limiter"]);
    }

    return cfg;
}

// ─── Module definition ───────────────────────────────────────────────────────

NB_MODULE(axis_py, m) {
    m.doc() = "AXIS Python bindings — spatial interpolation for Earth-system fields";

    // ─── InterpolationMethod enum ────────────────────────────────────────────
    nb::enum_<axis::solver::InterpolationMethod>(m, "Method")
        .value("Bilinear", axis::solver::InterpolationMethod::Bilinear)
        .value("NearestNeighbor", axis::solver::InterpolationMethod::NearestNeighbor)
        .value("Bicubic", axis::solver::InterpolationMethod::Bicubic)
        .value("Patch", axis::solver::InterpolationMethod::Patch)
        .value("Conservative", axis::solver::InterpolationMethod::Conservative1stOrder)
        .value("Conservative1stOrder", axis::solver::InterpolationMethod::Conservative1stOrder)
        .value("Conservative2ndOrder", axis::solver::InterpolationMethod::Conservative2ndOrder);

    // ─── NormType enum ───────────────────────────────────────────────────────
    nb::enum_<axis::solver::NormType>(m, "NormType")
        .value("DstArea", axis::solver::NormType::DstArea)
        .value("FracArea", axis::solver::NormType::FracArea);

    // ─── UnmappedAction enum ─────────────────────────────────────────────────
    nb::enum_<axis::solver::UnmappedAction>(m, "UnmappedAction")
        .value("Error", axis::solver::UnmappedAction::Error)
        .value("Ignore", axis::solver::UnmappedAction::Ignore);

    // ─── LineType enum ───────────────────────────────────────────────────────
    nb::enum_<axis::solver::LineType>(m, "LineType")
        .value("Cartesian", axis::solver::LineType::Cartesian)
        .value("GreatCircle", axis::solver::LineType::GreatCircle);

    // ─── Mesh wrapper class ──────────────────────────────────────────────────
    nb::class_<HostMesh>(m, "Mesh")
        .def_prop_ro("n_nodes", &HostMesh::n_nodes)
        .def_prop_ro("n_cells", &HostMesh::n_cells);

    // ─── InterpolationMatrix wrapper class ───────────────────────────────────
    nb::class_<HostMatrix>(m, "Matrix")
        .def_prop_ro("nnz", &HostMatrix::nnz)
        .def_prop_ro("n_src", &HostMatrix::n_src)
        .def_prop_ro("n_dst", &HostMatrix::n_dst)
        .def_prop_ro("is_csr", &HostMatrix::is_csr)

        // to_csr() — convert to CSR format for row-parallel apply
        .def("to_csr", &HostMatrix::to_csr,
             "Convert internal COO representation to CSR format")

        // to_bytes() — serialize to binary blob via WeightCache (Req 12.5)
        .def("to_bytes", [](const HostMatrix& matrix) -> nb::bytes {
            // Query required size
            const std::size_t size =
                axis::solver::WeightCache::serialize(matrix, nullptr, 0);

            // Allocate buffer and serialize
            std::vector<uint8_t> buf(size);
            axis::solver::WeightCache::serialize(matrix, buf.data(), size);

            // Return as Python bytes object
            return nb::bytes(reinterpret_cast<const char*>(buf.data()), size);
        }, "Serialize the matrix to a binary blob (WeightCache format)")

        // from_bytes() — static method to deserialize (Req 12.5)
        .def_static("from_bytes", [](nb::bytes data) -> HostMatrix {
            ensure_kokkos();
            const auto* ptr = reinterpret_cast<const void*>(data.c_str());
            const std::size_t size = data.size();
            return axis::solver::WeightCache::deserialize<Kokkos::HostSpace>(
                ptr, size);
        }, "data"_a,
           "Deserialize a binary blob (WeightCache format) into a Matrix");

    // ─── Mesh construction ───────────────────────────────────────────────────

    // Make a regular lat-lon mesh
    m.def("make_regular_mesh", &make_regular_mesh,
          "ni"_a, "nj"_a, "lon_start"_a, "lat_start"_a, "dlon"_a, "dlat"_a,
          "Create a regular lat-lon UnstructuredMesh");

    // Make a projected mesh (e.g. Lambert Conformal)
    m.def("make_projected_mesh", &make_projected_mesh,
          "ni"_a, "nj"_a, "proj_string"_a, "center_x"_a, "center_y"_a,
          "Create a projected UnstructuredMesh using PROJ");

    // Make an unstructured UGRID mesh
    m.def("make_ugrid_mesh", &make_ugrid_mesh,
          "node_coords"_a, "conn_offsets"_a, "conn_indices"_a,
          "Create an unstructured UGRID UnstructuredMesh");

    // Make a named grid (Req 12.4)
    m.def("make_named_mesh", [](const std::string& name) -> HostMesh {
        ensure_kokkos();
        return axis::topology::NamedGridRegistry::generate<Kokkos::HostSpace>(name);
    }, "name"_a, "Generate a named grid (e.g., 'O32', 'F64')");

    // ─── Weight generation (Req 12.2) ────────────────────────────────────────

    // Simple overload: method enum only
    m.def("generate_weights", [](const HostMesh& src, const HostMesh& dst,
                                  axis::solver::InterpolationMethod method) -> HostMatrix {
        ensure_kokkos();
        axis::solver::RegridConfig cfg;
        cfg.method = method;
        cfg.unmapped = axis::solver::UnmappedAction::Ignore;
        return axis::solver::WeightGenerator::generate<Kokkos::HostSpace>(src, dst, cfg);
    }, "src_mesh"_a, "dst_mesh"_a, "method"_a,
       "Generate interpolation weights between src and dst meshes");

    // Dict-based overload: full RegridConfig (Req 12.2)
    m.def("generate_weights", [](const HostMesh& src, const HostMesh& dst,
                                  const nb::dict& config) -> HostMatrix {
        ensure_kokkos();
        axis::solver::RegridConfig cfg = parse_regrid_config(config);
        return axis::solver::WeightGenerator::generate<Kokkos::HostSpace>(src, dst, cfg);
    }, "src_mesh"_a, "dst_mesh"_a, "config"_a,
       "Generate interpolation weights with a RegridConfig dictionary");

    // ─── Apply weights to a 1-D numpy array (Req 12.3) ──────────────────────

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

    // ─── Batch apply: multi-field SpMV (Req 12.6) ────────────────────────────

    m.def("batch_apply", [](const HostMatrix& matrix,
                             nb::ndarray<nb::numpy, double, nb::ndim<2>> src_arr)
          -> nb::ndarray<nb::numpy, double> {
        ensure_kokkos();

        const std::size_t n_src  = src_arr.shape(0);
        const std::size_t n_vars = src_arr.shape(1);
        const std::size_t n_dst  = matrix.n_dst();

        if (n_src != matrix.n_src()) {
            throw std::invalid_argument(
                "src array shape[0] (" + std::to_string(n_src) +
                ") != matrix.n_src (" + std::to_string(matrix.n_src()) + ")");
        }

        // Check memory layout (Req 12.7):
        // AXIS uses layout_left (Fortran-order / column-major).
        // numpy default is C-order (row-major / layout_right).
        // We accept both: if C-order, we interpret (n_src, n_vars) as row-major
        // and manually handle the stride difference by transposing into a
        // contiguous column-major buffer.
        const double* src_ptr = src_arr.data();

        // Detect if array is Fortran-contiguous (column-major)
        // stride(0) == sizeof(double) means column-major (fastest along axis 0)
        bool is_fortran_order = (src_arr.stride(0) == 1);

        // Allocate a column-major source buffer for AXIS
        std::vector<double> src_colmajor;
        if (!is_fortran_order) {
            // C-order (row-major): src_arr(i, v) is at ptr[i*n_vars + v]
            // We need column-major: buf(i, v) at buf[i + v*n_src]
            src_colmajor.resize(n_src * n_vars);
            for (std::size_t v = 0; v < n_vars; ++v) {
                for (std::size_t i = 0; i < n_src; ++i) {
                    src_colmajor[i + v * n_src] = src_ptr[i * n_vars + v];
                }
            }
            src_ptr = src_colmajor.data();
        }

        // Build field_view<const double, 2> over the column-major source
        axis::field_view<const double, 2> src_view(src_ptr, n_src, n_vars);

        // Allocate column-major destination buffer
        double* dst_buf = new double[n_dst * n_vars]();
        axis::field_view<double, 2> dst_view(dst_buf, n_dst, n_vars);

        // Execute batch apply
        axis::solver::batch_apply<Kokkos::HostSpace>(matrix, src_view, dst_view);

        // Convert back to C-order (row-major) for numpy return
        // numpy expects shape (n_dst, n_vars) in C-order: dst[j*n_vars + v]
        double* result = new double[n_dst * n_vars];
        for (std::size_t v = 0; v < n_vars; ++v) {
            for (std::size_t j = 0; j < n_dst; ++j) {
                result[j * n_vars + v] = dst_buf[j + v * n_dst];
            }
        }
        delete[] dst_buf;

        // Return as numpy array (n_dst, n_vars) in C-order
        nb::capsule owner(result, [](void* p) noexcept { delete[] static_cast<double*>(p); });
        std::size_t shape[2] = {n_dst, n_vars};
        return nb::ndarray<nb::numpy, double>(result, 2, shape, std::move(owner));
    }, "matrix"_a, "src"_a,
       "Apply interpolation matrix to multiple fields (cells × variables).\n"
       "Accepts both C-order and Fortran-order 2-D arrays.");

    // ─── Conservation check ──────────────────────────────────────────────────

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
