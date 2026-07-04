// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#include <netcdf.h>

#include <axis/io/esmf_weight_io.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace axis::io {

// ─────────────────────────────────────────────────────────────────────────────
// NetCdf_Handle — Move-only RAII wrapper for NetCDF file IDs (Local to I/O target)
// ─────────────────────────────────────────────────────────────────────────────

class NetCdf_Handle {
   public:
    explicit NetCdf_Handle(int ncid) noexcept : ncid_(ncid) {}

    ~NetCdf_Handle() noexcept {
        if (ncid_ >= 0) {
            ::nc_close(ncid_);
        }
    }

    NetCdf_Handle(NetCdf_Handle &&other) noexcept : ncid_(other.ncid_) {
        other.ncid_ = -1;
    }

    NetCdf_Handle &operator=(NetCdf_Handle &&other) noexcept {
        if (this != &other) {
            if (ncid_ >= 0) ::nc_close(ncid_);
            ncid_ = other.ncid_;
            other.ncid_ = -1;
        }
        return *this;
    }

    NetCdf_Handle(const NetCdf_Handle &) = delete;
    NetCdf_Handle &operator=(const NetCdf_Handle &) = delete;

    [[nodiscard]] int get() const noexcept {
        return ncid_;
    }
    [[nodiscard]] explicit operator bool() const noexcept {
        return ncid_ >= 0;
    }

   private:
    int ncid_ = -1;
};

// ─────────────────────────────────────────────────────────────────────────────
// Error handling macros for NetCDF C-API
// ─────────────────────────────────────────────────────────────────────────────

#define AXIS_NC_CHECK(err)                                                                 \
    do {                                                                                   \
        int e = (err);                                                                     \
        if (e != NC_NOERR) {                                                               \
            throw std::runtime_error("AXIS NetCDF Error: " + std::string(nc_strerror(e))); \
        }                                                                                  \
    } while (0)

template <typename MemorySpace>
void EsmfWeightIO<MemorySpace>::write_esmf(const std::string &filepath, const solver::InterpolationMatrix<MemorySpace> &matrix) {
    const std::size_t n_src = matrix.n_src();
    const std::size_t n_dst = matrix.n_dst();
    const std::size_t nnz = matrix.nnz();

    auto rows = matrix.factor_row_view();
    auto cols = matrix.factor_col_view();
    auto vals = matrix.factor_list_view();

    auto h_rows = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rows);
    auto h_cols = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cols);
    auto h_vals = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vals);

    // Translate 0-based indices to 1-based Fortran/ESMF indexes
    std::vector<int> col_idx(nnz);
    std::vector<int> row_idx(nnz);
    std::vector<double> weights(nnz);

    for (std::size_t k = 0; k < nnz; ++k) {
        col_idx[k] = static_cast<int>(h_cols(k) + 1);
        row_idx[k] = static_cast<int>(h_rows(k) + 1);
        weights[k] = h_vals(k);
    }

    int ncid_raw;
    AXIS_NC_CHECK(nc_create(filepath.c_str(), NC_CLOBBER | NC_NETCDF4, &ncid_raw));
    NetCdf_Handle handle(ncid_raw);
    int ncid = handle.get();

    // Define dimensions and variables
    int dim_id;
    AXIS_NC_CHECK(nc_def_dim(ncid, "num_wgts", nnz, &dim_id));

    int var_s, var_col, var_row;
    int dim_ids[1] = {dim_id};
    AXIS_NC_CHECK(nc_def_var(ncid, "S", NC_DOUBLE, 1, dim_ids, &var_s));
    AXIS_NC_CHECK(nc_def_var(ncid, "col_idx", NC_INT, 1, dim_ids, &var_col));
    AXIS_NC_CHECK(nc_def_var(ncid, "row_idx", NC_INT, 1, dim_ids, &var_row));

    // Global attributes indicating source (n_a) and destination (n_b) grid cell counts
    int n_a = static_cast<int>(n_src);
    int n_b = static_cast<int>(n_dst);
    AXIS_NC_CHECK(nc_put_att_int(ncid, NC_GLOBAL, "n_a", NC_INT, 1, &n_a));
    AXIS_NC_CHECK(nc_put_att_int(ncid, NC_GLOBAL, "n_b", NC_INT, 1, &n_b));

    AXIS_NC_CHECK(nc_enddef(ncid));

    // Write weights data
    AXIS_NC_CHECK(nc_put_var_double(ncid, var_s, weights.data()));
    AXIS_NC_CHECK(nc_put_var_int(ncid, var_col, col_idx.data()));
    AXIS_NC_CHECK(nc_put_var_int(ncid, var_row, row_idx.data()));

    // The RAII handle destructor automatically and safely closes the file.
}

template <typename MemorySpace>
solver::InterpolationMatrix<MemorySpace> EsmfWeightIO<MemorySpace>::read_esmf(const std::string &filepath) {
    int ncid_raw;
    AXIS_NC_CHECK(nc_open(filepath.c_str(), NC_NOWRITE, &ncid_raw));
    NetCdf_Handle handle(ncid_raw);
    int ncid = handle.get();

    // Read global dimensions
    int n_src_val, n_dst_val;
    AXIS_NC_CHECK(nc_get_att_int(ncid, NC_GLOBAL, "n_a", &n_src_val));
    AXIS_NC_CHECK(nc_get_att_int(ncid, NC_GLOBAL, "n_b", &n_dst_val));

    const std::size_t n_src = static_cast<std::size_t>(n_src_val);
    const std::size_t n_dst = static_cast<std::size_t>(n_dst_val);

    // Look up variable IDs
    int var_s, var_col, var_row;
    AXIS_NC_CHECK(nc_inq_varid(ncid, "S", &var_s));
    AXIS_NC_CHECK(nc_inq_varid(ncid, "col_idx", &var_col));
    AXIS_NC_CHECK(nc_inq_varid(ncid, "row_idx", &var_row));

    // Determine weights count (nnz)
    int ndims;
    int dimids[1];
    AXIS_NC_CHECK(nc_inq_var(ncid, var_s, nullptr, nullptr, &ndims, dimids, nullptr));

    std::size_t nnz;
    AXIS_NC_CHECK(nc_inq_dimlen(ncid, dimids[0], &nnz));

    std::vector<double> weights(nnz);
    std::vector<int> col_idx(nnz);
    std::vector<int> row_idx(nnz);

    // Read raw data
    AXIS_NC_CHECK(nc_get_var_double(ncid, var_s, weights.data()));
    AXIS_NC_CHECK(nc_get_var_int(ncid, var_col, col_idx.data()));
    AXIS_NC_CHECK(nc_get_var_int(ncid, var_row, row_idx.data()));

    // Translate 1-based back to 0-based
    Kokkos::View<index_t *, Kokkos::HostSpace> host_rows("h_rows", nnz);
    Kokkos::View<index_t *, Kokkos::HostSpace> host_cols("h_cols", nnz);
    Kokkos::View<double *, Kokkos::HostSpace> host_vals("h_vals", nnz);

    for (std::size_t k = 0; k < nnz; ++k) {
        host_rows(k) = static_cast<index_t>(row_idx[k] - 1);
        host_cols(k) = static_cast<index_t>(col_idx[k] - 1);
        host_vals(k) = weights[k];
    }

    auto dev_rows = Kokkos::create_mirror_view_and_copy(MemorySpace(), host_rows);
    auto dev_cols = Kokkos::create_mirror_view_and_copy(MemorySpace(), host_cols);
    auto dev_vals = Kokkos::create_mirror_view_and_copy(MemorySpace(), host_vals);

    return solver::InterpolationMatrix<MemorySpace>(
        dev_vals, dev_rows, dev_cols, Kokkos::View<double *, MemorySpace>("frac_a", n_src), Kokkos::View<double *, MemorySpace>("frac_b", n_dst),
        Kokkos::View<double *, MemorySpace>("area_a", n_src), Kokkos::View<double *, MemorySpace>("area_b", n_dst), n_src, n_dst);
}

template class EsmfWeightIO<Kokkos::HostSpace>;

#ifdef KOKKOS_ENABLE_CUDA
template class EsmfWeightIO<Kokkos::CudaSpace>;
#endif

#ifdef KOKKOS_ENABLE_HIP
template class EsmfWeightIO<Kokkos::HIPSpace>;
#endif

}  // namespace axis::io
