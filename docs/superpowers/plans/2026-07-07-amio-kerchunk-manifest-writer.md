# AMIO Kerchunk Manifest Writer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a unified Kerchunk ReferenceFileSystem manifest writer in AMIO to support high-performance cloud-native and local access for written NetCDF-4 and GRIB2 datasets.

**Architecture:** Create a header-only helper class `Kerchunk_Writer` inside `libs/AMIO/src/drivers/common/kerchunk_writer.hpp` that parses config, tracks written chunk byte ranges, performs post-close HDF5 introspection for NetCDF-4 variables (on rank 0), and serializes ReferenceFileSystem JSON using `nlohmann::json`.

**Tech Stack:** C++20, HDF5 C API, nlohmann/json, CMake, MPI.

## Global Constraints
- Target standard: C++20
- Public API headers under `include/` must remain untouched (C99-only). All changes must reside inside `src/` and be private to `AMIO_Core`.
- All ReferenceFileSystem output files must follow the v1 fsspec schema.
- Local mode must be assumed if `base_uri` is omitted.
- No changes to existing tests may break their current verification suites.

---

### Task 1: Scaffolding and Interface definition

**Files:**
- Create: `libs/AMIO/src/drivers/common/kerchunk_writer.hpp`
- Modify: `libs/AMIO/tests/integration/CMakeLists.txt`

**Interfaces:**
- Consumes: `conf::Config` from CONF.
- Produces: `amio::detail::Kerchunk_Writer` with static inline methods:
  - `static inline bool is_enabled(const conf::Config& config)`
  - `static inline void write_netcdf_manifest(const std::string& nc_file_path, const conf::Config& config, const std::vector<std::string>& written_variables)`
  - `static inline void write_grib2_manifest(const std::string& grib_file_path, const conf::Config& config, const std::vector<VirtualChunk>& written_chunks)`

- [ ] **Step 1: Write the header-only scaffolding**
  Create `libs/AMIO/src/drivers/common/kerchunk_writer.hpp` with basic configuration checking, type structures, and placeholders for NetCDF/GRIB2 writers.

```cpp
#ifndef AMIO_SRC_DRIVERS_COMMON_KERCHUNK_WRITER_HPP
#define AMIO_SRC_DRIVERS_COMMON_KERCHUNK_WRITER_HPP

#include <conf/config.hpp>
#include <string>
#include <vector>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <algorithm>
#include <nlohmann/json.hpp>

#ifdef AMIO_HAS_MPI
#include <mpi.h>
#endif

namespace amio::detail {

struct VirtualChunk {
    std::string variable_name;
    std::int64_t timestep = 0;
    std::vector<std::int64_t> chunk_coords;
    std::int64_t offset = 0;
    std::int64_t length = 0;
};

class Kerchunk_Writer {
public:
    static inline bool is_enabled(const conf::Config& config) {
        if (!config.has("kerchunk")) {
            return false;
        }
        if (config.has("kerchunk.enabled")) {
            return config.get_bool("kerchunk.enabled");
        }
        return false;
    }

    static inline void write_netcdf_manifest(
        const std::string& nc_file_path,
        const conf::Config& config,
        const std::vector<std::string>& written_variables)
    {
        (void)nc_file_path; (void)config; (void)written_variables;
    }

    static inline void write_grib2_manifest(
        const std::string& grib_file_path,
        const conf::Config& config,
        const std::vector<VirtualChunk>& written_chunks)
    {
        (void)grib_file_path; (void)config; (void)written_chunks;
    }
};

} // namespace amio::detail

#endif // AMIO_SRC_DRIVERS_COMMON_KERCHUNK_WRITER_HPP
```

- [ ] **Step 2: Verify the header compiles cleanly**
  Configure the build to ensure there are no syntax errors in the new header.
  Run: `cmake -S libs/AMIO -B libs/AMIO/build-local -DAMIO_BUILD_TESTING=ON`
  Expected: Success.

- [ ] **Step 3: Commit**
```bash
git add libs/AMIO/src/drivers/common/kerchunk_writer.hpp
git commit -m "feat: scaffold Kerchunk_Writer class"
```

---

### Task 2: GRIB2 Manifest Integration and Testing

**Files:**
- Modify: `libs/AMIO/src/drivers/grib2/grib2_driver.hpp`
- Modify: `libs/AMIO/src/drivers/grib2/grib2_driver.cpp`
- Create: `libs/AMIO/tests/integration/test_kerchunk_grib2.cpp`
- Modify: `libs/AMIO/tests/integration/CMakeLists.txt`

**Interfaces:**
- Consumes: `amio::detail::Kerchunk_Writer`
- Produces: Completed ReferenceFileSystem JSON manifests for written GRIB2 files.

- [ ] **Step 1: Track chunks in GRIB2_Driver**
  Add private tracking vectors in `libs/AMIO/src/drivers/grib2/grib2_driver.hpp`:
```cpp
    std::vector<VirtualChunk> written_chunks_;
    std::vector<std::string> written_variables_;
    conf::Config config_copy_ = conf::Config::from_string("path: dummy"); // copy to use during close
```
  In `GRIB2_Driver::open_write`, store the config copy:
```cpp
    config_copy_ = conf::Config::from_string(config.serialize()); // or similar
```
  In `GRIB2_Driver::write`, record chunk byte offsets:
```cpp
    if (out_file_ != nullptr) {
        std::int64_t offset = std::ftell(out_file_);
        // ... fwrite msglen ...
        if (Kerchunk_Writer::is_enabled(config_)) {
            VirtualChunk chunk;
            chunk.variable_name = meta.name;
            chunk.timestep = meta.timestep;
            chunk.chunk_coords = { meta.timestep };
            chunk.offset = offset;
            chunk.length = static_cast<std::int64_t>(msglen);
            written_chunks_.push_back(chunk);
            if (std::find(written_variables_.begin(), written_variables_.end(), meta.name) == written_variables_.end()) {
                written_variables_.push_back(meta.name);
            }
        }
    }
```
  In `GRIB2_Driver::close`, trigger manifest writing:
```cpp
    if (Kerchunk_Writer::is_enabled(config_copy_) && !written_chunks_.empty()) {
        Kerchunk_Writer::write_grib2_manifest(file_path_, config_copy_, written_chunks_);
    }
    written_chunks_.clear();
    written_variables_.clear();
```

- [ ] **Step 2: Implement write_grib2_manifest and JSON formatting in kerchunk_writer.hpp**
  Implement the standard Zarr Refs builder for GRIB2 sequential messages:
```cpp
    static inline void write_manifest_json(
        const std::string& output_json_path,
        const std::string& target_data_uri,
        const std::vector<VirtualChunk>& chunks,
        const std::vector<std::string>& variables,
        const conf::Config& config)
    {
        nlohmann::json manifest = nlohmann::json::object();
        manifest["version"] = 1;
        nlohmann::json refs = nlohmann::json::object();
        refs[".zgroup"] = "{\"zarr_format\": 2}";

        for (const auto& var : variables) {
            // Find total count for shape
            std::int64_t max_ts = 0;
            for (const auto& chunk : chunks) {
                if (chunk.variable_name == var) {
                    max_ts = std::max(max_ts, chunk.timestep + 1);
                }
            }

            nlohmann::json zarray;
            zarray["zarr_format"] = 2;
            zarray["shape"] = nlohmann::json::array({ max_ts, 181, 360 }); // default representative GFS size, can be customized or general
            zarray["chunks"] = nlohmann::json::array({ 1, 181, 360 });
            zarray["dtype"] = "<f4";
            zarray["order"] = "C";
            zarray["fill_value"] = nullptr;
            zarray["compressor"] = nullptr;
            zarray["filters"] = nullptr;
            refs[var + "/.zarray"] = zarray.dump();

            nlohmann::json zattrs;
            zattrs["_ARRAY_DIMENSIONS"] = nlohmann::json::array({ "time", "lat", "lon" });
            refs[var + "/.zattrs"] = zattrs.dump();
        }

        for (const auto& chunk : chunks) {
            std::string coord_str = "";
            for (size_t i = 0; i < chunk.chunk_coords.size(); ++i) {
                if (i > 0) coord_str += ".";
                coord_str += std::to_string(chunk.chunk_coords[i]);
            }
            // Kerchunk standard: pad to match dimension count e.g. "time.lat.lon" -> "t.0.0"
            std::string coord_suffix = coord_str + ".0.0";
            std::string key = chunk.variable_name + "/" + coord_suffix;
            refs[key] = nlohmann::json::array({ target_data_uri, chunk.offset, chunk.length });
        }

        manifest["refs"] = refs;
        std::ofstream out(output_json_path);
        if (out.is_open()) {
            out << manifest.dump(4);
            out.close();
        } else {
            throw std::runtime_error("Kerchunk_Writer: Failed to write JSON to " + output_json_path);
        }
    }

    static inline void write_grib2_manifest(
        const std::string& grib_file_path,
        const conf::Config& config,
        const std::vector<VirtualChunk>& written_chunks)
    {
        std::string base_uri = "";
        if (config.has("kerchunk.base_uri")) {
            base_uri = config.get_string("kerchunk.base_uri");
        }
        std::string target_uri;
        if (base_uri.empty()) {
            target_uri = std::filesystem::absolute(grib_file_path).string();
        } else {
            std::string file_name = std::filesystem::path(grib_file_path).filename().string();
            target_uri = (base_uri.back() == '/') ? base_uri + file_name : base_uri + "/" + file_name;
        }

        // Collect distinct variable names
        std::vector<std::string> variables;
        for (const auto& chunk : written_chunks) {
            if (std::find(variables.begin(), variables.end(), chunk.variable_name) == variables.end()) {
                variables.push_back(chunk.variable_name);
            }
        }

        std::string output_json_path = config.get_string("kerchunk.output_path");
        write_manifest_json(output_json_path, target_uri, written_chunks, variables, config);
    }
```

- [ ] **Step 3: Create integration test test_kerchunk_grib2.cpp**
  Create `libs/AMIO/tests/integration/test_kerchunk_grib2.cpp` verifying:
  1. The dataset compiles and writes GRIB2.
  2. The generated ReferenceFileSystem JSON exists.
  3. The JSON references correct local paths (local mode) and cloud base URIs (when base_uri is specified).
  4. The chunk coordinates, offset, and length match.

- [ ] **Step 4: Hook integration test into integration/CMakeLists.txt**
  Register the executable in `libs/AMIO/tests/integration/CMakeLists.txt`:
```cmake
if(g2c_FOUND AND eckit_FOUND)
    add_executable(test_kerchunk_grib2 test_kerchunk_grib2.cpp ${CMAKE_SOURCE_DIR}/src/drivers/grib2/grib2_driver.cpp
                                       ${CMAKE_SOURCE_DIR}/src/factory/backend_factory.cpp ${CMAKE_SOURCE_DIR}/src/staging/staging_pool.cpp)
    target_include_directories(test_kerchunk_grib2 PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_compile_definitions(test_kerchunk_grib2 PRIVATE AMIO_HAS_G2C=1 AMIO_HAS_ECKIT=1)
    target_link_libraries(test_kerchunk_grib2 PRIVATE AMIO::public_headers eckit g2c::g2c)
    add_test(NAME integration.kerchunk_grib2 COMMAND test_kerchunk_grib2)
endif()
```

- [ ] **Step 5: Run GRIB2 integration test to verify it passes**
  Build and execute the new test:
  `cmake -S libs/AMIO -B libs/AMIO/build-local -DAMIO_BUILD_TESTING=ON && cmake --build libs/AMIO/build-local --target test_kerchunk_grib2`
  Run: `libs/AMIO/build-local/tests/integration/test_kerchunk_grib2`
  Expected: PASS

- [ ] **Step 6: Commit**
```bash
git add libs/AMIO/src/drivers/grib2/grib2_driver.hpp libs/AMIO/src/drivers/grib2/grib2_driver.cpp libs/AMIO/tests/integration/test_kerchunk_grib2.cpp libs/AMIO/tests/integration/CMakeLists.txt
git commit -m "feat: add GRIB2 Kerchunk manifest writer and integration test"
```

---

### Task 3: NetCDF-4 Introspection and Manifest Integration

**Files:**
- Modify: `libs/AMIO/src/drivers/netcdf/netcdf_driver.hpp`
- Modify: `libs/AMIO/src/drivers/netcdf/netcdf_driver.cpp`
- Modify: `libs/AMIO/src/drivers/common/kerchunk_writer.hpp`

**Interfaces:**
- Consumes: `amio::detail::Kerchunk_Writer`, HDF5 C API.
- Produces: HDF5 chunk introspection of written NetCDF-4 files on close (on rank 0 only).

- [ ] **Step 1: Track written variables in NetCDF_Driver**
  Add member variables in `libs/AMIO/src/drivers/netcdf/netcdf_driver.hpp`:
```cpp
    std::vector<std::string> written_variables_;
    conf::Config config_copy_ = conf::Config::from_string("path: dummy");
```
  In `NetCDF_Driver::open_write`, store `config_copy_`.
  In `NetCDF_Driver::write`, record variable name:
```cpp
    if (std::find(written_variables_.begin(), written_variables_.end(), meta.name) == written_variables_.end()) {
        written_variables_.push_back(meta.name);
    }
```
  In `NetCDF_Driver::close`, right before closing or after closing, check rank and call manifest writer:
```cpp
    #ifdef AMIO_HAS_NETCDF
    int status = nc_close(ncid_);
    nc_check(status, "nc_close");
    ncid_ = -1;

    int mpi_rank = 0;
    #ifdef AMIO_HAS_MPI
    if (comm_ != MPI_COMM_NULL) {
        MPI_Comm_rank(comm_, &mpi_rank);
    }
    #endif

    // Only rank 0 handles introspection and writes the JSON manifest
    if (mpi_rank == 0 && Kerchunk_Writer::is_enabled(config_copy_) && !written_variables_.empty()) {
        Kerchunk_Writer::write_netcdf_manifest(file_path_, config_copy_, written_variables_);
    }
    #endif
    written_variables_.clear();
```

- [ ] **Step 2: Implement write_netcdf_manifest using HDF5 C APIs**
  In `libs/AMIO/src/drivers/common/kerchunk_writer.hpp`, conditionally compile the HDF5 part:
```cpp
    static inline void write_netcdf_manifest(
        const std::string& nc_file_path,
        const conf::Config& config,
        const std::vector<std::string>& written_variables)
    {
#ifdef AMIO_HAS_NETCDF
        // Include hdf5.h locally within the function to prevent leaks
        #include <hdf5.h>

        std::vector<VirtualChunk> written_chunks;

        hid_t file_id = H5Fopen(nc_file_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file_id < 0) {
            throw std::runtime_error("Kerchunk_Writer: Failed to open HDF5 file: " + nc_file_path);
        }

        for (const auto& var_name : written_variables) {
            hid_t dataset_id = H5Dopen2(file_id, var_name.c_str(), H5P_DEFAULT);
            if (dataset_id < 0) {
                continue; // Variable wasn't actually committed/created
            }

            hid_t plist_id = H5Dget_create_plist(dataset_id);
            H5D_layout_t layout = H5Pget_layout(plist_id);

            if (layout == H5D_CONTIGUOUS) {
                haddr_t offset = H5Dget_offset(dataset_id);
                hsize_t size = H5Dget_storage_size(dataset_id);
                if (offset != HADDR_UNDEF) {
                    VirtualChunk chunk;
                    chunk.variable_name = var_name;
                    chunk.timestep = 0;
                    chunk.chunk_coords = {0, 0, 0}; // Rank 3 padding for consistency
                    chunk.offset = static_cast<std::int64_t>(offset);
                    chunk.length = static_cast<std::int64_t>(size);
                    written_chunks.push_back(chunk);
                }
            } else if (layout == H5D_CHUNKED) {
                hid_t dataspace_id = H5Dget_space(dataset_id);
                int rank = H5Sget_simple_extent_ndims(dataspace_id);
                std::vector<hsize_t> chunk_dims(rank);
                H5Pget_chunk(plist_id, rank, chunk_dims.data());

                hsize_t nchunks = 0;
                H5Dget_num_chunks(dataset_id, dataspace_id, &nchunks);

                for (hsize_t idx = 0; idx < nchunks; ++idx) {
                    std::vector<hsize_t> offset(rank);
                    unsigned filter_mask = 0;
                    haddr_t chunk_addr = 0;
                    hsize_t chunk_size = 0;

                    H5Dget_chunk_info(dataset_id, dataspace_id, idx, offset.data(),
                                      &filter_mask, &chunk_addr, &chunk_size);

                    std::vector<std::int64_t> chunk_coords(rank);
                    for (int r = 0; r < rank; ++r) {
                        chunk_coords[r] = static_cast<std::int64_t>(offset[r] / chunk_dims[r]);
                    }

                    VirtualChunk chunk;
                    chunk.variable_name = var_name;
                    chunk.chunk_coords = chunk_coords;
                    chunk.offset = static_cast<std::int64_t>(chunk_addr);
                    chunk.length = static_cast<std::int64_t>(chunk_size);
                    written_chunks.push_back(chunk);
                }
                H5Sclose(dataspace_id);
            }

            H5Pclose(plist_id);
            H5Dclose(dataset_id);
        }
        H5Fclose(file_id);

        // Standard Target URI formatting
        std::string base_uri = "";
        if (config.has("kerchunk.base_uri")) {
            base_uri = config.get_string("kerchunk.base_uri");
        }
        std::string target_uri;
        if (base_uri.empty()) {
            target_uri = std::filesystem::absolute(nc_file_path).string();
        } else {
            std::string file_name = std::filesystem::path(nc_file_path).filename().string();
            target_uri = (base_uri.back() == '/') ? base_uri + file_name : base_uri + "/" + file_name;
        }

        std::string output_json_path = config.get_string("kerchunk.output_path");
        write_manifest_json(output_json_path, target_uri, written_chunks, written_variables, config);
#else
        throw std::runtime_error("Kerchunk_Writer::write_netcdf_manifest: AMIO built without NetCDF/HDF5 support");
#endif
    }
```

- [ ] **Step 3: Run existing unit and integration tests to verify compile-and-link**
  Configure and compile.
  Run: `cmake -S libs/AMIO -B libs/AMIO/build-local -DAMIO_BUILD_TESTING=ON && cmake --build libs/AMIO/build-local`
  Expected: Successful compile and link of all tests.

- [ ] **Step 4: Commit**
```bash
git add libs/AMIO/src/drivers/netcdf/netcdf_driver.hpp libs/AMIO/src/drivers/netcdf/netcdf_driver.cpp libs/AMIO/src/drivers/common/kerchunk_writer.hpp
git commit -m "feat: implement HDF5 chunk introspection for NetCDF"
```

---

### Task 4: NetCDF-4 Manifest Testing and Verification

**Files:**
- Create: `libs/AMIO/tests/integration/test_kerchunk_netcdf4.cpp`
- Modify: `libs/AMIO/tests/integration/CMakeLists.txt`

**Interfaces:**
- Consumes: NetCDF_Driver, Kerchunk_Writer, MPI.
- Produces: Correct ReferenceFileSystem JSON verifying chunk coordinate mapping, contiguous vs chunked layouts, and target URI mapping.

- [ ] **Step 1: Create test_kerchunk_netcdf4.cpp**
  Create `libs/AMIO/tests/integration/test_kerchunk_netcdf4.cpp` to write both contiguous and chunked variables in NetCDF-4 and verify that the manifest JSON matches layouts, coordinates, offsets, and target paths correctly.

- [ ] **Step 2: Register test in CMakeLists.txt**
  Add the executable in `libs/AMIO/tests/integration/CMakeLists.txt`:
```cmake
if(netCDF_FOUND AND MPI_C_FOUND AND eckit_FOUND)
    add_executable(test_kerchunk_netcdf4 test_kerchunk_netcdf4.cpp ${CMAKE_SOURCE_DIR}/src/drivers/netcdf/netcdf_driver.cpp
                                        ${CMAKE_SOURCE_DIR}/src/drivers/common/var_attributes.cpp ${CMAKE_SOURCE_DIR}/src/factory/backend_factory.cpp)
    target_include_directories(test_kerchunk_netcdf4 PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_compile_definitions(test_kerchunk_netcdf4 PRIVATE AMIO_HAS_NETCDF=1 AMIO_HAS_ECKIT=1)
    target_link_libraries(test_kerchunk_netcdf4 PRIVATE AMIO::public_headers eckit)

    if(AMIO_NETCDF_PAR_LIB)
        get_target_property(_nc_inc netCDF::netcdf INTERFACE_INCLUDE_DIRECTORIES)
        if(_nc_inc)
            target_include_directories(test_kerchunk_netcdf4 PRIVATE ${_nc_inc})
        endif()
        target_link_libraries(test_kerchunk_netcdf4 PRIVATE ${AMIO_NETCDF_PAR_LIB})
    else()
        target_link_libraries(test_kerchunk_netcdf4 PRIVATE netCDF::netcdf)
    endif()
    if(AMIO_NETCDF_MPI_INCLUDE_DIR)
        target_include_directories(test_kerchunk_netcdf4 PRIVATE ${AMIO_NETCDF_MPI_INCLUDE_DIR})
    endif()
    target_link_libraries(test_kerchunk_netcdf4 PRIVATE MPI::MPI_C)
    if(MPI_CXX_FOUND)
        target_link_libraries(test_kerchunk_netcdf4 PRIVATE MPI::MPI_CXX)
    endif()
    add_test(NAME integration.kerchunk_netcdf4 COMMAND test_kerchunk_netcdf4)
endif()
```

- [ ] **Step 3: Run the new test and verify it passes**
  Build and execute:
  Run: `cmake -S libs/AMIO -B libs/AMIO/build-local -DAMIO_BUILD_TESTING=ON && cmake --build libs/AMIO/build-local --target test_kerchunk_netcdf4`
  Run: `libs/AMIO/build-local/tests/integration/test_kerchunk_netcdf4`
  Expected: PASS

- [ ] **Step 4: Run full project test suite**
  Run: `ctest --test-dir libs/AMIO/build-local`
  Expected: All tests pass.

- [ ] **Step 5: Commit**
```bash
git add libs/AMIO/tests/integration/test_kerchunk_netcdf4.cpp libs/AMIO/tests/integration/CMakeLists.txt
git commit -m "feat: add NetCDF Kerchunk integration test and verify all tests"
```
