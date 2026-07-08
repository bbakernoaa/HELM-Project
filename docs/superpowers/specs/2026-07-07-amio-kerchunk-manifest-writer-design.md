# Design Specification: AMIO Kerchunk Manifest Writer

## 1. Overview
AMIO (Asynchronous Multidimensional I/O) provides high-performance, asynchronous reads and writes for Earth system models. This design adds the ability to write **Kerchunk manifests** (ReferenceFileSystem JSON representation) during the dataset close operation. 

This enables cloud-native, high-performance, and parallel read access to legacy file formats (NetCDF-4 and GRIB2) using a Zarr-like virtual metadata structure without needing to rewrite legacy datasets to Zarr format.

---

## 2. Configuration Schema
Kerchunk manifest writing is enabled via a `kerchunk` block in the dataset's YAML configuration.

```yaml
# Inside the dataset YAML configuration (e.g. manifest.yaml)
backend: netcdf4  # or grib2
path: /scratch/output/data.nc

kerchunk:
  enabled: true
  output_path: /scratch/output/data.json
  # base_uri: "s3://noaa-nws-pangeo"  # Optional. If omitted, uses local absolute data file path.
```

### Configuration Semantics:
- **`enabled`** (boolean, required): Toggles Kerchunk manifest generation.
- **`output_path`** (string, required): The target filesystem path where the ReferenceFileSystem JSON is written.
- **`base_uri`** (string, optional): The base URI/bucket path (e.g. `s3://bucket/prefix`).
  - **Local Mode (Omission):** If `base_uri` is omitted, AMIO automatically resolves the absolute path of the generated data file (e.g. `/scratch/output/data.nc`) and embeds that. This allows seamless local usage without cloud or external server dependencies.
  - **Cloud Mode:** If `base_uri` is specified, AMIO prepends the base URI to the data file's basename to construct the target URI.

---

## 3. Class Interface
The manifest generation logic is encapsulated in a unified, static utility class `Kerchunk_Writer` inside the private namespace `amio::detail`.

- Header file: `libs/AMIO/src/drivers/common/kerchunk_writer.hpp`
- Source file: `libs/AMIO/src/drivers/common/kerchunk_writer.cpp`

### C++ Header Definition:
```cpp
#ifndef AMIO_SRC_DRIVERS_COMMON_KERCHUNK_WRITER_HPP
#define AMIO_SRC_DRIVERS_COMMON_KERCHUNK_WRITER_HPP

#include <conf/config.hpp>
#include <string>
#include <vector>
#include <cstdint>

namespace amio::detail {

// Represents a virtual chunk or record byte-range segment.
struct VirtualChunk {
    std::string variable_name;
    std::int64_t timestep = 0;
    std::vector<std::int64_t> chunk_coords; // e.g. {t, y, x} or {t}
    std::int64_t offset = 0;               // Byte offset in the data file
    std::int64_t length = 0;               // Byte length of the segment
};

class Kerchunk_Writer {
public:
    // Helper to check if Kerchunk generation is configured and enabled.
    static bool is_enabled(const conf::Config& config);

    // Introspects HDF5 chunk allocations in a closed NetCDF-4 file and writes the JSON.
    static void write_netcdf_manifest(
        const std::string& nc_file_path,
        const conf::Config& config,
        const std::vector<std::string>& written_variables);

    // Formats and writes the JSON using explicit in-memory GRIB2 record locations.
    static void write_grib2_manifest(
        const std::string& grib_file_path,
        const conf::Config& config,
        const std::vector<VirtualChunk>& written_chunks);

private:
    // Shared method to build and serialize the ReferenceFileSystem JSON structure.
    static void write_manifest_json(
        const std::string& output_json_path,
        const std::string& target_data_uri,
        const std::vector<VirtualChunk>& chunks,
        const std::vector<std::string>& variables,
        const conf::Config& config);
};

} // namespace amio::detail

#endif // AMIO_SRC_DRIVERS_COMMON_KERCHUNK_WRITER_HPP
```

---

## 4. NetCDF-4 (HDF5) Introspection
For NetCDF-4, because chunks are written asynchronously and possibly in parallel across multiple MPI ranks, chunk layouts are retrieved by introspecting the file right after it is closed.

On dataset close (after `nc_close()` has completed and flushed all bytes to disk):
1. **Rank 0 only** (to avoid write collisions) opens the generated NetCDF-4 file using the read-only HDF5 C API:
   ```cpp
   hid_t file_id = H5Fopen(nc_file_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
   ```
2. For each variable in the `written_variables` list, it opens the HDF5 dataset:
   ```cpp
   hid_t dataset_id = H5Dopen2(file_id, var_name.c_str(), H5P_DEFAULT);
   ```
3. It checks the dataset layout property list:
   - **Contiguous Layout:** Retrieve offset and length directly:
     ```cpp
     haddr_t offset = H5Dget_offset(dataset_id);
     hsize_t size = H5Dget_storage_size(dataset_id);
     ```
   - **Chunked Layout:** Iterate over chunks using the HDF5 chunk C API:
     ```cpp
     hsize_t nchunks = 0;
     H5Dget_num_chunks(dataset_id, dataspace_id, &nchunks);
     for (hsize_t i = 0; i < nchunks; ++i) {
         H5Dget_chunk_info(dataset_id, dataspace_id, i, offset.data(), &filter_mask, &chunk_addr, &chunk_size);
         // Virtual chunk coord = offset / chunk_dims
     }
     ```
4. All resources (`H5Dclose`, `H5Fclose`, etc.) are safely cleaned up.

---

## 5. GRIB2 Offset Tracking
Because the GRIB2 backend writes messages sequentially on a single thread and has no complex parallel chunk allocation mechanics, we track byte positions dynamically during write.

1. **`GRIB2_Driver`** maintains:
   ```cpp
   std::vector<VirtualChunk> written_chunks_;
   std::vector<std::string> written_variables_;
   ```
2. Inside **`GRIB2_Driver::write()`**, right before committing bytes, we query the current file position:
   ```cpp
   std::int64_t offset = std::ftell(out_file_);
   ```
3. Following a successful `std::fwrite()` of `msglen` bytes, we record a `VirtualChunk`:
   ```cpp
   VirtualChunk chunk;
   chunk.variable_name = meta.name;
   chunk.timestep = meta.timestep;
   chunk.chunk_coords = { meta.timestep };
   chunk.offset = offset;
   chunk.length = static_cast<std::int64_t>(msglen);
   written_chunks_.push_back(chunk);
   ```
4. In **`GRIB2_Driver::close()`**, after closing the file handle, if Kerchunk is enabled, the GRIB2 manifest is written:
   ```cpp
   Kerchunk_Writer::write_grib2_manifest(file_path_, config_, written_chunks_);
   ```

---

## 6. Formatting the ReferenceFileSystem JSON
The JSON is constructed using the header-only `nlohmann::json` library already included in AMIO's dependencies.

### Target URI Construction:
```cpp
std::string target_uri;
if (base_uri.empty()) {
    target_uri = std::filesystem::absolute(file_path).string();
} else {
    std::string file_name = std::filesystem::path(file_path).filename().string();
    target_uri = (base_uri.back() == '/') ? base_uri + file_name : base_uri + "/" + file_name;
}
```

### JSON refs Layout:
```json
{
  "version": 1,
  "refs": {
    ".zgroup": "{\n  \"zarr_format\": 2\n}",
    "temperature/.zarray": "{\n  \"chunks\": [1, 181, 360],\n  \"compressor\": null,\n  \"dtype\": \"<f4\",\n  \"fill_value\": null,\n  \"filters\": null,\n  \"order\": \"C\",\n  \"shape\": [6, 181, 360],\n  \"zarr_format\": 2\n}",
    "temperature/.zattrs": "{\n  \"_ARRAY_DIMENSIONS\": [\"time\", \"lat\", \"lon\"]\n}",
    "temperature/0.0.0": ["/scratch/output/data.nc", 1024, 25440],
    "temperature/1.0.0": ["/scratch/output/data.nc", 26464, 25440]
  }
}
```

---

## 7. Testing & Verification
We will add robust integration test coverage:
1. **`test_kerchunk_netcdf4.cpp`**: Writes a NetCDF-4 file with Kerchunk manifest writing enabled. Verifies the JSON is produced, is well-formed JSON, maps chunks to correct offsets/lengths, and accurately resolves the target URI (both local absolute and with `base_uri`).
2. **`test_kerchunk_grib2.cpp`**: Writes a multi-timestep GRIB2 file with Kerchunk enabled, verifying the ReferenceFileSystem JSON matches GRIB2 message coordinates exactly.
