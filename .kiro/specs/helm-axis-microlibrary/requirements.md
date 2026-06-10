# Requirements Document

## Introduction

AXIS (Arbitrary eXgrid Interpolation Solver) is a Tier 1 C++20 micro-library within the HELM ecosystem. It provides stateless spatial interpolation (regridding) for Earth-system fields, replacing the legacy ESMF spatial-discretization stack (`ESMF_Mesh`, `ESMF_Grid`, `ESMF_LocStream`, `ESMF_Regrid`, and the offline `ESMF_RegridWeightGen` application) with a decentralized, zero-copy, hardware-portable toolkit built on Kokkos. AXIS is scoped to stateless grid semantics and regridding math only — it opens no files, links no file-format libraries, and performs no MPI communication. All byte-level I/O is delegated to AMIO (the HELM Tier 1b bidirectional I/O engine) via a standard plain-data ingest contract (`GridDescriptor`), and all distributed communication is delegated to HALO via a neutral exchange-plan descriptor (`HaloPattern`). As a Tier 1 component, AXIS is completely blind to HALO, AMIO, TICK, LOGS, SPAN, DAGR, and all domain science code.

## Glossary

- **AXIS**: Arbitrary eXgrid Interpolation Solver; the Tier 1 C++20 micro-library providing stateless spatial interpolation and grid mesh construction within the HELM ecosystem.
- **GridDescriptor**: The standard, plain-data value type that serves as AXIS's public ingest contract. Contains only plain enums, plain fields, and non-owning `std::mdspan<layout_left>` buffer views — no Kokkos types, no AMIO types, no eckit types, no file handles.
- **ConventionKind**: An enum classifying how the GridDescriptor's metadata and buffers should be interpreted (CF, UGRID, GRIB, Projected, NamedGrid, GridRules).
- **MeshFactory**: The stateless facade through which all grids enter AXIS. `from_descriptor` is the single funnel for all file-backed sources; `from_named` and `from_rules` handle in-memory generation.
- **UnstructuredMesh**: The internal finite-element unstructured mesh representation that all source grids are converted into and that the solver operates on exclusively.
- **StructuredGrid**: A logically rectangular (rectilinear or curvilinear) grid representation that is converted to UnstructuredMesh before solver operations.
- **InterpolationMatrix**: A sparse interpolation operator in COO form — `factorList` (weights) and `factorIndexList` (source/destination index pairs) — matching ESMF weight-file semantics.
- **WeightGenerator**: The component that computes interpolation weights (bilinear and first-order conservative) from source and destination meshes.
- **HaloPattern**: A plain-data description of off-rank source values a local sparse-matrix apply needs, published by AXIS for a caller to hand directly to HALO. Contains no MPI types.
- **WeightEgress**: A plain-data, non-owning view over an InterpolationMatrix's weight buffers for consumers to serialize.
- **MeshEgress**: A plain-data, non-owning view over a mesh's node coordinates and CSR connectivity for consumers to serialize.
- **GmshWriter**: The native Gmsh `.msh` format exporter — AXIS's only native file writer, using plain stdio with no third-party library.
- **NamedGridRegistry**: A registry of hardcoded mathematical generators for standard global weather grids (O-octahedral, F-regular, N-reduced Gaussian families).
- **RuleGenerator**: A rule-based mesh generator that builds grids from GridRules descriptor parameters using Kokkos parallel kernels with zero file I/O.
- **ProjectionBuilder**: A builder that transforms projection-space grids to geographic coordinates using PROJ (AXIS's single optional third-party dependency).
- **Proj_Handle**: An RAII wrapper for the PROJ `PJ*` transformation object.
- **File_Handle**: An RAII wrapper for `std::FILE*`, used only by GmshWriter for plain-text `.msh` export.
- **RegridConfig**: Configuration struct capturing interpolation method, normalization type, line type, and unmapped-point handling.
- **ConservationReport**: A struct reporting source and destination integrals, absolute error, and relative error for conservation verification.
- **field_view**: A convenience alias for `std::mdspan<T, std::dextents<std::size_t, Rank>, std::layout_left>` — the non-owning, column-major view type AXIS uses at all boundaries.
- **AMIO**: Asynchronous Multidimensional Input Output Library; the HELM Tier 1b bidirectional I/O engine that owns the file-format stack (NetCDF-4, Zarr v3, GRIB2, eckit) and acts as a GridDescriptor producer.
- **HALO**: Hardware-Abstracted Link Operations; the HELM Tier 1 MPI communication library that executes the gather described by a HaloPattern.
- **DAGR**: Directed Acyclic Graph Router; the HELM Tier 3 orchestrator. Explicitly NOT involved in the AXIS ↔ AMIO or AXIS ↔ HALO handoffs.
- **CSR**: Compressed Sparse Row; the connectivity storage format for mixed-element unstructured meshes.
- **SpMV**: Sparse Matrix-Vector Multiply; the parallel operation that applies interpolation weights to a source field to produce a destination field.
- **Kokkos**: The C++ performance portability framework providing parallel patterns and memory-space abstraction (HELM Law #2).
- **MemorySpace**: A Kokkos concept representing physical memory location (HostSpace, CudaSpace, HIPSpace). AXIS uses explicit MemorySpace template parameters — no UVM.
- **PROJ**: The libproj coordinate transformation library — AXIS's single optional third-party dependency for projection-string transforms.
- **layout_left**: The `std::layout_left` mapping policy for mdspan, matching Fortran column-major memory order.

## Requirements

### Requirement 1: Zero-Copy Memory Interface via layout_left mdspan

**User Story:** As a domain model developer, I want AXIS to accept and return field data as non-owning `std::mdspan<layout_left>` views, so that no field arrays are copied between the model and the interpolation engine and the Fortran column-major memory layout is preserved bit-for-bit.

#### Acceptance Criteria

1. THE field_view type alias SHALL be defined as `std::mdspan<T, std::dextents<std::size_t, Rank>, std::layout_left>` for all field and coordinate data crossing the AXIS boundary.
2. WHEN a field_view is passed to any AXIS public function (apply, from_descriptor, weight_egress, mesh_egress), THE function SHALL operate on the memory the view addresses without allocating or copying the underlying buffer.
3. WHEN AXIS produces mesh or weight results, THE results SHALL be exposed to consumers through non-owning field_view accessors that reference AXIS-owned internal buffers.
4. THE detail::to_view adapter SHALL convert a layout_left field_view to a Kokkos LayoutLeft unmanaged View in the same MemorySpace without copying the underlying buffer.
5. THE detail::to_mdspan adapter SHALL convert a Kokkos LayoutLeft unmanaged View back to a layout_left field_view without copying the underlying buffer.
6. FOR ALL field_view values, applying to_view then to_mdspan SHALL yield a view addressing the identical memory with identical extents and identical element values (round-trip identity).

### Requirement 2: Explicit Memory Space and Hardware Portability

**User Story:** As an HPC engineer, I want AXIS to use explicit Kokkos MemorySpace template parameters and Kokkos-only parallelism, so that data placement is deterministic, GPU execution requires no Unified Virtual Memory reliance, and one code path runs on both CPU and GPU.

#### Acceptance Criteria

1. THE topology and solver components SHALL be templated on a Kokkos MemorySpace parameter that determines where internal arrays are allocated and where parallel kernels execute.
2. WHEN a MemorySpace is a device space (CudaSpace, HIPSpace), THE AXIS solver and topology code SHALL execute Kokkos parallel patterns on the corresponding device execution space.
3. WHEN generating or applying weights entirely on Kokkos::HostSpace and entirely on a device space, THE produced destination fields SHALL agree within round-off tolerance.
4. AXIS SHALL NOT rely on Unified Virtual Memory page migration for any data movement between host and device.
5. WHEN data must be moved between host and device (e.g., staging descriptor buffers into device MemorySpace), AXIS SHALL use explicit `Kokkos::deep_copy` calls.
6. AXIS SHALL NOT contain any raw CUDA, HIP, or OpenMP parallel constructs — all parallelism SHALL use Kokkos patterns exclusively.

### Requirement 3: Standard Plain-Data GridDescriptor Ingest Contract

**User Story:** As a library integrator, I want AXIS to publish a single, standard, plain-data GridDescriptor value type as its public ingest contract, so that any producer (AMIO today, Python/pybind11 tomorrow) can populate the same descriptor and feed AXIS identically without coupling to AXIS internals.

#### Acceptance Criteria

1. THE GridDescriptor struct SHALL contain only plain enums, plain scalar fields, standard library containers of plain types, and non-owning `std::mdspan<layout_left>` buffer views.
2. THE GridDescriptor SHALL NOT contain any Kokkos type, AMIO type, eckit type, file handle, or file path.
3. THE GridDescriptor SHALL include a ConventionKind enum field selecting from CF, UGRID, GRIB, Projected, NamedGrid, and GridRules.
4. THE GridDescriptor SHALL be trivially copyable as a value type that owns no resources and has no destructor logic.
5. THE GridDescriptor SHALL include convention-specific metadata structs (CfParams, UgridParams, GribParams, ProjectedParams, NamedGridParams, GridRulesParams) where only the member matching the ConventionKind is meaningful.
6. THE GridDescriptor SHALL include a BufferViews struct containing non-owning layout_left mdspan views over the decoded coordinate/connectivity/area/mask arrays the producer supplies.
7. THE GridDescriptor SHALL be header-only with no associated .cpp compilation unit, so that producers include a single header to access the contract.
8. THE GridDescriptor design SHALL be easily duplicatable in Python (pybind11 + numpy) — a Python layer SHALL be able to construct the identical descriptor from numpy arrays using only public fields, with no AXIS-internal machinery required.

### Requirement 4: MeshFactory from_descriptor — Single File-Backed Funnel

**User Story:** As a library integrator, I want all file-backed grids to enter AXIS through a single `MeshFactory::from_descriptor` entry point, so that there is one unified interpretation path regardless of grid format or producer identity.

#### Acceptance Criteria

1. THE MeshFactory::from_descriptor function SHALL be the single entry point through which all file-backed grid sources are interpreted and converted to an UnstructuredMesh.
2. WHEN from_descriptor is called, THE function SHALL branch on the descriptor's ConventionKind exactly once and SHALL NOT branch on producer identity.
3. WHEN the ConventionKind is CF, THE function SHALL interpret CfParams and BufferViews center_x/center_y to construct a StructuredGrid and convert it to an UnstructuredMesh via to_unstructured().
4. WHEN the ConventionKind is UGRID, THE function SHALL adopt the BufferViews node_coords and CSR connectivity (conn_offsets, conn_indices) directly into an UnstructuredMesh.
5. WHEN the ConventionKind is GRIB, THE function SHALL reconstruct geometry from GribParams fields plus BufferViews to construct a StructuredGrid and convert it to an UnstructuredMesh.
6. WHEN the ConventionKind is Projected, THE function SHALL invoke ProjectionBuilder with ProjectedParams to transform projection-space coordinates to geographic coordinates.
7. WHEN the ConventionKind is NamedGrid, THE function SHALL delegate to NamedGridRegistry::generate using the NamedGridParams name token, ignoring BufferViews.
8. WHEN the ConventionKind is GridRules, THE function SHALL delegate to RuleGenerator::generate using the GridRulesParams fields, ignoring BufferViews.
9. WHEN the descriptor's buffer views already address memory in the target MemorySpace, THE function SHALL adopt those buffers without copying the underlying arrays.
10. WHEN the descriptor's buffer views address memory in a different space than the target MemorySpace, THE function SHALL stage them via explicit Kokkos::deep_copy.

### Requirement 5: Descriptor Validation

**User Story:** As a library developer, I want from_descriptor to validate the GridDescriptor thoroughly before building a mesh, so that malformed inputs are caught early with descriptive error messages rather than causing undefined behavior.

#### Acceptance Criteria

1. WHEN from_descriptor receives a GridDescriptor with an unknown or unsupported ConventionKind, THE function SHALL throw std::invalid_argument naming the offending kind.
2. WHEN from_descriptor receives a GridDescriptor missing a required field for its declared kind (e.g., GRIB with empty grid_type, Projected with empty proj_string), THE function SHALL throw std::invalid_argument naming the missing field.
3. WHEN from_descriptor receives a GridDescriptor with inconsistent buffer extents (e.g., conn_offsets.extent(0) != n_cells + 1, or center_x/center_y extents disagreeing with ni/nj), THE function SHALL throw std::invalid_argument describing the extent mismatch.
4. WHEN from_descriptor receives a GridDescriptor with a null or empty buffer where the kind requires one (e.g., UGRID with empty node_coords or conn_indices), THE function SHALL throw std::invalid_argument naming the empty buffer.
5. WHEN from_descriptor throws a validation exception, THE function SHALL NOT produce a partial mesh — the guarantee is strong (all-or-nothing).

### Requirement 6: Named-Grid In-Memory Generation

**User Story:** As a domain scientist, I want to generate standard global weather grids (O-octahedral, F-regular, N-reduced Gaussian families) by name in parallel with zero file I/O, so that common grids are available immediately without external data files.

#### Acceptance Criteria

1. WHEN NamedGridRegistry::generate is called with a valid grid name string (e.g., "O1280", "F128", "N320"), THE registry SHALL produce a complete UnstructuredMesh using a Kokkos parallel kernel with zero file I/O.
2. THE NamedGridRegistry SHALL support at minimum the O (octahedral Gaussian), F (regular Gaussian), and N (reduced Gaussian) family prefixes.
3. WHEN NamedGridRegistry::parse is called with a valid name, THE function SHALL return a ParsedName with the correct family character and positive number.
4. WHEN NamedGridRegistry::parse is called with an unknown family prefix or a non-positive number, THE function SHALL throw std::invalid_argument.
5. FOR ALL registered named-grid strings, two independent calls to generate SHALL produce meshes with identical node counts, identical cell counts, and bitwise-identical node coordinates (deterministic generation).

### Requirement 7: Rule-Based Mesh Generation

**User Story:** As a domain scientist, I want to generate meshes from rule parameters (kind, bounding box, resolution, Gaussian N) without parsing any file, so that grid generation is driven by decoded parameters the producer supplied.

#### Acceptance Criteria

1. WHEN RuleGenerator::generate is called with valid GridRulesParams, THE generator SHALL produce an UnstructuredMesh using a Kokkos parallel kernel with zero file I/O and zero YAML/JSON parsing.
2. FOR GridRulesParams of kind RegularLatLon, THE generated mesh SHALL have exactly `floor((max_x - min_x)/r_x) * floor((max_y - min_y)/r_y)` cells matching the bbox and resolution.
3. THE RuleGenerator SHALL support at minimum RegularLatLon, GaussianRegular, GaussianReduced, and Projected rule kinds.
4. WHEN GridRulesParams of kind Projected includes a proj_string, THE generator SHALL use ProjectionBuilder for the coordinate transform.
5. WHEN GridRulesParams contains invalid or inconsistent parameters (e.g., zero resolution, bbox max < min), THE generator SHALL throw std::invalid_argument.

### Requirement 8: Interpolation Weight Generation

**User Story:** As a domain scientist, I want AXIS to compute interpolation weights (bilinear and first-order conservative) from source and destination meshes, so that field data can be transferred accurately between different grid resolutions.

#### Acceptance Criteria

1. WHEN WeightGenerator::generate is called with Conservative1stOrder method, THE weights SHALL satisfy first-order conservation: `Σ_i src(i)·area_a(i)·frac_a(i) == Σ_j dst(j)·area_b(j)[·frac_b(j)]` within relative tolerance 1e-12 for meshes tiling the same domain.
2. WHEN WeightGenerator::generate is called with Conservative1stOrder method, THE weight values in factor_list SHALL all be greater than or equal to zero (non-negative weights).
3. WHEN WeightGenerator::generate is called with Bilinear method and the source field is an affine function `f(x,y) = a·x + b·y + c`, THE applied result SHALL reproduce `f` at each destination point within round-off tolerance (bilinear exactness).
4. FOR ALL produced InterpolationMatrix instances, every factor_index[k].col SHALL be in [0, n_src) and every factor_index[k].row SHALL be in [0, n_dst) (valid index bounds).
5. WHEN WeightGenerator::generate is called with Conservative1stOrder and a constant source field value c over all fully-covered source cells, THE applied and frac_b-adjusted destination field SHALL equal c at every fully-covered destination cell within round-off tolerance (partition of unity / constant-field preservation).
6. WHEN WeightGenerator::generate is called with Bilinear method, THE source cell areas (area_a) SHALL be set to 0.0, matching ESMF bilinear weight-file semantics.

### Requirement 9: Sparse Matrix-Vector Apply (SpMV)

**User Story:** As a domain scientist, I want to apply precomputed interpolation weights to source fields efficiently using Kokkos-parallel sparse matrix-vector multiplication, so that regridding executes at full hardware throughput on both CPU and GPU.

#### Acceptance Criteria

1. THE solver::apply function SHALL accept an InterpolationMatrix, source field_view, and destination field_view, implementing `dst(row(k)) += S(k) * src(col(k))` via Kokkos atomic_add after zero-initializing dst.
2. FOR ALL valid InterpolationMatrix and source field inputs, solver::apply SHALL produce a destination field equal to the reference scalar loop within round-off tolerance.
3. THE solver::apply function SHALL execute entirely within the MemorySpace of the InterpolationMatrix template parameter, using Kokkos parallel patterns.
4. WHEN solver::apply is called with src.extent(0) != matrix.n_src() or dst.extent(0) != matrix.n_dst(), THE function SHALL throw std::invalid_argument and SHALL NOT write to dst.
5. THE solver::apply function SHALL support a distributed overload that reads locally-owned source cells from a local_src view and off-rank source cells from a gathered_halo_src buffer produced by HALO.

### Requirement 10: Normalization Semantics (DstArea and FracArea)

**User Story:** As a domain scientist, I want AXIS to support both DstArea and FracArea normalization matching ESMF semantics, so that partially-covered destination cells are handled correctly for my operational workflow.

#### Acceptance Criteria

1. WHEN NormType is DstArea (the default), THE conservative weights SHALL be unnormalized such that `dst_raw(j) = frac_b(j) · dst_true(j)` for partially-covered cells.
2. WHEN NormType is DstArea, THE user SHALL be able to recover the true interpolated value by dividing by frac_b(j) via adjust_by_fraction.
3. WHEN NormType is FracArea, THE conservative weights SHALL bake the fraction into the weight so that the applied destination field requires no division by frac_b to yield the true interpolated value.
4. THE source_integral function SHALL compute `Σ src(i) * area_a(i) * frac_a(i)` via Kokkos reduction.
5. THE destination_integral function SHALL compute the integral consistent with the configured NormType: for DstArea, `Σ dst(j) * area_b(j)` over cells with frac_b != 0; for FracArea, `Σ dst(j) * area_b(j) * frac_b(j)`.
6. THE adjust_by_fraction function SHALL divide dst(j) by frac_b(j) for all cells where frac_b != 0, matching ESMF guidance.
7. THE check_conservation function SHALL produce a ConservationReport with src_integral, dst_integral, absolute_error, and relative_error.

### Requirement 11: Unmapped Destination Handling

**User Story:** As a domain scientist, I want to control what happens when destination cells have no source coverage, so that I can either silently ignore gaps or fail fast in workflows requiring full coverage.

#### Acceptance Criteria

1. WHEN RegridConfig::unmapped is set to Ignore and a destination cell receives no source coverage, THE InterpolationMatrix SHALL contain no nonzero entry for that cell and solver::apply SHALL leave the destination value at zero.
2. WHEN RegridConfig::unmapped is set to Error and any destination cell receives no source coverage, THE WeightGenerator::generate function SHALL throw std::runtime_error identifying the first unmapped destination index.
3. WHEN WeightGenerator::generate is called with source and destination meshes whose CoordinateSystem values differ, THE function SHALL throw std::invalid_argument and SHALL NOT produce a matrix.

### Requirement 12: Tier 1 Isolation (HELM Law #4 — Dependency Inversion)

**User Story:** As an architect, I want AXIS to have zero compile-time or link-time dependencies on HALO, AMIO, TICK, LOGS, SPAN, DAGR, eckit, or any domain-science header, so that the no-circular-dependency invariant of the HELM architecture is preserved and AXIS remains a blind, self-contained Tier 1 utility.

#### Acceptance Criteria

1. THE AXIS library SHALL NOT include any header from HALO, AMIO, TICK, LOGS, SPAN, DAGR, eckit, or domain-science models in any source file, public header, or internal header.
2. THE AXIS library SHALL NOT link against any HELM library target (HELM::HALO, HELM::AMIO, HELM::TICK, HELM::LOGS, HELM::SPAN, HELM::DAGR) at build time — AMIO is explicitly NOT a CMake link dependency of AXIS.
3. THE AXIS public headers SHALL contain only `#include` directives referencing C++ standard library headers, Kokkos headers, and (optionally) PROJ headers.
4. NO MPI_Comm, HALO type, nc_* id, codes_handle*, eckit::*, TensorStore type, or any file-format handle SHALL appear in any AXIS public signature or internal source.
5. THE AXIS build system SHALL execute a static scan (check_tier1_isolation.sh) that fails the build if any forbidden `#include` or Fortran `use` directive is found in shipping sources — the forbidden list SHALL include HALO, AMIO, eckit, TICK, LOGS, SPAN, DAGR, and domain headers.
6. THE dependency direction SHALL be one-way and acyclic: AMIO and the future Python layer depend on AXIS's public contract header; AXIS depends on none of them.

### Requirement 13: No Redundant I/O Stack — Delegate File I/O to AMIO

**User Story:** As an architect, I want AXIS to contain zero file-format parsing code and link no file-format libraries, so that the software stack is not duplicated between AXIS and AMIO and AXIS remains a pure math/grid-semantics library.

#### Acceptance Criteria

1. AXIS SHALL NOT include or link ecCodes, yaml-cpp, NetCDF-C, HDF5, TensorStore, nceplibs-g2c, or any other file-format library.
2. AXIS SHALL NOT open any file for reading grid data — all file-based grid data SHALL arrive through the GridDescriptor populated by a producer.
3. THE ONLY optional third-party dependency of AXIS SHALL be PROJ (libproj) for coordinate transforms — guarded by the AXIS_ENABLE_PROJ option.
4. AXIS SHALL NOT contain any file-reader classes (no CFReader, UgridReader, GribReader, or YamlGridBuilder) — those responsibilities are delegated to the producer (AMIO or Python).
5. Convention detection (sniffing CF vs UGRID vs GRIB at the byte level) SHALL be performed by the producer (AMIO), not by AXIS — AXIS only interprets the decoded descriptor fields.
6. THE GridDescriptor's GridRulesParams SHALL contain plain fields (kind, bbox, resolution, gaussian_n, proj_string) that the producer parsed from YAML via eckit — AXIS SHALL NOT parse YAML or JSON.

### Requirement 14: HALO Integration Seam (Distributed Regridding)

**User Story:** As a coupled-model developer, I want AXIS to publish the off-rank communication pattern as plain data that a caller hands directly to HALO without DAGR mediation, so that distributed regridding works across MPI ranks while AXIS remains MPI-free.

#### Acceptance Criteria

1. WHEN WeightGenerator::generate is called in distributed mode (with global_ids and owner_of_src), THE function SHALL produce a local InterpolationMatrix and additionally publish a HaloPattern describing the off-rank source indices needed per neighbor rank.
2. THE HaloPattern struct SHALL contain only plain integer arrays (source_ranks, rank_offsets, needed_global_src_ids, gather_slot) — no MPI types, no HALO types, no AMIO types.
3. THE HaloPattern SHALL group off-rank global source IDs in CSR form by owning neighbor rank, with gather_slot specifying the local index in the gathered buffer where each off-rank value should be deposited.
4. A caller SHALL hand the HaloPattern directly to HALO for the actual MPI gather — NO DAGR mediation is required.
5. THE distributed solver::apply overload SHALL accept local_src (locally-owned source values), gathered_halo_src (off-rank values from HALO in gather_slot order), the HaloPattern, and the InterpolationMatrix, and SHALL produce a destination field bitwise-identical (within round-off) to a single-rank apply over the full source field.
6. AXIS SHALL additionally support a distributed apply overload accepting a caller-supplied gather callback/functor that returns off-rank source values, keeping AXIS blind to how they were communicated.
7. WHEN gathered_halo_src.extent(0) != pattern.num_remote() or local_src extent disagrees with the matrix dimensions, THE distributed apply SHALL throw std::invalid_argument.

### Requirement 15: Symmetric Egress Contract

**User Story:** As a library integrator, I want AXIS to expose its weight and mesh outputs as plain-data egress views that consumers (AMIO or Python) can serialize, so that output serialization is decoupled from AXIS the same way input is.

#### Acceptance Criteria

1. THE WeightEgress struct SHALL provide non-owning layout_left field_view members over factor_list, factor_col, factor_row, frac_a, frac_b, area_a, area_b, and dimension counts, matching ESMF/SCRIP weight-file structure.
2. THE MeshEgress struct SHALL provide non-owning layout_left field_view members over node_coords, conn_offsets, conn_indices, cell_areas, cell_mask, and a CoordinateSystem field.
3. THE weight_egress function SHALL build a WeightEgress view over an InterpolationMatrix without copying any data.
4. THE mesh_egress function SHALL build a MeshEgress view over an UnstructuredMesh without copying any data.
5. THE egress types SHALL own no resources, name no Kokkos/AMIO/eckit type, and require no DAGR to reach a consumer.
6. AXIS's only native file writer SHALL be GmshWriter for Gmsh .msh format — all other output formats (SCRIP, ESMF weight files, NetCDF, Zarr) SHALL be produced by a consumer (AMIO or Python) reading the egress views.

### Requirement 16: Native Gmsh Export

**User Story:** As a domain scientist, I want AXIS to natively write meshes to Gmsh `.msh` format for offline visualization and analysis, so that I can inspect grids without external tooling dependencies.

#### Acceptance Criteria

1. THE GmshWriter SHALL serialize an UnstructuredMesh to Gmsh .msh v2.2 ASCII format using plain stdio (std::FILE* via File_Handle) with no third-party library dependency.
2. WHEN the mesh resides in device memory, THE GmshWriter SHALL mirror node/connectivity arrays to host via explicit Kokkos::deep_copy before serialization.
3. FOR ALL UnstructuredMesh instances, writing with GmshWriter and reading the resulting .msh file back SHALL reconstruct a mesh with identical node count, identical cell count, identical connectivity, and node coordinates equal within round-off tolerance (Gmsh round-trip).
4. IF the file cannot be opened or a write error occurs, THE GmshWriter SHALL throw std::runtime_error naming the file path.

### Requirement 17: RAII Resource Handle Management

**User Story:** As a library developer, I want the only two retained C resource handles in AXIS (Proj_Handle for PROJ and File_Handle for Gmsh stdio) wrapped in move-only RAII types, so that resources are released on all exit paths including exceptions.

#### Acceptance Criteria

1. THE Proj_Handle SHALL acquire a PROJ PJ* on construction from a proj_string and SHALL call proj_destroy in its destructor.
2. THE File_Handle SHALL acquire a std::FILE* on construction and SHALL call std::fclose in its destructor.
3. BOTH Proj_Handle and File_Handle SHALL be move-only (delete copy construction and copy assignment).
4. FOR ALL code paths that acquire either handle, THE underlying C resource SHALL be released exactly once whether the operation returns normally or exits via exception.
5. BOTH destructors SHALL be noexcept — errors during resource release SHALL be swallowed silently.
6. AXIS SHALL NOT contain any NetCDF_Handle, Grib_Handle, YAML handle, or any other file-format handle — those are deleted.
7. IF Proj_Handle is constructed with an invalid proj_string that PROJ cannot parse, THE constructor SHALL throw std::runtime_error with the PROJ error text.

### Requirement 18: Structured-to-Unstructured Conversion

**User Story:** As a library developer, I want StructuredGrid to provide a to_unstructured() conversion, so that all grid types funnel into the single internal FEM representation the solver operates on.

#### Acceptance Criteria

1. WHEN StructuredGrid::to_unstructured() is called on a grid of dimensions ni × nj, THE function SHALL produce an UnstructuredMesh with exactly ni * nj quadrilateral cells.
2. THE four node coordinates of each cell in the produced mesh SHALL equal the four corner coordinates of the corresponding logical grid cell.
3. THE conversion SHALL execute via a Kokkos parallel kernel for hardware portability.

### Requirement 19: CSR Connectivity Integrity

**User Story:** As a library developer, I want all produced UnstructuredMesh instances to maintain valid CSR connectivity invariants, so that downstream consumers can safely traverse the mesh without bounds-checking every access.

#### Acceptance Criteria

1. FOR ALL UnstructuredMesh instances produced by any source (from_descriptor, NamedGridRegistry, RuleGenerator, to_unstructured), THE cell_node_offsets array SHALL be non-decreasing.
2. THE first element cell_node_offsets[0] SHALL equal 0.
3. THE last element cell_node_offsets[num_cells] SHALL equal cell_node_indices.size().
4. EVERY value in cell_node_indices SHALL be in the range [0, num_nodes).

### Requirement 20: Producer-Equivalence (Python-Mirrorable Contract)

**User Story:** As a Python integration developer, I want AXIS's GridDescriptor to be completely producer-agnostic such that an AMIO-populated descriptor and a Python/numpy-populated descriptor with identical fields produce identical meshes, so that a native Python interface is a thin wrapper rather than a re-implementation.

#### Acceptance Criteria

1. FOR ALL pairs of producers that populate a GridDescriptor with field-for-field identical content, MeshFactory::from_descriptor SHALL produce UnstructuredMesh instances with identical node counts, identical cell counts, and identical connectivity.
2. THE from_descriptor function SHALL contain no producer-specific branching — the only branch SHALL be on ConventionKind.
3. THE GridDescriptor's "easily duplicated in Python" property SHALL be verified by tests that construct descriptors with plain in-memory arrays (simulating both AMIO-style and Python-style population) and confirm identical mesh output.

### Requirement 21: Descriptor Round-Trip (Mesh → Descriptor → Mesh)

**User Story:** As a library developer, I want the ingest and egress contracts to be structural inverses, so that a mesh can be exported via MeshEgress and re-imported via a UGRID GridDescriptor to reproduce an equivalent mesh.

#### Acceptance Criteria

1. FOR ALL UnstructuredMesh instances, building a GridDescriptor of kind UGRID from the mesh's own node-coordinate and CSR-connectivity arrays (as surfaced by mesh_egress) and feeding it to from_descriptor SHALL reproduce a mesh with identical node count, identical cell count, identical connectivity, and node coordinates equal within round-off tolerance.

### Requirement 22: CMake Build System

**User Story:** As a build engineer, I want AXIS to provide a standalone CMake build system that requires only Kokkos (and optionally PROJ), so that AXIS builds independently of AMIO, HALO, and all file-format libraries.

#### Acceptance Criteria

1. THE AXIS build system SHALL use CMake version 3.21 or later and require the C++20 standard.
2. THE AXIS build system SHALL locate Kokkos using find_package(Kokkos REQUIRED).
3. THE AXIS build system SHALL produce a library target named axis with a namespace alias HELM::AXIS, linking Kokkos as a PUBLIC dependency.
4. THE AXIS build system SHALL provide an AXIS_ENABLE_PROJ option (default ON) that, when enabled, locates PROJ and links it as a PRIVATE dependency guarding only projection_builder.cpp.
5. THE AXIS build system SHALL provide a BUILD_TESTING option (default OFF) that, when ON, locates GTest and RapidCheck and builds the test suite.
6. THE AXIS build system SHALL provide a BUILD_FORTRAN option (default OFF) that, when ON, builds the iso_c_binding Fortran interop layer.
7. THE AXIS build system SHALL NOT contain any find_package for NetCDF, HDF5, ecCodes, yaml-cpp, TensorStore, g2c, eckit, AMIO, or HALO.
8. THE AXIS build system SHALL export CMake configuration files (AXISConfig.cmake, AXISConfigVersion.cmake, AXISTargets.cmake) so downstream projects can consume AXIS via find_package(AXIS).
9. THE AXIS build system SHALL include the check_tier1_isolation.sh script as both a custom target and a CTest that fails the build on any forbidden include.

### Requirement 23: Repository and Container Structure

**User Story:** As a build engineer, I want AXIS to live in its own dedicated repository added as a Git submodule within the HELM project, so that Tier 1 libraries maintain independent version histories and CI pipelines while remaining composable.

#### Acceptance Criteria

1. THE AXIS source code SHALL reside in a dedicated Git repository that is added as a Git submodule at the path `libs/axis` within the HELM project workspace root.
2. THE AXIS build and test workflow SHALL use the Docker image built from the HELM project DockerFile as its development and CI container environment.
3. THE AXIS repository SHALL contain a standalone CMakeLists.txt at its root that, when configured and built inside the helm-project Docker container without any other HELM source trees present, produces the HELM::AXIS library target without build errors.
4. THE AXIS repository SHALL include a README at its root documenting prerequisites, Docker launch command, CMake configure/build commands, and test execution.
