# Implementation Plan: AXIS (Arbitrary eXgrid Interpolation Solver)

## Overview

Incremental implementation of the AXIS Tier 1 C++20 micro-library providing stateless spatial interpolation (regridding) for Earth-system fields via Kokkos, with a plain-data GridDescriptor ingest contract, sparse-matrix interpolation engine (bilinear + first-order conservative), distributed HaloPattern seam, symmetric egress contract, native Gmsh export, and a Fortran C-interop layer. Tasks build foundational types and memory-space traits first, then layer the ingest contract, topology builders, solver engine, distributed seam, egress, and finally CI verification.

## Tasks

- [x] 1. Repository scaffolding and CMake build system
  - [x] 1.1 Create repository directory structure and CMakeLists.txt
    - Create `libs/axis/` directory with subdirectories: `include/axis/`, `include/axis/ingest/`, `include/axis/topology/`, `include/axis/solver/`, `include/axis/detail/`, `src/`, `src/topology/`, `src/solver/`, `src/fortran/`, `fortran/`, `tests/`, `data/`, `cmake/`
    - Write root `CMakeLists.txt` with C++20 requirement, `find_package(Kokkos REQUIRED)`, `AXIS_ENABLE_PROJ` option (default ON), `BUILD_TESTING` option (default OFF), `BUILD_FORTRAN` option (default OFF)
    - Create `HELM::AXIS` alias target linking `Kokkos::kokkos` as PUBLIC dependency
    - Conditionally link PROJ as PRIVATE dependency guarding `src/topology/projection_builder.cpp`
    - Verify NO `find_package` for NetCDF, HDF5, ecCodes, yaml-cpp, TensorStore, g2c, eckit, AMIO, or HALO
    - _Requirements: 22.1, 22.2, 22.3, 22.4, 22.7, 12.2, 13.1_

  - [x] 1.2 Create CMake export and install configuration
    - Write `cmake/AXISConfig.cmake.in` and `cmake/AXISConfigVersion.cmake.in` templates
    - Configure `install(EXPORT AXISTargets ...)` with `NAMESPACE HELM::` and `SameMajorVersion` compatibility
    - Ensure downstream `find_package(AXIS)` works with transitive Kokkos dependency
    - _Requirements: 22.8_

  - [x] 1.3 Create test infrastructure CMakeLists.txt
    - Write `tests/CMakeLists.txt` that locates GTest and RapidCheck when `BUILD_TESTING=ON`
    - Define test executable targets for unit tests and property tests
    - _Requirements: 22.5_

  - [x] 1.4 Create README and .gitignore
    - Write `libs/axis/README.md` documenting prerequisites, Docker container launch, CMake configure/build commands, and test execution
    - Write `libs/axis/.gitignore` for build artifacts
    - _Requirements: 23.4_

- [x] 2. Compile-time memory space traits and mdspan interop
  - [x] 2.1 Implement detail::memory_traits and mdspan adapters
    - Create `include/axis/detail/memory_traits.hpp` with `is_device_space` trait specializations (CudaSpace, HIPSpace), `is_device_space_v`, `exec_space_t`
    - Create `include/axis/detail/mdspan_interop.hpp` with `to_view<T, MemorySpace, Rank>` (field_view → Kokkos::View) and `to_mdspan<ViewType>` (View → field_view) adapters
    - Define `field_view`, `field1d`, `field2d`, and `index_t` aliases in a top-level `include/axis/types.hpp` or within namespace header
    - Ensure both adapters are noexcept and perform zero copies
    - _Requirements: 1.1, 1.4, 1.5, 2.1, 2.6_

  - [x] 2.2 Write property test: layout_left Field View Round-Trip
    - **Property 1: layout_left Field View Round-Trip**
    - Generate random field arrays, wrap as field_view, pass through to_view then to_mdspan, verify identical memory address, extents, and element values
    - **Validates: Requirements 1.4, 1.5, 1.6**

- [x] 3. RAII resource handles
  - [x] 3.1 Implement detail::Proj_Handle and detail::File_Handle
    - Create `include/axis/detail/raii_handles.hpp` with both class declarations
    - Create `src/detail/raii_handles.cpp` implementing: Proj_Handle ctor (proj_create + nullptr check → throw), dtor (proj_destroy), move ops; File_Handle ctor (fopen + nullptr check → throw), dtor (fclose), move ops
    - Both are move-only with noexcept destructors
    - Guard Proj_Handle implementation behind `AXIS_ENABLE_PROJ` preprocessor flag
    - _Requirements: 17.1, 17.2, 17.3, 17.5, 17.6, 17.7_

  - [x] 3.2 Write property test: RAII Handle Releases on All Exit Paths
    - **Property 8: RAII Handle Releases on All Exit Paths**
    - Using release-counter spy, verify resource released exactly once on normal scope exit and on exception-driven scope exit for both Proj_Handle and File_Handle
    - **Validates: Requirements 17.1, 17.2, 17.4, 17.5**

- [x] 4. Checkpoint — Foundation layer complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Plain-data ingest contract (GridDescriptor)
  - [x] 5.1 Implement ingest::GridDescriptor header-only contract
    - Create `include/axis/ingest/grid_descriptor.hpp` with: ConventionKind enum (CF, UGRID, GRIB, Projected, NamedGrid, GridRules), CfParams, UgridParams, GribParams, ProjectedParams, NamedGridParams, GridRulesParams structs, BufferViews struct (layout_left mdspan views), and GridDescriptor struct
    - Ensure NO Kokkos type, AMIO type, eckit type, file handle, or file path in the header
    - Header-only: no associated .cpp compilation unit
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8_

  - [x] 5.2 Implement ingest::WeightEgress and MeshEgress egress contract
    - Create `include/axis/ingest/egress.hpp` with WeightEgress and MeshEgress structs (non-owning layout_left field_view members)
    - Implement `weight_egress()` and `mesh_egress()` free functions that build egress views without copying
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5, 15.6_

- [x] 6. Topology data structures
  - [x] 6.1 Implement topology::UnstructuredMesh class template
    - Create `include/axis/topology/unstructured_mesh.hpp` with full class declaration (constructor, accessors returning field_view, compute_areas)
    - Create `src/topology/unstructured_mesh.cpp` implementing compute_areas via Kokkos parallel kernel (GreatCircle spherical-excess and Cartesian planar polygon)
    - Create `include/axis/topology/enums.hpp` with ElementType, StaggerLoc, CoordinateSystem enums
    - _Requirements: 19.1, 19.2, 19.3, 19.4, 2.1, 2.2_

  - [x] 6.2 Implement topology::StructuredGrid class template
    - Create `include/axis/topology/structured_grid.hpp` with full class declaration (constructor, ni/nj, lon/lat accessors, set_corners, to_unstructured)
    - Create `src/topology/structured_grid.cpp` implementing to_unstructured() via Kokkos parallel kernel (each logical cell → one quadrilateral element)
    - _Requirements: 18.1, 18.2, 18.3_

  - [x] 6.3 Write property test: Structured-to-Unstructured Conversion Preserves Geometry
    - **Property 2: Structured-to-Unstructured Conversion Preserves Geometry**
    - Generate random StructuredGrid dimensions and coordinates, call to_unstructured(), verify ni*nj quadrilateral cells and corner coordinate equality
    - **Validates: Requirements 18.1, 18.2**

  - [x] 6.4 Write property test: CSR Connectivity Validity
    - **Property 3: CSR Connectivity Validity**
    - For meshes produced by any source, verify: offsets non-decreasing, offsets[0]==0, offsets[num_cells]==indices.size(), all indices in [0, num_nodes)
    - **Validates: Requirements 19.1, 19.2, 19.3, 19.4**

- [x] 7. Topology builders and generators
  - [x] 7.1 Implement topology::NamedGridRegistry
    - Create `include/axis/topology/named_grid_registry.hpp` with ParsedName, parse, is_registered, generate, registered_families
    - Create `src/topology/named_grid_registry.cpp` implementing O (octahedral Gaussian), F (regular Gaussian), N (reduced Gaussian) family generators via Kokkos parallel kernels with zero file I/O
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [x] 7.2 Implement topology::RuleGenerator
    - Create `include/axis/topology/rule_generator.hpp` with generate function template
    - Create `src/topology/rule_generator.cpp` implementing RegularLatLon, GaussianRegular, GaussianReduced, and Projected rule kinds via Kokkos parallel kernels (no YAML/JSON parsing)
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [x] 7.3 Implement topology::ProjectionBuilder
    - Create `include/axis/topology/projection_builder.hpp` with build function template
    - Create `src/topology/projection_builder.cpp` implementing PROJ-based coordinate transform (guarded by AXIS_ENABLE_PROJ)
    - _Requirements: 4.6, 7.4, 13.3_

  - [x] 7.4 Implement topology::MeshFactory (from_descriptor funnel)
    - Create `include/axis/topology/mesh_factory.hpp` with from_descriptor, from_named, from_rules static methods
    - Create `src/topology/mesh_factory.cpp` implementing: ConventionKind branch (CF, UGRID, GRIB, Projected, NamedGrid, GridRules), descriptor validation (unknown kind, missing field, inconsistent extents, null buffers), explicit Kokkos::deep_copy for space mismatch, zero-copy adoption when space matches
    - Ensure NO producer-specific branching — branch only on ConventionKind
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 4.10, 5.1, 5.2, 5.3, 5.4, 5.5_

  - [x] 7.5 Write property test: Named-Grid Generation Is Deterministic and File-Free
    - **Property 4: Named-Grid Generation Is Deterministic and File-Free**
    - Generate valid named-grid strings, call generate twice, verify identical node/cell counts and bitwise-identical coordinates
    - **Validates: Requirements 6.1, 6.5**

  - [x] 7.6 Write property test: Named-Grid Family Registration Round-Trip
    - **Property 5: Named-Grid Family Registration Round-Trip**
    - Generate valid/invalid name strings, verify is_registered/parse returns correct family and number or throws std::invalid_argument
    - **Validates: Requirements 6.3, 6.4**

  - [x] 7.7 Write property test: Rule-Based Generation Matches Resolution
    - **Property 6: Rule-Based Generation Matches Resolution**
    - Generate random RegularLatLon GridRulesParams, verify cell count == floor((max_x-min_x)/r_x) * floor((max_y-min_y)/r_y)
    - **Validates: Requirements 7.1, 7.2**

  - [x] 7.8 Write property test: Descriptor Field Validation
    - **Property 24: Descriptor Field Validation**
    - Generate deliberately malformed GridDescriptors (unknown kind, missing field, inconsistent extents, null buffers), verify from_descriptor throws std::invalid_argument naming the field
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5**

  - [x] 7.9 Write property test: Producer-Equivalence
    - **Property 9: Producer-Equivalence (Descriptor Is Producer-Agnostic)**
    - Construct paired "AMIO-style" and "Python-style" GridDescriptors with identical fields, verify from_descriptor produces meshes with identical node/cell counts and connectivity
    - **Validates: Requirements 20.1, 20.2, 3.8**

  - [x] 7.10 Write property test: Buffer Adoption With No Copy When Space Matches
    - **Property 25: layout_left Buffer Adoption With No Copy When Space Matches**
    - Construct descriptor with BufferViews in target MemorySpace, verify from_descriptor adopts without copying; construct in different space, verify deep_copy occurs
    - **Validates: Requirements 1.2, 4.9, 2.5**

- [x] 8. Checkpoint — Topology layer complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 9. Solver configuration and InterpolationMatrix
  - [-] 9.1 Implement solver enums and RegridConfig
    - Create `include/axis/solver/regrid_config.hpp` with InterpolationMethod, NormType, LineType, UnmappedAction enums and RegridConfig struct
    - _Requirements: 8.1, 10.1, 11.1, 11.2_

  - [-] 9.2 Implement solver::InterpolationMatrix class template
    - Create `include/axis/solver/interpolation_matrix.hpp` with IndexPair struct and full class declaration (constructor, nnz, n_src, n_dst, factor_list, factor_index, frac_a, frac_b, area_a, area_b accessors)
    - Create `src/solver/interpolation_matrix.cpp` implementing non-template members
    - _Requirements: 8.4, 9.1_

- [ ] 10. Weight generation engine
  - [ ] 10.1 Implement solver::WeightGenerator (single-rank)
    - Create `include/axis/solver/weight_generator.hpp` with generate (single-rank) and generate (distributed) static methods
    - Create `src/solver/weight_generator.cpp` implementing:
      - Bilinear: locate destination point in source cell, compute distance-based barycentric weights, set area_a=0.0
      - Conservative1stOrder: parallel cell-overlap w_ij=(f_ij*A_i)/A_j, accumulate frac_a/frac_b, NormType-aware weight normalization
      - CoordinateSystem mismatch check (throw std::invalid_argument)
      - Unmapped destination detection (Error → throw, Ignore → no entry)
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 11.1, 11.2, 11.3_

  - [ ] 10.2 Write property test: Sparse Matrix Index Bounds
    - **Property 10: Sparse Matrix Index Bounds**
    - For any generated InterpolationMatrix, verify all factor_index[k].col in [0, n_src) and factor_index[k].row in [0, n_dst)
    - **Validates: Requirements 8.4**

  - [ ] 10.3 Write property test: Conservative Weight Non-Negativity
    - **Property 11: Conservative Weight Non-Negativity**
    - For any matrix generated with Conservative1stOrder, verify all factor_list values >= 0
    - **Validates: Requirements 8.2**

- [ ] 11. SpMV apply and conservation accounting
  - [ ] 11.1 Implement solver::apply function templates
    - Create `include/axis/solver/apply.hpp` with: local apply (matrix + src + dst), distributed apply (matrix + pattern + local_src + gathered_halo_src + dst), and callback-based distributed apply
    - Implement Kokkos-parallel SpMV: zero dst, atomic_add S(k)*src(col_k) → dst(row_k)
    - Implement extent validation (throw std::invalid_argument if mismatched)
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 14.5, 14.6, 14.7_

  - [ ] 11.2 Implement solver::ConservationReport and integral functions
    - Create `include/axis/solver/conservation.hpp` with ConservationReport, source_integral, destination_integral, check_conservation, adjust_by_fraction
    - Create `src/solver/conservation.cpp` implementing Kokkos reductions
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7_

  - [ ] 11.3 Write property test: Partition of Unity (Constant-Field Preservation)
    - **Property 12: Partition of Unity (Constant-Field Preservation)**
    - For any conservative matrix and constant source field c, apply + adjust_by_fraction yields dst==c at all fully-covered cells within round-off
    - **Validates: Requirements 8.5, 10.2**

  - [ ] 11.4 Write property test: First-Order Conservation
    - **Property 13: First-Order Conservation (Σ src = Σ dst)**
    - For conservative regridding on meshes tiling the same domain, verify source_integral == destination_integral within relative tolerance 1e-12
    - **Validates: Requirements 8.1, 10.4, 10.5**

  - [ ] 11.5 Write property test: Bilinear Exactness on Linear Fields
    - **Property 14: Bilinear Exactness on Linear Fields**
    - Generate random affine field f(x,y)=a·x+b·y+c, apply bilinear weights, verify exact reproduction within round-off
    - **Validates: Requirements 8.3**

  - [ ] 11.6 Write property test: SpMV Apply Equals Reference Loop
    - **Property 17: SpMV Apply Equals Reference Loop**
    - For any matrix and source field, verify solver::apply equals scalar loop dst(row)+=S·src(col) within round-off
    - **Validates: Requirements 9.1, 9.2**

  - [ ] 11.7 Write property test: Apply Extent Validation
    - **Property 18: Apply Extent Validation**
    - Call apply with mismatched src/dst extents, verify std::invalid_argument thrown and dst untouched
    - **Validates: Requirements 9.4**

- [ ] 12. Normalization semantics tests
  - [ ] 12.1 Write property test: DstArea Normalization
    - **Property 15: DstArea Normalization Leaves Fraction Unbaked**
    - For DstArea matrices, verify dst_raw(j) == frac_b(j) * dst_true(j) and adjust_by_fraction recovers true value
    - **Validates: Requirements 10.1, 10.2**

  - [ ] 12.2 Write property test: FracArea Normalization
    - **Property 16: FracArea Normalization Bakes Fraction In**
    - For FracArea matrices, verify apply produces true interpolated value without division by frac_b
    - **Validates: Requirements 10.3**

  - [ ] 12.3 Write property test: Unmapped Destination Handling
    - **Property 20: Unmapped Destination Handling**
    - For uncovered destination cells: Ignore → no nonzero entry + apply leaves at zero; Error → throw std::runtime_error with unmapped index
    - **Validates: Requirements 11.1, 11.2**

  - [ ] 12.4 Write property test: Coordinate-System Consistency Enforcement
    - **Property 21: Coordinate-System Consistency Enforcement**
    - For src/dst meshes with differing CoordinateSystem, verify generate throws std::invalid_argument
    - **Validates: Requirements 11.3**

- [ ] 13. Checkpoint — Solver engine complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 14. Distributed regridding seam (HaloPattern)
  - [ ] 14.1 Implement solver::HaloPattern and distributed WeightGenerator overload
    - Create `include/axis/solver/halo_pattern.hpp` with HaloPattern struct (source_ranks, rank_offsets, needed_global_src_ids, gather_slot, num_remote, num_ranks)
    - Create `src/solver/halo_pattern.cpp` implementing num_remote/num_ranks
    - Integrate distributed-mode generate into WeightGenerator: scan col indices, group off-rank global src ids by neighbor rank, populate HaloPattern
    - _Requirements: 14.1, 14.2, 14.3, 14.4_

  - [ ] 14.2 Write property test: HaloPattern Completeness and Distributed Apply Equivalence
    - **Property 22: HaloPattern Completeness and Distributed Apply Equivalence**
    - Generate local matrices referencing off-rank global ids, verify HaloPattern enumerates exactly the off-rank source indices; distributed apply with manually-gathered buffer equals single-rank apply
    - **Validates: Requirements 14.1, 14.3, 14.5**

- [ ] 15. Gmsh native export
  - [ ] 15.1 Implement topology::GmshWriter
    - Create `include/axis/topology/gmsh_writer.hpp` with write static method template
    - Create `src/topology/gmsh_writer.cpp` implementing: Gmsh .msh v2.2 ASCII serialization, device→host mirror via explicit Kokkos::deep_copy, error handling via File_Handle RAII
    - _Requirements: 16.1, 16.2, 16.4_

  - [ ] 15.2 Write property test: Gmsh Export Round-Trip
    - **Property 7: Gmsh Export Round-Trip**
    - Write mesh with GmshWriter, read back, verify identical node/cell counts, connectivity, and coordinates within round-off
    - **Validates: Requirements 16.1, 16.3**

- [ ] 16. Umbrella header and Tier 1 isolation verification
  - [ ] 16.1 Create umbrella header and static isolation scan
    - Create `include/axis/axis.hpp` that includes all public headers (ingest, topology, solver, detail)
    - Create `cmake/check_tier1_isolation.sh` that scans all AXIS source/header files for `#include` directives matching forbidden HELM component paths (TICK, LOGS, HALO, AMIO, SPAN, DAGR, eckit, domain headers)
    - Verify no MPI_Comm, nc_*, codes_handle*, eckit::*, or file-format handle appears in any AXIS signature
    - Integrate as both a CMake custom target and a CTest
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 13.1, 13.2, 13.4, 13.5, 22.9_

- [ ] 17. Checkpoint — Core C++ library complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 18. Descriptor round-trip and host/device equivalence tests
  - [ ] 18.1 Write property test: Descriptor Round-Trip
    - **Property 23: Descriptor Round-Trip (Mesh → Descriptor → Mesh)**
    - Export mesh via mesh_egress, build UGRID GridDescriptor from those views, feed to from_descriptor, verify identical node/cell/connectivity
    - **Validates: Requirements 21.1, 15.2, 15.4**

  - [ ] 18.2 Write property test: Host/Device Result Equivalence
    - **Property 19: Host/Device Result Equivalence (No UVM)**
    - Generate and apply weights on HostSpace and on a device space, verify destination fields agree within round-off with no UVM reliance
    - **Validates: Requirements 2.3, 2.4**

- [ ] 19. Fortran C-interop layer
  - [ ] 19.1 Implement opaque handle registry
    - Create `src/fortran/handle_registry.hpp` with `axis::fortran::Handle_Registry` singleton
    - Implement thread-safe register_handle, lookup, release, valid methods
    - Use monotonically increasing integer tokens (0 reserved as invalid)
    - _Requirements: 22.6_

  - [ ] 19.2 Implement extern "C" interop functions
    - Create `src/fortran/axis_c_interop.cpp` with all extern "C" functions: axis_init_c, axis_mesh_from_named_c, axis_mesh_from_descriptor_c, axis_generate_weights_c, axis_apply_c, axis_destroy_mesh_c, axis_destroy_matrix_c
    - Implement AXIS_C_TRY macro for exception-to-error-code translation
    - Construct non-owning field_view over Fortran contiguous arrays via c_loc
    - _Requirements: 22.6_

  - [ ] 19.3 Implement axis_mod.f90 Fortran module
    - Create `fortran/axis_mod.f90` with iso_c_binding interfaces for all C interop functions
    - Implement public subroutines and error code constants
    - Add Fortran build targets to CMakeLists.txt (HELM::AXIS_Fortran alias)
    - _Requirements: 22.6_

- [ ] 20. Checkpoint — Fortran interop layer complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 21. CI pipeline and final integration
  - [ ] 21.1 Write unit test suite (example-based)
    - Create `tests/test_conservation.cpp` — exact-fraction conservation Σ src == Σ dst
    - Create `tests/test_constant_field.cpp` — partition-of-unity constant-field preservation
    - Create `tests/test_bilinear_exactness.cpp` — affine field exact reproduction
    - Create `tests/test_spmv_apply.cpp` — apply equals reference scalar loop + extent validation
    - Create `tests/test_descriptor_roundtrip.cpp` — mesh → UGRID descriptor → mesh
    - Create `tests/test_producer_equivalence.cpp` — AMIO-style vs Python-style identical output
    - Create `tests/test_descriptor_validation.cpp` — malformed descriptors throw correctly
    - Create `tests/test_named_and_rules.cpp` — named-grid and GridRules generation
    - Create `tests/test_gmsh_roundtrip.cpp` — Gmsh export → reimport
    - Create `tests/test_distributed_apply.cpp` — HaloPattern + gathered-buffer equivalence
    - Create `tests/test_conservation_norm.cpp` — DstArea vs FracArea normalization
    - Create `tests/test_raii_handles.cpp` — Proj_Handle and File_Handle release verification
    - _Requirements: 1.6, 8.1, 8.2, 8.3, 8.5, 9.2, 10.1, 10.3, 14.5, 16.3, 17.4, 20.1, 21.1_

  - [ ] 21.2 Configure CI pipeline documentation
    - Document CI pipeline stages: static analysis (isolation scan), standalone CMake build inside Docker, unit tests (GTest), property tests (RapidCheck ≥100 iterations), ASan + UBSan sanitizer builds
    - Verify standalone build produces HELM::AXIS without other HELM source trees present
    - _Requirements: 22.9, 23.2, 23.3_

- [ ] 22. Final checkpoint — All components integrated
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at logical boundaries
- Property tests validate universal correctness properties (Properties 1–25 from design)
- Unit tests validate specific examples, edge cases, and conservation accuracy
- RapidCheck is used for C++ property-based tests with minimum 100 iterations per property
- The design uses C++20 throughout — no language selection was needed
- AXIS's only optional third-party dependency is PROJ (libproj); all file-format I/O is delegated to AMIO via the GridDescriptor contract
- Tests build GridDescriptor objects directly in memory — no file libraries needed in AXIS CI
- The Fortran interop layer is optional (BUILD_FORTRAN=OFF by default)

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.4"] },
    { "id": 1, "tasks": ["1.2", "1.3"] },
    { "id": 2, "tasks": ["2.1", "3.1"] },
    { "id": 3, "tasks": ["2.2", "3.2", "5.1"] },
    { "id": 4, "tasks": ["5.2", "6.1", "6.2"] },
    { "id": 5, "tasks": ["6.3", "6.4", "7.1", "7.2", "7.3"] },
    { "id": 6, "tasks": ["7.4"] },
    { "id": 7, "tasks": ["7.5", "7.6", "7.7", "7.8", "7.9", "7.10"] },
    { "id": 8, "tasks": ["9.1", "9.2"] },
    { "id": 9, "tasks": ["10.1"] },
    { "id": 10, "tasks": ["10.2", "10.3", "11.1", "11.2"] },
    { "id": 11, "tasks": ["11.3", "11.4", "11.5", "11.6", "11.7"] },
    { "id": 12, "tasks": ["12.1", "12.2", "12.3", "12.4"] },
    { "id": 13, "tasks": ["14.1"] },
    { "id": 14, "tasks": ["14.2", "15.1"] },
    { "id": 15, "tasks": ["15.2", "16.1"] },
    { "id": 16, "tasks": ["18.1", "18.2"] },
    { "id": 17, "tasks": ["19.1"] },
    { "id": 18, "tasks": ["19.2"] },
    { "id": 19, "tasks": ["19.3"] },
    { "id": 20, "tasks": ["21.1", "21.2"] }
  ]
}
```
