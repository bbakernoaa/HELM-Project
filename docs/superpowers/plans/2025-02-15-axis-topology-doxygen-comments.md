# AXIS Topology Doxygen Comments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add complete, highly descriptive, and consistent Doxygen comments to all public header files in the AXIS Topology library.

**Architecture:** Update only existing `///` Doxygen blocks and add new ones for all undocumented classes, structs, enums, values, public functions, and templates. Ensure standard tags (`@file`, `@brief`, `@tparam`, `@param`, `@return`, `@throw`) are used with complete sentences and explicit types.

**Tech Stack:** C++20, Doxygen

## Global Constraints

- Use standard `///` Doxygen blocks.
- Add file-level comments (`@file`, `@brief`).
- Document all classes, structs, enums, values, public functions, and templates.
- Include `@tparam`, `@param`, `@return`, and `@throw` where appropriate, using complete sentences and explicit types.
- Maintain existing C++20 formatting and compile-readiness. No functional logic should be altered.
- NEVER use hacks like disabling or suppressing warnings, and maintain exact C++20 validity.

---

### Task 1: Document `libs/axis/include/axis/topology/enums.hpp`

**Files:**
- Modify: `libs/axis/include/axis/topology/enums.hpp`

- [ ] **Step 1: Read and analyze `enums.hpp` thoroughly**
- [ ] **Step 2: Add highly descriptive Doxygen comments to `enums.hpp`**
  - Document the `axis::topology` namespace itself if appropriate.
  - Enrich the `@file` and `@brief` block.
  - Add highly descriptive `@brief` and detailed comments to `ElementType`, `StaggerLoc`, and `CoordinateSystem` enum classes and their specific enumeration values.
- [ ] **Step 3: Compile and verify correctness of headers**
  Running standard build verification commands to ensure no syntax errors are introduced.

---

### Task 2: Document `libs/axis/include/axis/topology/unstructured_mesh.hpp`

**Files:**
- Modify: `libs/axis/include/axis/topology/unstructured_mesh.hpp`

- [ ] **Step 1: Read and analyze `unstructured_mesh.hpp` thoroughly**
- [ ] **Step 2: Add comprehensive Doxygen comments to `unstructured_mesh.hpp`**
  - Enrich `@file` and `@brief` file-level comments.
  - Document `UnstructuredMesh` class, including `@tparam MemorySpace` with explicit type details.
  - Document public `using memory_space = MemorySpace;` type alias.
  - Document the default constructor and the parameterized constructor with all `@param` and `@throws` (none thrown, but document parameters).
  - Document all public scalar accessors (`n_nodes()`, `n_cells()`, `coord_system()`) with `@return` and explicit types.
  - Document all public field_view accessors with `@return` and explicit types.
  - Document all internal Kokkos::View accessors (`node_coords_view()`, `conn_offsets_view()`, etc.) with `@return` and explicit types.
  - Document `compute_areas()` with complete descriptions of what it computes, its algorithmic approach (Girard's theorem, shoelace formula), and behavior on device/host.
- [ ] **Step 3: Compile and verify correctness of headers**

---

### Task 3: Document `libs/axis/include/axis/topology/structured_grid.hpp`

**Files:**
- Modify: `libs/axis/include/axis/topology/structured_grid.hpp`

- [ ] **Step 1: Read and analyze `structured_grid.hpp` thoroughly**
- [ ] **Step 2: Add comprehensive Doxygen comments to `structured_grid.hpp`**
  - Enrich `@file` and `@brief` file-level comments.
  - Document `StructuredGrid` class, including `@tparam MemorySpace`.
  - Document `memory_space` type alias.
  - Document the constructor with detailed `@param` descriptions.
  - Document dimension queries (`ni()`, `nj()`, `coord_system()`) with `@return` blocks.
  - Document coordinate flat-array accessors (`center_lon()`, `center_lat()`, `corner_lon()`, `corner_lat()`) with `@return` blocks.
  - Document `set_corners(...)` with `@param` blocks.
  - Document `to_unstructured()` with `@return` describing the generated `UnstructuredMesh`.
- [ ] **Step 3: Compile and verify correctness of headers**

---

### Task 4: Document `libs/axis/include/axis/topology/mesh_factory.hpp`

**Files:**
- Modify: `libs/axis/include/axis/topology/mesh_factory.hpp`

- [ ] **Step 1: Read and analyze `mesh_factory.hpp` thoroughly**
- [ ] **Step 2: Add comprehensive Doxygen comments to `mesh_factory.hpp`**
  - Enrich `@file` and `@brief` file-level comments.
  - Document `MeshFactory` class with a note that it is static and non-instantiable.
  - Document the deleted constructor `MeshFactory() = delete;`.
  - Enrich `from_descriptor`, `from_named`, and `from_rules` functions with complete `@tparam`, `@param`, `@return`, and `@throw` clauses.
- [ ] **Step 3: Compile and verify correctness of headers**

---

### Task 5: Document `libs/axis/include/axis/topology/named_grid_registry.hpp`

**Files:**
- Modify: `libs/axis/include/axis/topology/named_grid_registry.hpp`

- [ ] **Step 1: Read and analyze `named_grid_registry.hpp` thoroughly**
- [ ] **Step 2: Add comprehensive Doxygen comments to `named_grid_registry.hpp`**
  - Enrich `@file` and `@brief` file-level comments.
  - Document `NamedGridRegistry` class.
  - Document nested `ParsedName` struct and its members (`family`, `number`).
  - Document `parse(...)`, `is_registered(...)`, `generate(...)`, and `registered_families()` with complete `@tparam`, `@param`, `@return`, and `@throw` clauses.
- [ ] **Step 3: Compile and verify correctness of headers**

---

### Task 6: Document `libs/axis/include/axis/topology/projection_builder.hpp`

**Files:**
- Modify: `libs/axis/include/axis/topology/projection_builder.hpp`

- [ ] **Step 1: Read and analyze `projection_builder.hpp` thoroughly**
- [ ] **Step 2: Add comprehensive Doxygen comments to `projection_builder.hpp`**
  - Enrich `@file` and `@brief` file-level comments.
  - Document `ProjectionBuilder` class.
  - Document `build(...)` with complete `@tparam`, `@param`, `@return`, and `@throw` clauses.
- [ ] **Step 3: Compile and verify correctness of headers**

---

### Task 7: Document `libs/axis/include/axis/topology/rule_generator.hpp`

**Files:**
- Modify: `libs/axis/include/axis/topology/rule_generator.hpp`

- [ ] **Step 1: Read and analyze `rule_generator.hpp` thoroughly**
- [ ] **Step 2: Add comprehensive Doxygen comments to `rule_generator.hpp`**
  - Enrich `@file` and `@brief` file-level comments.
  - Document `RuleGenerator` class.
  - Document `generate(...)` with complete `@tparam`, `@param`, `@return`, and `@throw` clauses.
- [ ] **Step 3: Compile and verify correctness of headers**

---

### Task 8: Document `libs/axis/include/axis/topology/gmsh_writer.hpp`

**Files:**
- Modify: `libs/axis/include/axis/topology/gmsh_writer.hpp`

- [ ] **Step 1: Read and analyze `gmsh_writer.hpp` thoroughly**
- [ ] **Step 2: Add comprehensive Doxygen comments to `gmsh_writer.hpp`**
  - Enrich `@file` and `@brief` file-level comments.
  - Document `GmshWriter` class.
  - Document `write(...)` with complete `@tparam`, `@param`, and `@throw` clauses.
- [ ] **Step 3: Compile and verify correctness of headers**
