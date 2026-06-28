# AXIS Topology Doxygen Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add comprehensive, complete, and descriptive Doxygen comments to three AXIS Topology public headers.

**Architecture:** We will inject standard `///` Doxygen comment blocks into the designated public headers, covering file-level headers, templates, classes, and all public members. No functional code will be altered.

**Tech Stack:** C++20, Doxygen.

## Global Constraints

- Use standard `///` Doxygen blocks.
- Add file-level comments (`@file`, `@brief`).
- Document all classes, structs, enums, values, public functions, and templates.
- Include `@tparam`, `@param`, `@return`, and `@throw` where appropriate, using complete sentences and explicit types.
- Maintain existing C++20 formatting and compile-readiness. No functional logic should be altered.

---

### Task 1: Document `unstructured_mesh.hpp`

**Files:**
- Modify: `libs/axis/include/axis/topology/unstructured_mesh.hpp`

**Interfaces:**
- Consumes: Existing declaration of `UnstructuredMesh` class and its members.
- Produces: Fully documented `UnstructuredMesh` class and its members.

- [ ] **Step 1: Write Doxygen comments for `unstructured_mesh.hpp`**

Review and document:
* File-level `@file` and `@brief`
* Class template `UnstructuredMesh` and its template parameter `MemorySpace` (`@tparam`)
* Type alias `memory_space`
* Constructors, including parameters (`@param`)
* All public member functions (`n_nodes()`, `n_cells()`, `coord_system()`, `node_coords()`, `conn_offsets()`, `conn_indices()`, `cell_areas()`, `cell_mask()`, `node_coords_view()`, `conn_offsets_view()`, `conn_indices_view()`, `cell_areas_view()`, `cell_mask_view()`, `compute_areas()`), documenting parameters, return values (`@return`), and potential exceptions/noexcept.

- [ ] **Step 2: Verify compiling**

Run a build/check script or compile the libraries to make sure no syntax errors were introduced.

- [ ] **Step 3: Commit changes**

### Task 2: Document `structured_grid.hpp`

**Files:**
- Modify: `libs/axis/include/axis/topology/structured_grid.hpp`

**Interfaces:**
- Consumes: Existing declaration of `StructuredGrid` class and its members.
- Produces: Fully documented `StructuredGrid` class and its members.

- [ ] **Step 1: Write Doxygen comments for `structured_grid.hpp`**

Review and document:
* File-level `@file` and `@brief`
* Class template `StructuredGrid` and its template parameter `MemorySpace` (`@tparam`)
* Constructors, including parameters (`@param`)
* All public member functions (`ni()`, `nj()`, `coord_system()`, `center_lon()`, `center_lat()`, `corner_lon()`, `corner_lat()`, `set_corners()`, `to_unstructured()`), documenting parameters, return values (`@return`), and exceptions.

- [ ] **Step 2: Verify compiling**

Run a build/check script or compile the libraries to make sure no syntax errors were introduced.

- [ ] **Step 3: Commit changes**

### Task 3: Document `mesh_factory.hpp`

**Files:**
- Modify: `libs/axis/include/axis/topology/mesh_factory.hpp`

**Interfaces:**
- Consumes: Existing declaration of `MeshFactory` class and its members.
- Produces: Fully documented `MeshFactory` class and its members.

- [ ] **Step 1: Write Doxygen comments for `mesh_factory.hpp`**

Review and document:
* File-level `@file` and `@brief`
* Class `MeshFactory`
* All public static template functions (`from_descriptor()`, `from_named()`, `from_rules()`), documenting template parameter `MemorySpace` (`@tparam`), parameters (`@param`), return values (`@return`), and exceptions (`@throws` or `@throw`).

- [ ] **Step 2: Verify compiling**

Run a build/check script or compile the libraries to make sure no syntax errors were introduced.

- [ ] **Step 3: Commit changes**
