# Design Specification: AXIS Python Advanced C++ Integration

## 1. Overview
AXIS (Arbitrary eXgrid Interpolation Solver) provides high-performance, stateless spatial remapping for Earth system grids. To maximize value for Python users, we expose additional foundational utilities from AXIS's core C++ engine. This specification details the design for exposing:
1. **GmshWriter**: Exposes native `.msh` visualization exporting.
2. **RuleGenerator**: Exposes rule-based in-memory mesh generation.
3. **GradientReconstructor**: Exposes least-squares spatial gradient reconstruction.

---

## 2. Component Design & Interfaces

### 2.1 GmshWriter
`GmshWriter` enables direct ASCII serialization of an `UnstructuredMesh` to Gmsh `.msh` v2.2 format, handling host-deep-copies internally if the mesh resides on a device space (Kokkos).

* **Binding Interface (`axis_py`)**:
  ```cpp
  m.def("write_gmsh", [](const std::string &filepath, const HostMesh &mesh) {
      axis::topology::GmshWriter::write<Kokkos::HostSpace>(filepath, mesh);
  }, "filepath"_a, "mesh"_a, "Write an UnstructuredMesh to a Gmsh .msh v2.2 ASCII file");
  ```
* **High-Level Python Wrapper (`axis.Mesh`)**:
  ```python
  class Mesh:
      def to_gmsh(self, filepath: str) -> None:
          """Export the mesh to Gmsh ASCII format for visualization."""
          axis_py.write_gmsh(filepath, self)
  ```

---

### 2.2 RuleGenerator
`RuleGenerator` constructs meshes directly from rule-based parameters (RegularLatLon, GaussianRegular, GaussianReduced, and Projected kinds) on device execution spaces using Kokkos parallel patterns.

* **C++ Binding Interface (`axis_py`)**:
  Exposes the rule generator accepting a Python dictionary:
  ```cpp
  m.def("generate_mesh_from_rules", [](const nb::dict &config) -> HostMesh {
      ensure_kokkos();
      axis::ingest::GridRulesParams rules;
      
      std::string kind_str = nb::cast<std::string>(config["kind"]);
      if (kind_str == "RegularLatLon") {
          rules.kind = axis::ingest::GridRulesKind::RegularLatLon;
      } else if (kind_str == "GaussianRegular") {
          rules.kind = axis::ingest::GridRulesKind::GaussianRegular;
      } else if (kind_str == "GaussianReduced") {
          rules.kind = axis::ingest::GridRulesKind::GaussianReduced;
      } else if (kind_str == "Projected") {
          rules.kind = axis::ingest::GridRulesKind::Projected;
      } else {
          throw std::invalid_argument("Unknown GridRulesKind: " + kind_str);
      }

      if (config.contains("bbox_min_x")) rules.bbox_min_x = nb::cast<double>(config["bbox_min_x"]);
      if (config.contains("bbox_max_x")) rules.bbox_max_x = nb::cast<double>(config["bbox_max_x"]);
      if (config.contains("bbox_min_y")) rules.bbox_min_y = nb::cast<double>(config["bbox_min_y"]);
      if (config.contains("bbox_max_y")) rules.bbox_max_y = nb::cast<double>(config["bbox_max_y"]);
      if (config.contains("r_x")) rules.r_x = nb::cast<double>(config["r_x"]);
      if (config.contains("r_y")) rules.r_y = nb::cast<double>(config["r_y"]);
      if (config.contains("gaussian_n")) rules.gaussian_n = nb::cast<axis::index_t>(config["gaussian_n"]);
      if (config.contains("proj_string")) rules.proj_string = nb::cast<std::string>(config["proj_string"]);

      return axis::topology::RuleGenerator::generate<Kokkos::HostSpace>(rules);
  }, "config"_a, "Generate an UnstructuredMesh using abstract mathematical parameters");
  ```
* **High-Level Python Wrapper (`axis.grid`)**:
  ```python
  class RuleGeometry(Geometry):
      def __init__(self, config: dict):
          self.config = config

      def to_mesh(self, method: str | None = None) -> axis_py.Mesh:
          return axis_py.generate_mesh_from_rules(self.config)
  ```

---

### 2.3 GradientReconstructor
`GradientReconstructor` computes a cell-centered spatial gradient vector field from scalar cell values via least-squares fitting over face-adjacent CSR neighbor offsets and indices.

* **C++ Binding Interface (`axis_py`)**:
  ```cpp
  m.def(
      "reconstruct_gradient",
      [](nb::ndarray<const double, nb::ndim<1>> cell_values,
         nb::ndarray<const double, nb::ndim<2>> centroids,
         nb::ndarray<const axis::index_t, nb::ndim<1>> adj_offsets,
         nb::ndarray<const axis::index_t, nb::ndim<1>> adj_indices,
         bool use_limiter) -> nb::ndarray<nb::numpy, double> {
          ensure_kokkos();

          std::size_t n_cells = cell_values.shape(0);
          if (centroids.shape(0) != n_cells || centroids.shape(1) != 3) {
              throw std::invalid_argument("centroids shape must be (n_cells, 3)");
          }
          if (adj_offsets.shape(0) != n_cells + 1) {
              throw std::invalid_argument("adj_offsets shape must be (n_cells + 1)");
          }

          auto out_grad_uniq = std::make_unique<double[]>(n_cells * 3);
          double *out_ptr = out_grad_uniq.get();

          Kokkos::View<const double *, Kokkos::HostSpace> val_view(cell_values.data(), n_cells);
          Kokkos::View<const double *[3], Kokkos::HostSpace> cent_view(centroids.data(), n_cells);
          Kokkos::View<const axis::index_t *, Kokkos::HostSpace> off_view(adj_offsets.data(), n_cells + 1);
          Kokkos::View<const axis::index_t *, Kokkos::HostSpace> ind_view(adj_indices.data(), adj_indices.shape(0));
          Kokkos::View<double *[3], Kokkos::HostSpace> grad_view(out_ptr, n_cells);

          axis::solver::GradientReconstructor<Kokkos::HostSpace>::compute(
              val_view, cent_view, off_view, ind_view, grad_view, use_limiter
          );

          double *raw_ptr = out_grad_uniq.release();
          nb::capsule owner(raw_ptr, [](void *p) noexcept { delete[] static_cast<double *>(p); });
          std::size_t shape[2] = {n_cells, 3};
          return nb::ndarray<nb::numpy, double>(raw_ptr, 2, shape, std::move(owner));
      },
      "cell_values"_a, "centroids"_a, "adj_offsets"_a, "adj_indices"_a, "use_limiter"_a = false,
      "Reconstruct cell-centered linear gradients via least-squares over CSR neighbors"
  );
  ```

---

## 3. Verification Plan
To ensure absolutely robust integration, we will write:
1. **`test_gmsh_writer.py`**: Asserts that `to_gmsh()` exports to a valid Gmsh file and that file is generated successfully.
2. **`test_rule_generator.py`**: Generates a standard RegularLatLon mesh via `RuleGeometry` and verifies the node/cell counts match analytical expectations.
3. **`test_gradient_reconstructor.py`**: Computes gradients for a linear field ($f(x,y,z) = 2x + 3y + 4z$) and verifies that reconstructed gradients match $[2, 3, 4]$ exactly within machine epsilon.
