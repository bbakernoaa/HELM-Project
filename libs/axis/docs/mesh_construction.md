# Mesh Construction {#mesh_construction}

AXIS operates on `UnstructuredMesh` objects that represent the source and destination grids.
Meshes can be created from descriptors, named grids, or rule-based generators.

## From GridDescriptor

The primary mesh creation path uses `MeshFactory::from_descriptor()` which accepts a
`GridDescriptor` — a plain-data struct containing coordinates and connectivity arrays.
This is the ingest contract between AMIO (file I/O) and AXIS (interpolation).

```cpp
#include <axis/topology/mesh_factory.hpp>

// Build descriptor from your coordinate arrays
axis::topology::GridDescriptor desc;
desc.cell_lon = lon_view;      // Kokkos::View<const double*>
desc.cell_lat = lat_view;      // Kokkos::View<const double*>
desc.connectivity = conn_view; // Kokkos::View<const index_t*>
desc.offsets = off_view;       // Kokkos::View<const index_t*>
desc.n_cells = n;
desc.coord_system = axis::CoordSystem::SphericalDeg;

auto mesh = axis::topology::MeshFactory::from_descriptor(desc);
```

### Coordinate Systems

AXIS supports the following coordinate systems:

| Enum | Description |
|------|-------------|
| `SphericalDeg` | Longitude/latitude in degrees |
| `SphericalRad` | Longitude/latitude in radians |
| `Cartesian3D` | Unit-sphere Cartesian (x, y, z) |

The coordinate system affects how AXIS interprets vertices internally. All internal
computation operates on unit-sphere Cartesian coordinates; conversion from spherical
is performed at mesh construction time.

### Connectivity Format

Connectivity is stored as a CSR-like structure:
- `offsets[i]` to `offsets[i+1]` spans the vertex indices for cell `i`
- `connectivity[offsets[i] + k]` is the k-th vertex of cell `i`

Vertices must be ordered counter-clockwise when viewed from outside the sphere.

## Named Grids

AXIS includes a registry of standard Earth-system grids:

```cpp
#include <axis/topology/named_grid_registry.hpp>

// Generate an octahedral reduced Gaussian grid
auto mesh = axis::topology::NamedGridRegistry::generate("O48");

// Generate a regular lon-lat grid
auto mesh = axis::topology::NamedGridRegistry::generate("N128");
```

### Available Grid Types

| Pattern | Description | Example |
|---------|-------------|---------|
| `O<N>` | Octahedral reduced Gaussian | O48, O96, O320 |
| `N<N>` | Regular Gaussian | N48, N128, N256 |
| `F<N>` | Full (regular lat-lon) | F90, F180 |
| `R<N>` | Global Rectilinear Normal Lat-Lon | R90, R360 |

## Rule-Based Generation

For custom grids, the rule generator creates meshes from a specification:

```cpp
#include <axis/topology/rule_generator.hpp>

axis::topology::RuleSpec spec;
spec.type = axis::topology::GridType::ReducedGaussian;
spec.parameter = 48;  // Gaussian parameter

auto mesh = axis::topology::RuleGenerator::generate(spec);
```

## Structured Grids

For structured (regular) grids, AXIS provides a fast path that avoids explicit
connectivity storage:

```cpp
#include <axis/topology/structured_grid.hpp>

auto grid = axis::topology::StructuredGrid(n_lon, n_lat, lon_start, lat_start, dlon, dlat);
auto mesh = grid.to_unstructured();
```

## Memory Spaces

All mesh construction functions are templated on `MemorySpace`:

```cpp
// Host mesh (default)
auto host_mesh = axis::topology::MeshFactory::from_descriptor<Kokkos::HostSpace>(desc);

// Device mesh (for GPU weight generation)
auto device_mesh = axis::topology::MeshFactory::from_descriptor<Kokkos::CudaSpace>(desc);
```

## Mesh Queries

Once constructed, meshes expose:

```cpp
mesh.n_cells();        // Total cell count
mesh.n_vertices();     // Total vertex count
mesh.cell_area(i);     // Area of cell i in steradians
mesh.centroid(i);      // Centroid of cell i as Vec3
mesh.vertices(i);      // View of vertices for cell i
```

## Degenerate Cell Detection

AXIS v2 automatically detects and reports degenerate cells during mesh construction
via the `DegenerateCellHandler`. See @ref masking for how degenerate cells interact
with weight generation.
