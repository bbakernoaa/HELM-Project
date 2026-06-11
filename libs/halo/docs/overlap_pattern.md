# Communication/Computation Overlap Pattern

This guide describes how to overlap interior computation with halo communication
using HALO's asynchronous structured exchange API. This is the primary
technique for hiding MPI latency in structured-grid NWP models that perform
thousands of halo exchanges per timestep.

> **Requirements covered:** 8.1, 8.2, 8.3

---

## The Interior/Halo Decomposition

A structured grid field can be decomposed into two non-overlapping regions:

```
┌─────────────────────────────────┐
│  halo (ghost zone)              │
│  ┌───────────────────────────┐  │
│  │                           │  │
│  │   interior                │  │
│  │   (no neighbor dependency)│  │
│  │                           │  │
│  └───────────────────────────┘  │
│                                 │
└─────────────────────────────────┘
```

- **Interior**: Grid points whose stencil depends only on owned data. These
  can be computed immediately without waiting for neighbor communication.
- **Halo-dependent boundary**: Grid points whose stencil touches ghost zones
  filled by neighbors. These must wait until the exchange completes.

For a grid with extent `N` and halo width `h`, the interior spans indices
`[2*h, N-2*h)` in each dimension (the stencil needs `h` valid points inward
from the halo boundary).

---

## The Async Overlap Pattern

The pattern has four phases:

1. **Initiate async exchange** — pack halo regions and post MPI sends/receives
2. **Compute interior** — process grid points that don't depend on ghost data
3. **Wait for exchange** — block until all communication completes and unpack
4. **Compute boundary** — process grid points that depend on the fresh halo data

```cpp
#include <Kokkos_Core.hpp>
#include <halo/exchange_structured.hpp>
#include <halo/structured_halo_plan.hpp>

// Assume plan and field are already set up (see below for full example)
// plan: Structured_Halo_Plan<3> for a 3D periodic grid
// field: Kokkos::View<double***, Kokkos::CudaSpace> with halo zones

// ─── Phase 1: Initiate async exchange ───────────────────────────────────
auto handle = halo::exchange_structured_async(plan, field);

// ─── Phase 2: Compute the interior (no halo dependency) ─────────────────
// The interior is the region [2*h, Nx-2*h) x [2*h, Ny-2*h) x [2*h, Nz-2*h)
// where h is the halo width. This region's stencil stays within owned data.
compute_interior(field, halo_width);

// ─── Phase 3: Wait for exchange completion ──────────────────────────────
handle.wait();  // blocks until all MPI completes, then unpacks recv buffers

// ─── Phase 4: Compute the halo-dependent boundary ───────────────────────
// Now ghost zones are valid — process boundary points whose stencil touches them.
compute_boundary(field, halo_width);
```

---

## Complete Example: 3D Diffusion Timestep with Overlap

```cpp
#include <Kokkos_Core.hpp>
#include <halo/communicator.hpp>
#include <halo/environment.hpp>
#include <halo/exchange_structured.hpp>
#include <halo/structured_halo_plan.hpp>

/// Apply a 7-point Laplacian stencil to interior points only.
/// Interior: indices [2*h, N-2*h) per dimension — the stencil (width h)
/// reads only from owned interior + inner-halo data, never from ghost zones.
template <typename ViewType>
void diffuse_interior(const ViewType& src, ViewType& dst,
                      double alpha, std::size_t h) {
    const std::size_t nx = src.extent(0);
    const std::size_t ny = src.extent(1);
    const std::size_t nz = src.extent(2);

    // Interior bounds: stencil of width h needs h valid neighbors,
    // and the first h points are ghost zone, so interior starts at 2*h.
    const std::size_t ilo = 2 * h, ihi = nx - 2 * h;
    const std::size_t jlo = 2 * h, jhi = ny - 2 * h;
    const std::size_t klo = 2 * h, khi = nz - 2 * h;

    using exec_space = typename ViewType::execution_space;
    using policy_t = Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>;

    Kokkos::parallel_for("diffuse_interior",
        policy_t({ilo, jlo, klo}, {ihi, jhi, khi}),
        KOKKOS_LAMBDA(std::size_t i, std::size_t j, std::size_t k) {
            dst(i, j, k) = src(i, j, k) + alpha * (
                src(i-1,j,k) + src(i+1,j,k) +
                src(i,j-1,k) + src(i,j+1,k) +
                src(i,j,k-1) + src(i,j,k+1) - 6.0 * src(i,j,k));
        });
}

/// Apply the same stencil to boundary points (those whose stencil
/// touches ghost zones filled by the halo exchange).
template <typename ViewType>
void diffuse_boundary(const ViewType& src, ViewType& dst,
                      double alpha, std::size_t h) {
    const std::size_t nx = src.extent(0);
    const std::size_t ny = src.extent(1);
    const std::size_t nz = src.extent(2);

    // Boundary = [h, 2*h) and [N-2*h, N-h) per dimension (the owned
    // points adjacent to ghost zones).
    using exec_space = typename ViewType::execution_space;
    using policy_t = Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<3>>;

    // Full owned region is [h, N-h); interior is [2*h, N-2*h).
    // Boundary = owned \ interior. We process it in 6 slabs (faces).
    // For brevity, here we just re-process the full owned region excluding
    // interior — the stencil is valid because ghost zones are now filled.
    // In production, you'd partition into exactly the boundary slabs.
    Kokkos::parallel_for("diffuse_boundary",
        policy_t({h, h, h}, {nx - h, ny - h, nz - h}),
        KOKKOS_LAMBDA(std::size_t i, std::size_t j, std::size_t k) {
            // Skip interior points (already computed)
            if (i >= 2*h && i < nx - 2*h &&
                j >= 2*h && j < ny - 2*h &&
                k >= 2*h && k < nz - 2*h) {
                return;
            }
            dst(i, j, k) = src(i, j, k) + alpha * (
                src(i-1,j,k) + src(i+1,j,k) +
                src(i,j-1,k) + src(i,j+1,k) +
                src(i,j,k-1) + src(i,j,k+1) - 6.0 * src(i,j,k));
        });
}

void run_timestep(halo::Structured_Halo_Plan<3>& plan,
                  Kokkos::View<double***, Kokkos::HostSpace>& field,
                  Kokkos::View<double***, Kokkos::HostSpace>& scratch,
                  double alpha, std::size_t halo_width, int num_steps) {

    for (int step = 0; step < num_steps; ++step) {
        // Phase 1: Initiate async halo exchange on 'field'
        auto handle = halo::exchange_structured_async(plan, field);

        // Phase 2: Compute interior while communication proceeds
        diffuse_interior(field, scratch, alpha, halo_width);

        // Phase 3: Wait for halo exchange to complete
        handle.wait();

        // Phase 4: Compute boundary (stencil now has valid ghost data)
        diffuse_boundary(field, scratch, alpha, halo_width);

        // Swap for next timestep
        auto tmp = field;
        field = scratch;
        scratch = tmp;
    }
}
```

---

## Buffer Lifetime Requirements

**Critical rule:** Between `exchange_structured_async()` and `handle.wait()`,
you **must not write** to the halo (ghost) regions of the view.

The async exchange works as follows:

1. `exchange_structured_async(plan, field)` — packs the *send* halo regions
   into internal contiguous buffers, posts `MPI_Irecv` into internal receive
   buffers, and posts `MPI_Isend` from the packed send buffers.
2. `handle.wait()` — waits for MPI completion, then **unpacks** receive
   buffers back into the view's ghost zones.

### What is safe during the async window

| Operation | Safe? | Reason |
|-----------|-------|--------|
| Read interior points of `field` | ✅ Yes | Interior is independent of ghost zones |
| Write interior points of `field` | ✅ Yes | Interior does not overlap send/recv regions |
| Read ghost zones of `field` | ⚠️ Stale | Ghost data is not yet updated (still holds old values) |
| Write ghost zones of `field` | ❌ **No** | Would corrupt data that `wait()` will overwrite during unpack |
| Read/write a **different** view | ✅ Yes | No aliasing with exchange buffers |
| Destroy `field` | ❌ **No** | Internal lambda holds a reference for unpack |

### Memory ownership diagram

```
Time ──────────────────────────────────────────────────────────►

  exchange_async()          handle.wait()
       │                         │
       ▼                         ▼
  ┌────────────────────────────────┐
  │  send buffers live (internal)  │  ← freed after wait()
  └────────────────────────────────┘
  ┌────────────────────────────────┐
  │  recv buffers live (internal)  │  ← unpacked into view, then freed
  └────────────────────────────────┘
  ┌────────────────────────────────────────────────────────────┐
  │  field view must remain valid and halo regions unmodified   │
  └────────────────────────────────────────────────────────────┘
       │◄─── async window ────►│
       │  ✅ interior compute   │
       │  ❌ halo writes        │
```

### Internal buffer management

HALO manages all MPI buffers internally. The send and receive buffers are
allocated as `Kokkos::View` objects captured by the completion handle via
`std::shared_ptr`. They are automatically freed when `handle.wait()` completes
(or when the handle is destroyed). You do not need to manage any staging
buffers yourself.

---

## Plan Reuse Across Timesteps

The `Structured_Halo_Plan` is designed for the **bind-once, exchange-many**
pattern. Create the plan once during initialization and reuse it every timestep:

```cpp
// During initialization (once)
halo::Structured_Halo_Plan<3> plan(
    /*global_extents=*/ {nx, ny, nz},
    /*neighbor_ranks=*/ {west, east, south, north, bottom, top},
    /*halo_widths=*/    {h, h, h},
    comm
);

// Every timestep (thousands of times)
for (int step = 0; step < num_steps; ++step) {
    auto handle = halo::exchange_structured_async(plan, field);
    compute_interior(field, h);
    handle.wait();
    compute_boundary(field, h);
}
```

The plan stores precomputed subview index ranges. Reusing it eliminates
per-exchange setup overhead. Different view instances of the **same shape**
can share the same plan.

---

## Performance Considerations

1. **Interior/boundary ratio**: The overlap pattern is most effective when the
   interior is large relative to the boundary. For a local grid of `N³` with
   halo width `h`, the interior fraction is approximately `((N-4h)/N)³`.

2. **GPU kernel overlap**: On GPU, the interior compute kernel runs on the
   default stream. The MPI library handles data transfers independently.
   Ensure the interior kernel is large enough to hide the communication
   latency.

3. **Persistent communication**: For repeated exchanges with the same view
   (same buffer addresses), consider `Persistent_Halo_Handle` which eliminates
   per-call `MPI_Isend`/`MPI_Irecv` setup overhead:

   ```cpp
   // Bind once
   halo::Persistent_Halo_Handle persistent(plan, field);

   // Every timestep
   for (int step = 0; step < num_steps; ++step) {
       persistent.start();        // MPI_Startall (no buffer re-registration)
       compute_interior(field, h);
       persistent.wait();         // MPI_Waitall + unpack
       compute_boundary(field, h);
   }
   ```

4. **Non-blocking test**: Use `handle.test()` for polling-based overlap where
   you want to check for completion without blocking:

   ```cpp
   auto handle = halo::exchange_structured_async(plan, field);
   compute_interior_part1(field, h);
   if (!handle.test()) {
       compute_interior_part2(field, h);  // more work while waiting
   }
   handle.wait();  // ensure completion
   compute_boundary(field, h);
   ```
