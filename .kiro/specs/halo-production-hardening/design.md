# Design: HALO Production Hardening for NWP

## Overview

This design extends the existing HALO Tier 1 foundation with production-grade capabilities required by operational NWP models. The core addition is a **structured halo exchange** layer that operates on multi-dimensional Kokkos views with GPU-resident pack/unpack kernels, complemented by persistent communication, runtime GPU-MPI detection, instrumentation hooks, and ecosystem tooling (Spack, versioning, adapter examples).

The design preserves all existing invariants:
- RAII resource safety
- Tier 1 isolation (no HELM cross-dependencies)
- Compile-time memory space dispatch
- Thread safety via `Serialized_MPI_Guard`
- The existing flat-buffer exchange API remains unchanged (backward compatible)

---

## Architecture Extension

### New Components

```
halo::
├── (existing API unchanged)
├── Structured_Halo_Plan<Rank>   // Precomputed multi-dim halo geometry
├── Persistent_Halo_Handle       // MPI_Send_init/Recv_init RAII wrapper
├── exchange_structured_blocking  // Multi-dim blocking exchange
├── exchange_structured_async     // Multi-dim async exchange
├── Diagnostics                   // Optional instrumentation callback
├── ErrorPolicy                   // throw vs abort configuration
└── detail::
    ├── pack_kernel<ViewType>     // Kokkos parallel_for pack
    ├── unpack_kernel<ViewType>   // Kokkos parallel_for unpack
    └── gpu_aware_probe()         // Runtime GPU-MPI detection
```

### Header Layout (additions)

```
include/halo/
├── (existing headers unchanged)
├── version.hpp                   // Generated: HALO_VERSION_* macros
├── structured_halo_plan.hpp      // Structured_Halo_Plan<Rank>
├── persistent_halo_handle.hpp    // Persistent communication RAII
├── exchange_structured.hpp       // structured blocking/async templates
├── diagnostics.hpp               // Instrumentation hook interface
├── error_policy.hpp              // ErrorPolicy enum + abort helper
└── detail/
    ├── (existing detail headers unchanged)
    ├── pack_unpack.hpp           // Pack/unpack kernel implementations
    └── gpu_aware_probe.hpp       // MPIX_Query_*_support wrappers
```

---

## Key Design Decisions

### Structured Halo Plan

The `Structured_Halo_Plan` is templated on view rank (1-4) and stores:
- Grid extents per dimension (from the view)
- Halo width per dimension (symmetric or asymmetric: left/right per dim)
- Neighbor map: which face/edge/corner connects to which MPI rank
- Precomputed subview descriptors for each send/recv region

For a 3D field with halo width `h`:
```
Send regions (6 faces for a cube):
  west  : view(0:h, :, :)          → neighbor_west
  east  : view(nx-h:nx, :, :)      → neighbor_east
  south : view(:, 0:h, :)          → neighbor_south
  north : view(:, ny-h:ny, :)      → neighbor_north
  bottom: view(:, :, 0:h)          → neighbor_bottom
  top   : view(:, :, nz-h:nz)      → neighbor_top
```

The plan expresses this generically for any rank via `Kokkos::subview` index ranges, supporting both LayoutLeft and LayoutRight.

### Pack/Unpack Kernels

```cpp
namespace halo::detail {

template <typename SrcView, typename DstView>
void pack(const SrcView& field_subview, DstView& buffer,
          typename SrcView::execution_space exec = {}) {
    const auto n = field_subview.size();
    Kokkos::parallel_for("halo_pack",
        Kokkos::RangePolicy<typename SrcView::execution_space>(exec, 0, n),
        KOKKOS_LAMBDA(int i) {
            buffer(i) = field_subview.data()[i];  // linearized access
        });
    exec.fence();  // ensure pack completes before MPI_Isend
}

template <typename SrcView, typename DstView>
void unpack(const SrcView& buffer, DstView& field_subview,
            typename DstView::execution_space exec = {}) {
    const auto n = buffer.size();
    Kokkos::parallel_for("halo_unpack",
        Kokkos::RangePolicy<typename DstView::execution_space>(exec, 0, n),
        KOKKOS_LAMBDA(int i) {
            field_subview.data()[i] = buffer(i);
        });
    exec.fence();  // ensure unpack completes before computation resumes
}

} // namespace halo::detail
```

For strided subviews (non-contiguous), the linearized `data()[i]` won't work; instead we use multi-dimensional index mapping via `Kokkos::MDRangePolicy`. The design detects contiguity at plan-build time and selects the optimal kernel.

### Persistent Communication

```cpp
class Persistent_Halo_Handle {
public:
    // Bind a plan to a specific view (calls MPI_Send_init/Recv_init)
    template <typename ViewType>
    Persistent_Halo_Handle(const Halo_Plan& plan, ViewType& view);

    ~Persistent_Halo_Handle();  // MPI_Request_free on all persistent reqs

    void start();   // MPI_Startall
    void wait();    // MPI_Waitall (+ post-recv unpack if staged)
    bool test();    // MPI_Testall

    // Move-only
    Persistent_Halo_Handle(Persistent_Halo_Handle&&) noexcept;
    Persistent_Halo_Handle& operator=(Persistent_Halo_Handle&&) noexcept;
    Persistent_Halo_Handle(const Persistent_Halo_Handle&) = delete;
    Persistent_Halo_Handle& operator=(const Persistent_Halo_Handle&) = delete;

private:
    std::vector<MPI_Request> persistent_requests_;
    // ... staging state when needed
};
```

### Runtime GPU-Aware Detection

```cpp
namespace halo::detail {

inline bool gpu_aware_probe() noexcept {
#if defined(KOKKOS_ENABLE_CUDA)
    #if defined(MPIX_CUDA_AWARE_SUPPORT)
        return MPIX_Query_cuda_support() != 0;
    #else
        #if defined(HALO_GPU_AWARE_MPI)
            return true;
        #else
            return false;
        #endif
    #endif
#elif defined(KOKKOS_ENABLE_HIP)
    // Similar for ROCm
#else
    return false;  // no device backend
#endif
}

} // namespace halo::detail
```

This is cached in `Environment::initialize()` and replaces the pure compile-time `requires_staging_v` with a runtime-or-compiletime hybrid.

### Diagnostics Hook

```cpp
namespace halo {

struct Exchange_Event {
    enum class Phase { begin, send_posted, recv_complete, end };
    Phase phase;
    int local_rank;
    int neighbor_count;
    std::size_t total_bytes;
    std::chrono::nanoseconds elapsed;  // since exchange begin
    bool is_async;
};

using Diagnostics_Callback = std::function<void(const Exchange_Event&)>;

class Diagnostics {
public:
    static void set_callback(Diagnostics_Callback cb);
    static void clear_callback();
    // Internal: called by exchange functions
    static void emit(const Exchange_Event& event);
private:
    static inline Diagnostics_Callback callback_{nullptr};
};

} // namespace halo
```

When no callback is registered, `emit()` is a no-op (just a null check — no timing, no allocation).

### Neighbor Collectives

For symmetric neighbor lists, `MPI_Dist_graph_create_adjacent` creates a topology communicator, then `MPI_Neighbor_alltoallv` performs the exchange in a single collective that the MPI implementation can optimize for network topology.

```cpp
// Selectable at plan creation or via a separate function
template <typename ViewType>
void exchange_neighbor_collective(const Structured_Halo_Plan<ViewType::rank>& plan,
                                  ViewType& view);
```

Falls back to the standard Isend/Irecv path when:
- The neighbor list is asymmetric (different send/recv sets)
- The MPI does not support neighborhood collectives (MPI-3 minimum)

---

## Task Execution Waves

| Wave | Tasks | Dependencies |
|------|-------|-------------|
| 0 | Version header, CHANGELOG, error_policy.hpp | None |
| 1 | Pack/unpack kernels, GPU-aware runtime probe | Wave 0 |
| 2 | Structured_Halo_Plan, strided view support | Wave 1 |
| 3 | Structured exchange (blocking + async) | Wave 2 |
| 4 | Persistent communication handle | Wave 1 |
| 5 | Diagnostics hook, enhanced error messages | Wave 0 |
| 6 | MPI_ERRORS_RETURN on owned comms | Wave 0 |
| 7 | Neighbor collectives (optional path) | Wave 2 |
| 8 | Spack package.py | Wave 0 |
| 9 | Documentation (overlap, thread safety) | Wave 3, 4 |
| 10 | FV3/MPAS adapter examples | Wave 3 |
| 11 | Integration tests + CI updates | All above |

</content>
</invoke>