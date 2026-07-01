# HELM Architecture Overview: The Decoupled Stack

The core philosophy of HELM is the absolute separation of concerns. By dividing the infrastructure into a **Control Plane**, a **Data Plane**, and an **Execution Plane**, HELM ensures that components never become entangled. Orchestrators do not do math, memory managers do not read files, and the framework itself is completely agnostic to the scientific models it drives.

### The HELM Stack Topology

```text
=========================================================
  TIER 3: THE CONTROL PLANE (Orchestration)
                          │
              ┌───────────┴───────────┐
              │         DAGR          │
              │  (Task Graph Router)  │
              └───────────┬───────────┘
                          │ (Routes Control/Events)
==========================│==============================
  TIER 2: THE DATA PLANE (Memory Visibility)
                          │
              ┌───────────┴───────────┐
              │         SPAN          │
              │  (Zero-Copy Ledger)   │
              └───────────┬───────────┘
                          │ (Routes Raw Pointers)
==========================│==============================
  TIER 1 & UTILITIES: THE EXECUTION PLANE (Mechanics)
                          │
      ┌─────────┬─────────┼─────────┬─────────┐
      ▼         ▼         ▼         ▼         ▼
  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐
  │ TICK  │ │ HALO  │ │ AMIO  │ │ AXIS  │ │ BLEND │
  │(Clock)│ │(Comm) │ │ (I/O) │ │(Space)│ │(Math) │
  └───────┘ └───────┘ └───────┘ └───────┘ └───────┘
                          │
==========================│==============================
  THE PAYLOAD (External Scientific Models)
                          │
              ┌───────────▼───────────┐
              │   CECE (Emissions)    │
              │   MOM6 (Ocean)        │
              │   FV3  (Atmosphere)   │
              └───────────────────────┘

```

---

### 1. The Control Plane (Tier 3)

*The pure orchestrator. It dictates* when *things happen and* who *has the hardware resources, but it never touches the data or calculates math.*

* **DAGR (Directed Acyclic Graph Router):** The master conductor. It evaluates the asynchronous execution graph, triggers BMI components, and routes execution tokens. It treats all Tier 1 libraries and External Models equally—as independent nodes in its graph.

### 2. The Data Plane (Tier 2)

*The universal ledger. It tracks where data lives in memory and ensures it is safely accessible across different hardware architectures without framework lock-in.*

* **SPAN (Shared Pointer & Array Network):** The zero-copy bridge. It ingests raw C/Fortran/Kokkos pointers and wraps them in non-owning `std::experimental::mdspan` views.
* **Coherency State Machine:** Tracks `HOST_DIRTY` and `DEVICE_DIRTY` states, acting as the strict gatekeeper that triggers `Kokkos::deep_copy` transfers *only* when necessary.
* **Triple-Buffer I/O Protection:** Secures Write, Read, and I/O tracks to prevent race conditions during asynchronous cloud streaming.



### 3. The Execution Plane (Tier 1 & Utilities)

*The HELM mechanics. These are hyper-optimized, strictly bounded engines that perform exactly one framework job at Exascale speed upon DAGR's command.*

* **TICK (Time Integration & Chronology Kernel):** The master clock and temporal translator. It resolves calendar out-of-bounds rules, leap-holds, and cyclic climatologies to provide exact interpolation weights.
* **HALO:** The network protocol layer. Triggered by DAGR, it reallocates idle CPU ranks to perform asynchronous MPI/Libfabric boundary exchanges in the background.
* **AMIO (Asynchronous Multidimensional I/O):** The byte-fetcher. It operates on background threads, moving compressed Zarr datasets between cloud storage and `SPAN` memory buffers.
* **AXIS (Advanced eXchange & Interpolation System):** The spatial engine. It permanently caches grid intersections as Kokkos Sparse Matrices (`CrsMatrix`) and executes ultra-fast SpMV math on GPU Tensor Cores.
* **BLEND (Utility Math Kernels):** The stateless interpolator. It executes generalized BLAS-style `axpy` math (linear blending, step-functions) to temporally interpolate arrays.

### 4. The Payload (External Models)

*The science. These are 100% independent codes.*

* **Target Models (e.g., CECE, MOM6, FV3):** The model receives a perfectly interpolated, regridded array via a standard BMI `update()` call. It runs its chemistry or physics natively, completely unaware that HELM exists, that the data came from a file, or that temporal blending occurred milliseconds prior.Spot on. If HELM is truly a standalone, framework-agnostic ecosystem, we shouldn't anchor its master architecture diagram to a legacy adapter. DAGR is the top of the pyramid; how the host application decides to boot DAGR is irrelevant to the HELM stack itself.

Here is the finalized, completely independent HELM architecture overview.

---

# HELM Architecture Overview: The Decoupled Stack

The core philosophy of HELM is the absolute separation of concerns. By dividing the infrastructure into a **Control Plane**, a **Data Plane**, and an **Execution Plane**, HELM ensures that components never become entangled. Orchestrators do not do math, memory managers do not read files, and the framework itself is completely agnostic to the scientific models it drives.

### The HELM Stack Topology

```text
=========================================================
  TIER 3: THE CONTROL PLANE (Orchestration)
                          │
              ┌───────────┴───────────┐
              │         DAGR          │
              │  (Task Graph Router)  │
              └───────────┬───────────┘
                          │ (Routes Control/Events)
==========================│==============================
  TIER 2: THE DATA PLANE (Memory Visibility)
                          │
              ┌───────────┴───────────┐
              │         SPAN          │
              │  (Zero-Copy Ledger)   │
              └───────────┬───────────┘
                          │ (Routes Raw Pointers)
==========================│==============================
  TIER 1 & UTILITIES: THE EXECUTION PLANE (Mechanics)
                          │
      ┌─────────┬─────────┼─────────┬─────────┐
      ▼         ▼         ▼         ▼         ▼
  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐
  │ TICK  │ │ HALO  │ │ AMIO  │ │ AXIS  │ │ BLEND │
  │(Clock)│ │(Comm) │ │ (I/O) │ │(Space)│ │(Math) │
  └───────┘ └───────┘ └───────┘ └───────┘ └───────┘
                          │
==========================│==============================
  THE PAYLOAD (External Scientific Models)
                          │
              ┌───────────▼───────────┐
              │   CECE (Emissions)    │
              │   MOM6 (Ocean)        │
              │   FV3  (Atmosphere)   │
              └───────────────────────┘

```

---

### 1. The Control Plane (Tier 3)

*The pure orchestrator. It dictates* when *things happen and* who *has the hardware resources, but it never touches the data or calculates math.*

* **DAGR (Directed Acyclic Graph Router):** The master conductor. It evaluates the asynchronous execution graph, triggers BMI components, and routes execution tokens. It treats all Tier 1 libraries and External Models equally—as independent nodes in its graph.

### 2. The Data Plane (Tier 2)

*The universal ledger. It tracks where data lives in memory and ensures it is safely accessible across different hardware architectures without framework lock-in.*

* **SPAN (Shared Pointer & Array Network):** The zero-copy bridge. It ingests raw C/Fortran/Kokkos pointers and wraps them in non-owning `std::experimental::mdspan` views.
* **Coherency State Machine:** Tracks `HOST_DIRTY` and `DEVICE_DIRTY` states, acting as the strict gatekeeper that triggers `Kokkos::deep_copy` transfers *only* when necessary.
* **Triple-Buffer I/O Protection:** Secures Write, Read, and I/O tracks to prevent race conditions during asynchronous cloud streaming.



### 3. The Execution Plane (Tier 1 & Utilities)

*The HELM mechanics. These are hyper-optimized, strictly bounded engines that perform exactly one framework job at Exascale speed upon DAGR's command.*

* **TICK (Time Integration & Chronology Kernel):** The master clock and temporal translator. It resolves calendar out-of-bounds rules, leap-holds, and cyclic climatologies to provide exact interpolation weights.
* **HALO:** The network protocol layer. Triggered by DAGR, it reallocates idle CPU ranks to perform asynchronous MPI/Libfabric boundary exchanges in the background.
* **AMIO (Asynchronous Multidimensional I/O):** The byte-fetcher. It operates on background threads, moving compressed Zarr datasets between cloud storage and `SPAN` memory buffers.
* **AXIS (Advanced eXchange & Interpolation System):** The spatial engine. It permanently caches grid intersections as Kokkos Sparse Matrices (`CrsMatrix`) and executes ultra-fast SpMV math on GPU Tensor Cores.
* **BLEND (Utility Math Kernels):** The stateless interpolator. It executes generalized BLAS-style `axpy` math (linear blending, step-functions) to temporally interpolate arrays.

### 4. The Payload (External Models)

*The science. These are 100% independent codes.*

* **Target Models (e.g., CECE, MOM6, FV3):** The model receives a perfectly interpolated, regridded array via a standard BMI `update()` call. It runs its chemistry or physics natively, completely unaware that HELM exists, that the data came from a file, or that temporal blending occurred milliseconds prior.
