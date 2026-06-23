# HELM: High-Performance Earth System Library Modules

### Master Architectural Steering & Execution Document

---

## 1. Executive Summary

The HELM initiative is a complete paradigm shift in Earth System Modeling (ESM) infrastructure. It is designed to replace aging, stateful monolithic frameworks (e.g., ESMF/NUOPC) with a stateless, highly decoupled micro-library ecosystem built natively for C++20 and Kokkos.

HELM guarantees zero-copy memory operations, asynchronous graph-based task execution, and 100% hardware saturation on both Exascale supercomputers and elastic cloud networks. It achieves this by strictly enforcing the Single Responsibility Principle: orchestrators do no math, memory bridges hold no data, and I/O tools manage no calendars.

Our Minimum Viable Product (MVP) is the **Community Emissions Computing Engine (CECE)**, utilizing HELM to operate both as a blazing-fast standalone driver for cloud research and as a "Trojan Horse" embedded within the legacy operational Unified Forecast System (UFS).

---

## 2. The Micro-Library Ecosystem

The framework is stratified into three distinct operational tiers and a generic utility layer.

### Tier 3: Orchestration & Control

* **DAGR (Directed Acyclic Graph Router):** The master orchestrator. DAGR abolishes sequential time loops, replacing them with an asynchronous, event-driven task graph. It interacts with components purely via Basic Model Interface (BMI) calls. It unrolls multi-rate timelines, actively hijacks idle MPI ranks to execute background framework tasks, and routes pointers. **DAGR performs zero mathematical or calendar calculations.**

### Tier 2: The Memory Bridge

* **SPAN (Shared Pointer & Array Network):** The universal, zero-copy data ledger. SPAN intercepts native Fortran, C, or Kokkos pointers and wraps them in lightweight `std::experimental::mdspan` views.
* *Coherency Tracker ("Dirty Bit"):* Tracks `HOST_CLEAN`, `HOST_DIRTY`, and `DEVICE_DIRTY` states, executing targeted `Kokkos::deep_copy` operations only when a CPU/GPU hardware domain mismatch occurs.
* *Triple-Buffer Isolation:* Manages Write, Read, and I/O pointer tracks to prevent data races during asynchronous cloud exports.



### Tier 1: Execution Engines (The "Mechanics")

* **AXIS (Advanced eXchange & Interpolation System):** The spatial math engine. Employs a two-phase execution: Phase 1 dynamically searches and permanently caches unstructured grid intersections as `KokkosSparse::CrsMatrix` weights. Phase 2 executes microsecond-fast Sparse Matrix-Vector Multiplications (SpMV) on GPU Tensor Cores.
* **AMIO (Asynchronous Multidimensional I/O):** The pure byte-fetcher. Streams compressed Zarr datasets directly to/from cloud object storage (S3) or Lustre via background threads. AMIO performs no data manipulation.
* **TICK (Time Integration & Chronology Kernel):** The master clock and temporal translator. Uses zero-drift fixed-point integer math to govern synthetic ESM calendars. It houses the **Time Aliasing Engine** to handle out-of-bounds dates, dummy-year climatologies, and leap-holds without polluting other modules.
* **HALO:** The network layer. Utilizes asynchronous MPI (or Libfabric for cloud) to execute non-blocking spatial boundary exchanges over DAGR's dynamically reallocated CPU pools.

### The Utility Layer

* **BLEND (Stateless Math Kernels):** A generic Kokkos math utility. Exists as an independent node in DAGR’s graph to perform high-speed array mathematics (e.g., linear blending, step-functions) on `SPAN` views.

---

## 3. Core Operational Pipelines

### A. The "Temporal Bookend" Pipeline (Handling I/O-Bound Data)

To process massive input datasets without bottlenecking the GPU, HELM distributes the interpolation workload perfectly across its specialized micro-libraries:

1. **The Fetch:** DAGR commands AMIO to load two discrete data snapshots (Left and Right Bookends) from disk into `SPAN` views.
2. **The Spatial Map:** DAGR routes both bookends through AXIS, mapping the data to the target component's grid via cached SpMV weights.
3. **The Time Math:** DAGR queries TICK for the temporal weight ($\alpha$) between the bookends based on the master simulation time and active Time Aliasing policies.
4. **The Interpolation:** DAGR passes the regridded bookends and the $\alpha$ weight to the BLEND utility, which executes a fused Kokkos `axpy` kernel on the GPU.
5. **The Delivery:** The target model receives a single, perfectly interpolated array.

### B. Fluid CPU Sharing & Rank Hijacking

In legacy systems, processors sit idle waiting for slower components to finish. In HELM, DAGR treats MPI ranks dynamically. When a fast component (e.g., Atmosphere) finishes its step, DAGR intercepts its MPI sub-communicator and immediately commands those processors to execute background framework tasks—such as AXIS regridding matrices or HALO boundary synchronizations—ensuring zero wasted core-hours.

---

## 4. Phase 1 Execution: The CECE Minimum Viable Product (MVP)

CECE (Community Emissions Computing Engine) is our proving ground. Currently trapped by ESMF internal dependencies, HELM will completely untether it.

* **The Standalone Driver Architecture:** We will use DAGR to build a lightweight, C++20 standalone driver for CECE.
* **Gutting the Legacy:** CECE’s internal ESMF regridding and I/O calls will be completely stripped out. CECE will compute on native `Kokkos::View` arrays, which SPAN will expose to AXIS for online regridding and AMIO for asynchronous Zarr output.
* **The Win:** CECE transforms into a highly composable micro-service capable of running natively inside a lightweight Docker container on a single cloud GPU, completely free of heavy legacy framework overhead.

---

## 5. Phase 2 Execution: The "Sandwich" Migration Strategy

To deploy CECE operationally into the Unified Forecast System (UFS) without requiring a high-risk, "big bang" rewrite of the entire supercomputer codebase, HELM utilizes the **Trojan Horse NUOPC Cap** pattern.

1. **The Outer Crust:** We write a compliant NUOPC app cap for CECE. To the master UFS framework, it appears as a standard, slow legacy component.
2. **The Intercept:** When UFS calls the `Run` phase, the NUOPC cap intercepts the heavy `ESMF_State` objects, extracts the raw Fortran/C memory pointers, and registers them directly into the **SPAN** ledger.
3. **The Modern Core:** The cap yields control to **DAGR**. DAGR runs CECE, AXIS, and AMIO natively at Exascale speeds inside this protective enclave, writing results directly back into the intercepted raw pointers via zero-copy references.
4. **The Return:** Control is handed back to UFS, which continues its master forecast loop, completely oblivious that its data was just routed through an asynchronous, GPU-accelerated execution graph.

### The Ultimate Fail-Safe

By maintaining this dual-target deployment (Standalone Executable vs. Dynamic Shared Library), the HELM architecture guarantees that the core C++20 modeling codebase remains 100% pristine. When the UFS eventually decommissions ESMF/NUOPC in the future, the Trojan Horse cap is simply deleted, and the component is already natively running on the future-proof HELM architecture.