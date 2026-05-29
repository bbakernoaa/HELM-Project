# Project Charter: The HELM Modernization Initiative

## 1. Executive Summary

The legacy Earth System Modeling Framework (ESMF) has historically functioned as an integrated monolith, tightly blending hardware abstraction, memory management, spatial regridding, and model orchestration. While successful for older generation modeling systems, this monolithic architecture introduces severe bottlenecks to developer velocity, increases the computational footprint, and complicates modern hardware optimization.

The **HELM (Heterogeneous Execution & Load-balancing Manager) Initiative** will deconstruct the monolithic framework into a modern, highly composable toolkit of stateless micro-libraries. By enforcing an Inversion of Control paradigm—where domain models dictate their own memory and the infrastructure simply routes it via zero-copy views—this project will deliver maximum performance, Exascale GPU portability, and a frictionless development environment for domain scientists across the Unified Forecast System (UFS).

---

## 2. Project Goals & Objectives

* **Inversion of Control:** Transform the architecture from an opinionated "framework" into a passive, stateless "library" toolkit. The domain model remains sovereign over its physics and memory.
* **Zero-Copy Memory Layer:** Eliminate the performance overhead of packing and unpacking custom framework containers (like `ESMF_State`) by utilizing native, non-owning C++20 memory views.
* **Hardware Portability:** Abstract node-level parallelization using Exascale standards (Kokkos) to enable immediate GPU support without altering domain science code.
* **Memory Safety:** Wrap legacy C-style OpenMPI handles in modern C++ RAII (Resource Acquisition Is Initialization) patterns to definitively eliminate resource leaks and zombie communicators.
* **Dynamic Load Balancing:** Maximize High-Performance Computing (HPC) resource utilization by using a DAG-based orchestrator to fluidly share hardware between active models, math engines, and asynchronous I/O tasks.

---

## 3. Architectural Target State: The HELM Ecosystem

The modernized landscape is strictly divided into a multi-tiered structure to enforce composability. Models interact with the framework entirely through thin, stateless interfaces.

| **Tier** | **Component Type** | **The HELM Component** | **Primary Function** |
| --- | --- | --- | --- |
| **Tier 3** | The Orchestrator | **DAGR** (Directed Acyclic Graph Router) | A YAML-driven application that manages dependencies and dynamically schedules processor pools to optimize load balancing. |
| **Tier 2** | Interface Layer | **SPAN** (Shared Pointer & Array Network) | The C-API boundary. Relies entirely on `std::mdspan` to map Fortran arrays to the math engine without copying data. |
| **Tier 1** | Math & Hardware | **AXIS** (Arbitrary eXgrid Interpolation Solver)<br>

<br>**HALO** (Hardware-Abstracted Link Operations) | *AXIS:* The spatial interpolation engine executed via Kokkos Tensor Cores.<br>

<br>*HALO:* The RAII-protected MPI abstraction for halo exchanges. |
| **Tier 1** | Core State | **TICK** (Time Integration & Chronology Kernel)<br>

<br>**LOGS** (Lightweight Operational Global Status) | *TICK:* Stateless, zero-dependency time and calendar manager.<br>

<br>*LOGS:* High-performance, synchronized error handler. |
| **Tier 1b** | Elective Utilities | **AMIO** (Fast Large-data Object Writer) | Optional, asynchronous background utility for non-blocking NetCDF/Zarr outputs. |

---

## 4. The Migration Strategy: The "Sandwich" Approach

To ensure the daily operational forecasts of the UFS are not disrupted, HELM will not rely on a "Big Bang" release. Instead, migration will be executed using the **Strangler Fig Pattern**, specifically through a "Sandwich Strategy."

1. **Build the Foundation (Tier 1):** The isolated `TICK`, `LOGS`, `HALO`, and `AXIS` libraries are built first in isolated C++20 repositories to mathematically prove their Exascale readiness.
2. **Build the Brain (Tier 3):** The `DAGR` orchestrator is scaffolded to parse dependencies.
3. **The Adapter Bridge (The Temporary Middle):** Rather than forcing UFS models (like Atmosphere and CECE) to immediately rewrite their legacy `NUOPC` caps, a temporary Adapter is built. This Adapter dynamically translates legacy `ESMF_State` objects into `SPAN` zero-copy views on the fly, keeping operations running while the infrastructure swaps beneath them.
4. **Component Strangulation:** One by one, models will deprecate their thick NUOPC caps and adopt the native, ultra-thin `SPAN` interface.

---

## 5. Architecture Stress Test: Execution Walkthrough

Using a coupled UFS Atmosphere (ATM) and CECE integration as the baseline pilot:

1. **The Clock Strikes (`TICK`):** The `DAGR` Orchestrator queries `TICK`. It triggers the ATM model to compute based on the YAML scheduling.
2. **The Physics & Hardware (`HALO`):** ATM runs its fluid dynamics. When boundary updates are needed, `HALO` uses Kokkos to execute an asynchronous, CUDA-aware MPI halo transfer.
3. **The Zero-Copy Handoff (`SPAN`):** ATM passes a non-owning `SPAN` memory view (`std::mdspan`) of its surface variables to the Orchestrator, then yields its processors.
4. **Dynamic Load Balancing (`AXIS`):** `DAGR` "borrows" the idle ATM CPUs, passing the `SPAN` views to `AXIS` to natively execute the sparse-matrix interpolations for the CECE boundary.
5. **The Fork in the Road (`AMIO`):** `DAGR` forks execution, handing a memory view to the elective `AMIO` library, which offloads the NetCDF/Zarr disk write to a background thread pool.
6. **The Cycle Continues (`LOGS`):** `DAGR` wakes the CECE Model and hands it the interpolated view. If errors occur anywhere, `LOGS` safely synchronizes the stack trace across nodes without filesystem bottlenecking.

---

## 6. Strategic & Scientific Impacts

* **Native Machine Learning (AI/ML) Integration:** Legacy frameworks make coupling Python-based AI emulators to Fortran physics highly restrictive. HELM's flat `SPAN` C-API allows Python wrappers (via PyBind11) to be integrated trivially. AI emulators drop into `DAGR` as standard compute nodes.
* **Bit-for-Bit (B4B) Reproducibility:** Because the micro-libraries are fundamentally decoupled, Continuous Integration (CI) pipelines can test the `AXIS` math engine in complete isolation, mathematically proving conservation laws across millions of permutations in minutes.
* **Cloud-Native Readiness:** By abstracting I/O into the asynchronous `AMIO` utility, HELM is intrinsically cloud-native. Background threads can write directly to cloud-optimized object storage (like Zarr on AWS S3) in parallel without slowing the forecast clock.

---

## 7. Implementation Phases

| **Phase** | **Milestone** | **Deliverables** |
| --- | --- | --- |
| **Phase A** | The Foundation | Spin up independent greenfield repositories for Tier 1 (`TICK`, `LOGS`, `HALO`). Enforce strict C++20, Kokkos, and RAII compliance. |
| **Phase B** | The Bridge | Scaffold `DAGR` Orchestrator. Develop the Legacy NUOPC Adapter to bridge legacy `ESMF_State` objects into zero-copy `SPAN` views. |
| **Phase C** | Pilot Integration | Execute the 3-Step Pilot strategy with the ATM and CECE models (Standalone Tests $\rightarrow$ Data Stub Coupling $\rightarrow$ Active 2-Way Coupling). |
| **Phase D** | Migration | Component-by-component removal of legacy NUOPC caps across the UFS, finalizing the adoption of native `SPAN` interfaces. |