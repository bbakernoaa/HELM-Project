Project Charter: High-Performance Earth System Library Modules (HELM)
1. Project Purpose and Justification
The Earth System Modeling (ESM) community is currently constrained by legacy, stateful, monolithic frameworks (e.g., ESMF/NUOPC) that were designed for older generations of CPU-bound supercomputers. These frameworks enforce sequential execution, heavy data copying, and rigid dependencies, preventing modern models from fully utilizing GPU-accelerated Exascale hardware and elastic cloud environments.

The HELM project is initiated to build a universal, future-proof infrastructure bridge. By introducing a completely stateless, decoupled micro-library ecosystem written strictly in C++20 and Kokkos, HELM will enable modern components to run natively at hardware speed while providing a safe, backward-compatible integration path for legacy operational systems like the Unified Forecast System (UFS).

2. Vision and Mission
Vision: To establish a perfectly composable, Exascale-ready infrastructure standard where Earth System models operate at 100% hardware saturation without being permanently tethered to any single monolithic orchestrator.

Mission: To engineer a suite of single-responsibility micro-libraries that enforce strict boundaries between memory visibility, execution orchestration, chronology, and mathematical computation, culminating in the seamless deployment of the Community Emissions Computing Engine (CECE) as our Minimum Viable Product (MVP).

3. Project Scope
In Scope:
Engineering the core HELM micro-libraries (DAGR, SPAN, AXIS, AMIO, TICK, HALO, and BLEND) using strict C++20 and Kokkos standards.

Development of the "Temporal Bookend" pipeline for asynchronous, generic I/O and interpolation.

Implementation of the standalone DAGR-driven execution framework for cloud and local deployments.

Development of the "Trojan Horse" NUOPC adapter cap to encapsulate HELM within legacy operational systems.

Full integration of the CECE MVP, gutting its internal ESMF dependencies in favor of HELM natively.

Out of Scope:
Rewriting the core scientific chemistry algorithms or physics routines inside CECE.

Modifying the internal source code of legacy frameworks like ESMF or NUOPC.

Managing components or models outside of the defined CECE MVP scope during Phase 1 and 2.

4. Key Deliverables
Tier 3: Orchestration Layer
DAGR: An asynchronous, event-driven task graph scheduler that routes BMI-compliant component pointers, unrolls multi-rate timelines, and dynamically hijacks idle MPI ranks for framework operations without performing mathematical execution.

Tier 2: Memory Layer
SPAN: A zero-copy memory bridge wrapping raw Fortran/C/Kokkos pointers in std::experimental::mdspan views. Features a "Dirty Bit" Coherency Tracker for localized CPU/GPU syncing and a Triple-Buffer architecture to isolate I/O.

Tier 1: Execution Layer
AXIS: A two-phase spatial math engine for calculating unstructured grid intersections and executing lightning-fast Sparse Matrix-Vector Multiplications (SpMV) on GPU Tensor Cores.

AMIO: A strictly non-mathematical asynchronous multidimensional I/O engine utilizing background threads to stream Zarr datasets.

TICK: A zero-drift integer chronology engine featuring the "Time Aliasing" state machine (leap-holds, cyclic climatologies, out-of-bounds policies) to translate simulation clocks into dataset fetch requests.

HALO: A non-blocking spatial boundary exchange network engine running on dynamically reallocated MPI sub-communicators.

Utility Layer
BLEND: A statically compiled, stateless Kokkos math utility capable of array interpolation (e.g., linear, step-function) entirely agnostic of time or grid definitions.

5. Strategic Milestones
Phase 1: Foundation & The CECE Standalone MVP
Objective: Establish the core HELM micro-libraries and untether CECE from its legacy ESMF framework dependencies.

Deliverable: A lightweight, highly composable CECE micro-service driven entirely by DAGR, capable of executing within a containerized environment on a single cloud GPU.

Phase 2: The "Sandwich" Migration (UFS Integration)
Objective: Prove backward compatibility and operational readiness without a "big bang" framework rewrite.

Deliverable: The C++ NUOPC cap for CECE. This adapter will intercept UFS master loop calls, translate ESMF_State payloads into raw SPAN pointers, delegate execution to the internal DAGR graph, and seamlessly return Exascale-computed data back to the legacy pipeline.

Phase 3: Hardware Tuning & Production Profiling
Objective: Ensure maximum utilization of target Exascale computing centers.

Deliverable: Profiling reports validating zero-copy pointer exchanges across the Fortran-C++ boundary, successful temporal bookend executions with AMIO/AXIS/BLEND, and verified active rank-hijacking via DAGR.

6. Success Criteria
Purity of Science Code: The CECE codebase contains absolutely zero ESMF/NUOPC framework-specific #include directives or macro logic.

Zero-Copy Validation: Profiling tools confirm that SPAN successfully routes legacy Fortran array addresses through the GPU without initiating framework-level data buffering.

Hardware Saturation: DAGR proves the ability to execute network exchanges (HALO) or matrix math (AXIS) on background MPI ranks while the main forecast component remains blocked.

Transparent Operation: The host UFS system successfully completes a 24-hour simulation cycle coupled to CECE without detecting the presence of the internal HELM graph architecture.