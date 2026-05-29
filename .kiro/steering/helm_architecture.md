# MISSION: The HELM Ecosystem
You are building HELM (Heterogeneous Execution & Load-balancing Manager), a modern C++20 replacement for legacy ESMF. You are NOT writing a framework; you are writing a decentralized toolkit of stateless micro-libraries.

## THE VOCABULARY (Strict Naming Convention)
When generating code, classes, or namespaces, you must strictly adhere to the following ecosystem components:
* **DAGR:** (Tier 3) The YAML-driven Directed Acyclic Graph Orchestrator.
* **SPAN:** (Tier 2) The Interface layer mapping Fortran pointers to C++.
* **AXIS:** (Tier 1) The Math and Grid Interpolation engine.
* **HALO:** (Tier 1) The hardware-abstracted MPI communication layer.
* **TICK:** (Tier 1) The stateless time and calendar manager.
* **LOGS:** (Tier 1) The global error handler and status synchronizer.
* **AMIO:** (Tier 1b) The asynchronous background I/O writer.

## THE UNBREAKABLE LAWS OF PHYSICS
1. **Zero-Copy Memory:** You must NEVER copy data arrays from Fortran domain models. All interfaces in SPAN must use `std::mdspan` (C++20) to create non-owning views of native Fortran memory.
2. **Hardware Portability:** You must NEVER write raw CUDA, HIP, or OpenMP. All parallel loops and GPU offloading in AXIS and HALO must be written using the Kokkos programming model.
3. **RAII MPI:** Legacy C-style MPI handles (`MPI_Comm`, `MPI_Request`) must be immediately wrapped in modern C++ RAII objects. Destructors must handle `MPI_Comm_free` to prevent zombie communicators.
4. **No Circular Dependencies:** The Tier 1 utilities (TICK, LOGS, HALO, AXIS) are completely blind to each other and to the domain science. Do not include domain-specific headers inside Tier 1.