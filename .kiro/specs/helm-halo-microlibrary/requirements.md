# Requirements Document

## Introduction

HALO (Hardware-Abstracted Link Operations) is a Tier 1 C++20 micro-library within the HELM ecosystem. It provides RAII-wrapped MPI communication primitives with GPU-aware halo exchange capabilities via Kokkos. HALO replaces the legacy ESMF VM class and halo exchange infrastructure with a modern, stateless, zero-copy design that eliminates resource leaks and enables Exascale GPU portability. As a Tier 1 component, HALO is completely blind to other HELM libraries (TICK, LOGS, AXIS) and domain science code.

## Glossary

- **HALO**: Hardware-Abstracted Link Operations; the Tier 1 RAII-protected MPI abstraction micro-library for halo exchanges within the HELM ecosystem.
- **Communicator**: An RAII wrapper around an MPI_Comm handle that manages communicator lifetime via constructor/destructor semantics.
- **Request_Guard**: An RAII wrapper around an MPI_Request handle that ensures non-blocking operations are completed or cancelled on destruction.
- **Window_Guard**: An RAII wrapper around an MPI_Win handle for one-sided RMA operations with automatic lifetime management.
- **Halo_Exchange**: A reusable communication pattern object that precomputes neighbor relationships and executes boundary data transfers between distributed subdomains.
- **Halo_Plan**: A precomputed, immutable description of send/receive neighbor lists and buffer geometry, analogous to the legacy ESMF RouteHandle.
- **Halo_Handle**: An RAII object returned by asynchronous halo exchange operations that owns pending Request_Guard objects and manages completion.
- **Kokkos_View**: A Kokkos multi-dimensional array abstraction that manages memory across host and device (GPU) memory spaces.
- **Memory_Space**: A Kokkos concept representing where data physically resides (e.g., HostSpace, CudaSpace, HIPSpace).
- **Execution_Space**: A Kokkos concept representing where computation runs (e.g., Serial, OpenMP, Cuda, HIP).
- **GPU_Aware_MPI**: An MPI implementation capable of directly sending and receiving device (GPU) pointers without explicit host staging.
- **mdspan**: The C++23/C++20 non-owning multi-dimensional array view (std::mdspan) used for zero-copy interfaces.
- **PET**: Persistent Execution Thread; the legacy ESMF abstraction for an MPI process or OS thread.
- **RAII**: Resource Acquisition Is Initialization; a C++ idiom where resource lifetime is bound to object scope.
- **Sub_Communicator**: A Communicator derived from a parent via splitting or duplication, representing a subset of processes.
- **iso_c_binding**: The Fortran 2003+ intrinsic module that provides interoperability types and attributes for calling C functions from Fortran and vice versa.
- **bind_c**: The Fortran attribute (`bind(c)`) applied to procedures and derived types to make them compatible with C calling conventions and struct layouts.
- **Opaque_Handle**: An integer token exposed to Fortran that maps internally to a C++ object pointer, avoiding direct pointer exposure across the language boundary.
- **halo_mod**: The Fortran module providing the public Fortran API for HALO operations, implemented using iso_c_binding wrappers around the C interop layer.
- **C_Interop_Layer**: The set of `extern "C"` functions in the HALO C++ library that expose a flat C API suitable for consumption by Fortran iso_c_binding modules.
- **NUOPC**: National Unified Operational Prediction Capability; the legacy coupling framework built on ESMF that HELM replaces, whose Fortran models are the primary consumers of the Fortran interface.
- **Fortran_Contiguous_Array**: A Fortran array with the `contiguous` attribute, guaranteeing sequential memory layout compatible with C pointer semantics.

## Requirements

### Requirement 1: RAII Communicator Wrapper

**User Story:** As a library developer, I want MPI communicator handles wrapped in RAII objects, so that communicator resources are automatically freed when they leave scope and zombie communicators are eliminated.

#### Acceptance Criteria

1. WHEN a Communicator is constructed from an MPI_Comm handle, THE Communicator SHALL take exclusive ownership of that handle and store it internally.
2. WHEN a Communicator is destroyed and the owned handle is not MPI_COMM_NULL and is not a predefined communicator (MPI_COMM_WORLD, MPI_COMM_SELF) and MPI has not been finalized, THE Communicator SHALL call MPI_Comm_free on the owned handle.
3. WHEN a Communicator is move-constructed or move-assigned from another Communicator, THE source Communicator SHALL set its internal handle to MPI_COMM_NULL and the destination Communicator SHALL assume ownership of the transferred handle without calling any MPI functions.
4. THE Communicator SHALL delete copy construction and copy assignment operators to enforce unique ownership semantics.
5. WHEN a Communicator exposes its raw MPI_Comm handle via an accessor, THE Communicator SHALL return the handle by value without transferring ownership.
6. IF an exception is thrown during a scope containing a Communicator, THEN THE Communicator destructor SHALL still execute the same destruction logic defined in criterion 2.
7. WHEN a Communicator wraps MPI_COMM_WORLD or MPI_COMM_SELF, THE Communicator SHALL NOT call MPI_Comm_free on destruction because those are predefined communicators.
8. IF a Communicator is constructed from MPI_COMM_NULL, THEN THE Communicator SHALL store MPI_COMM_NULL and SHALL NOT call MPI_Comm_free on destruction.

### Requirement 2: RAII Request Guard

**User Story:** As a library developer, I want non-blocking MPI request handles wrapped in RAII objects, so that pending asynchronous operations are always completed or cancelled and request resources are never leaked.

#### Acceptance Criteria

1. WHEN a Request_Guard is constructed from an MPI_Request handle, THE Request_Guard SHALL take ownership of that handle and store it internally, setting the source variable to MPI_REQUEST_NULL.
2. WHEN a Request_Guard is destroyed outside of stack unwinding and the owned handle is not MPI_REQUEST_NULL, THE Request_Guard SHALL call MPI_Wait to complete the operation and set the internal handle to MPI_REQUEST_NULL.
3. WHEN a Request_Guard is move-constructed or move-assigned, THE source Request_Guard SHALL set its internal handle to MPI_REQUEST_NULL and the destination Request_Guard SHALL assume ownership of the handle without calling any MPI functions.
4. THE Request_Guard SHALL delete copy construction and copy assignment operators to enforce unique ownership semantics.
5. WHEN Request_Guard::test is called and the internal handle is not MPI_REQUEST_NULL, THE Request_Guard SHALL call MPI_Test and return true if the operation completed, false otherwise.
6. WHEN Request_Guard::wait is called and the internal handle is not MPI_REQUEST_NULL, THE Request_Guard SHALL call MPI_Wait and block until the operation completes.
7. IF an exception is unwinding the stack (std::uncaught_exceptions() returns greater than zero at destructor entry) and the owned handle is not MPI_REQUEST_NULL, THEN THE Request_Guard destructor SHALL call MPI_Cancel followed by MPI_Request_free and set the internal handle to MPI_REQUEST_NULL.
8. IF Request_Guard::test or Request_Guard::wait is called when the internal handle is MPI_REQUEST_NULL, THEN THE Request_Guard SHALL return immediately (test returns true, wait is a no-op) without calling any MPI functions.

### Requirement 3: RAII Window Guard

**User Story:** As a library developer, I want MPI window handles wrapped in RAII objects, so that one-sided RMA resources are automatically freed and fence/unlock operations are never missed.

#### Acceptance Criteria

1. WHEN a Window_Guard is constructed from an MPI_Win handle, THE Window_Guard SHALL take exclusive ownership of that handle.
2. WHEN a Window_Guard is destroyed, THE Window_Guard SHALL call MPI_Win_free on the owned handle if the handle is not equal to MPI_WIN_NULL and MPI has not been finalized.
3. WHEN a Window_Guard is move-constructed, THE source Window_Guard SHALL set its internal handle to MPI_WIN_NULL and the destination Window_Guard SHALL assume ownership of the original handle without calling any MPI functions.
4. THE Window_Guard SHALL delete copy construction and copy assignment operators to enforce unique ownership semantics.
5. WHEN Window_Guard exposes its raw MPI_Win handle via an accessor, THE Window_Guard SHALL return the handle by value without transferring ownership.
6. IF an exception is thrown during a scope containing a Window_Guard, THEN THE Window_Guard destructor SHALL still execute MPI_Win_free on the owned handle provided the handle is not MPI_WIN_NULL and MPI has not been finalized.
7. WHEN a Window_Guard is destroyed while an access epoch is active on the owned window, THE Window_Guard SHALL call MPI_Win_fence with assertion 0 to close the epoch before calling MPI_Win_free.
8. IF MPI_Win_free or MPI_Win_fence returns an error code during destruction, THEN THE Window_Guard SHALL not throw an exception and SHALL leave the handle in its current state.

### Requirement 4: Communicator Splitting and Duplication

**User Story:** As a library developer, I want to create sub-communicators by splitting or duplicating existing communicators, so that component groups can operate on isolated communication domains without interfering with each other.

#### Acceptance Criteria

1. WHEN Communicator::split is called with an int color and int key, THE Communicator SHALL call MPI_Comm_split and return a new Communicator owning the resulting sub-communicator.
2. WHEN Communicator::duplicate is called, THE Communicator SHALL call MPI_Comm_dup and return a new Communicator owning the duplicated communicator.
3. WHEN Communicator::rank is called, THE Communicator SHALL call MPI_Comm_rank and return the int rank of the calling process within the communicator context.
4. WHEN Communicator::size is called, THE Communicator SHALL call MPI_Comm_size and return the int total number of processes in the communicator.
5. IF MPI_Comm_split or MPI_Comm_dup returns a non-success error code, THEN THE Communicator SHALL throw a std::runtime_error containing the MPI error string and SHALL NOT modify the state of the original Communicator.
6. IF Communicator::split is called with color equal to MPI_UNDEFINED, THEN THE Communicator SHALL return a Communicator in a valid empty state that holds MPI_COMM_NULL and does not call MPI_Comm_free on destruction.

### Requirement 5: Halo Plan Precomputation

**User Story:** As a domain scientist, I want halo exchange communication patterns precomputed and stored for reuse, so that repeated halo exchanges avoid redundant neighbor discovery and buffer allocation overhead.

#### Acceptance Criteria

1. WHEN a Halo_Plan is constructed with a Communicator reference, neighbor rank lists, and per-neighbor buffer extents expressed in number of elements, THE Halo_Plan SHALL precompute and store the send-neighbor ranks, receive-neighbor ranks, and their corresponding buffer element counts as immutable state that cannot be modified after construction.
2. THE Halo_Plan SHALL store the list of send-neighbor ranks and corresponding send buffer sizes (in number of elements) for each neighbor, preserving the order provided at construction.
3. THE Halo_Plan SHALL store the list of receive-neighbor ranks and corresponding receive buffer sizes (in number of elements) for each neighbor, preserving the order provided at construction.
4. IF any neighbor rank in the send or receive list is negative or greater than or equal to the Communicator size, THEN THE Halo_Plan SHALL throw std::invalid_argument indicating the invalid rank value and the valid range.
5. THE Halo_Plan SHALL be copyable and movable to allow storage in containers and reuse across multiple exchange invocations.
6. WHEN Halo_Plan::num_send_neighbors is called, THE Halo_Plan SHALL return the count of distinct send-neighbor ranks.
7. WHEN Halo_Plan::num_recv_neighbors is called, THE Halo_Plan SHALL return the count of distinct receive-neighbor ranks.
8. WHEN a Halo_Plan is constructed with an empty send-neighbor list or an empty receive-neighbor list, THE Halo_Plan SHALL succeed and store zero neighbors for the respective direction.
9. IF a neighbor rank appears more than once in the same send or receive list, THEN THE Halo_Plan SHALL throw std::invalid_argument indicating the duplicate rank.

### Requirement 6: Blocking Halo Exchange Execution

**User Story:** As a domain scientist, I want to execute a blocking halo exchange using a precomputed plan and Kokkos views, so that boundary data is transferred between subdomains synchronously with GPU-aware MPI when available.

#### Acceptance Criteria

1. WHEN halo_exchange_blocking is called with a Halo_Plan and a Kokkos_View, THE Halo_Exchange SHALL post MPI_Irecv for each receive-neighbor before posting any MPI_Isend, using the Communicator referenced by the Halo_Plan.
2. WHEN halo_exchange_blocking posts MPI_Isend for each send-neighbor, THE Halo_Exchange SHALL use a deterministic MPI tag derived from the sender and receiver ranks to ensure correct message matching.
3. WHEN all sends and receives are posted, THE Halo_Exchange SHALL call MPI_Waitall to block until all transfers complete and then return control to the caller.
4. IF the Kokkos_View resides in device memory and GPU-aware MPI is enabled via a compile-time preprocessor flag, THEN THE Halo_Exchange SHALL pass device pointers directly to MPI without staging through host memory.
5. IF the Kokkos_View resides in device memory and GPU-aware MPI is NOT enabled, THEN THE Halo_Exchange SHALL deep-copy send data to host memory before posting sends and deep-copy received data back to device memory after MPI_Waitall completes.
6. IF any MPI send or receive operation returns an error, THEN THE Halo_Exchange SHALL throw a std::runtime_error containing the MPI error string and the failing neighbor rank.
7. IF the Halo_Plan contains zero send-neighbors and zero receive-neighbors, THEN THE Halo_Exchange SHALL return immediately without posting any MPI operations.

### Requirement 7: Non-Blocking Asynchronous Halo Exchange

**User Story:** As a domain scientist, I want to initiate a halo exchange asynchronously and test or wait for completion later, so that computation can overlap with communication to hide latency.

#### Acceptance Criteria

1. WHEN halo_exchange_async is called with a Halo_Plan and a Kokkos_View, THE Halo_Exchange SHALL post all MPI_Isend and MPI_Irecv operations and return a Halo_Handle object.
2. THE Halo_Handle SHALL own all Request_Guard objects associated with the pending sends and receives.
3. WHEN Halo_Handle::test is called and all associated operations have completed, THE Halo_Handle SHALL return true and, if host-staged receive buffers exist, perform a deep-copy from host receive buffers back to the device Kokkos_View before returning.
4. WHEN Halo_Handle::wait is called, THE Halo_Handle SHALL block until all associated operations complete and, if host-staged receive buffers exist, perform a deep-copy from host receive buffers back to the device Kokkos_View before returning.
5. IF a Halo_Handle is destroyed before wait is called, THEN THE Halo_Handle destructor SHALL call wait to ensure all pending operations complete and post-receive deep-copies execute before resources are released.
6. WHEN the Kokkos_View resides in device memory and GPU-aware MPI is available, THE Halo_Exchange SHALL pass device pointers directly to MPI without staging through host memory.
7. WHEN the Kokkos_View resides in device memory and GPU-aware MPI is NOT available, THE Halo_Exchange SHALL deep-copy send data to host memory before posting sends and SHALL allocate host receive buffers for MPI_Irecv, retaining a reference to the device Kokkos_View for post-receive deep-copy upon wait or test-completion.
8. IF any MPI_Isend or MPI_Irecv operation fails during halo_exchange_async, THEN THE Halo_Exchange SHALL throw a std::runtime_error containing the MPI error string and the failing neighbor rank.
9. THE Halo_Handle SHALL delete copy construction and copy assignment operators and SHALL be move-constructible and move-assignable, with the source left in a valid empty state that does not invoke MPI operations on destruction.

### Requirement 8: Safe Raw Pointer Exposure for Kokkos Interop

**User Story:** As a library developer, I want RAII wrappers to safely expose raw MPI handles to Kokkos kernels, so that GPU-aware MPI calls can be made from device code without compromising resource safety.

#### Acceptance Criteria

1. WHEN Communicator::handle is called, THE Communicator SHALL return the raw MPI_Comm value by value without transferring ownership, and the accessor SHALL be marked noexcept and callable from host code to allow passing the handle to MPI C functions used in conjunction with Kokkos parallel regions.
2. WHEN Request_Guard::handle is called, THE Request_Guard SHALL return a non-const pointer to the internal MPI_Request without transferring ownership, and the accessor SHALL be marked noexcept.
3. IF Communicator::handle or Request_Guard::handle is called on a moved-from object, THEN THE accessor SHALL return MPI_COMM_NULL or a null pointer respectively.
4. THE HALO library SHALL document in the public header comments of each raw handle accessor that the accessor does not extend the lifetime of the owning RAII object and that callers must ensure the RAII object outlives any use of the returned handle.
5. THE HALO library SHALL use Kokkos::View::data() to obtain raw device pointers for MPI buffer arguments within halo exchange operations.

### Requirement 9: Thread Safety and MPI Threading Support

**User Story:** As a library developer, I want HALO to support MPI_THREAD_MULTIPLE environments, so that multi-threaded Kokkos execution spaces can safely invoke MPI operations concurrently.

#### Acceptance Criteria

1. WHEN HALO is initialized, THE HALO library SHALL query the MPI thread support level via MPI_Query_thread and store the result.
2. IF the detected MPI thread support level is MPI_THREAD_MULTIPLE, THEN THE HALO library SHALL permit concurrent halo exchange calls from multiple threads without internal serialization.
3. IF the detected MPI thread support level is below MPI_THREAD_MULTIPLE, THEN THE HALO library SHALL serialize all MPI communication calls issued by halo exchange operations through an internal mutex to prevent undefined behavior.
4. THE HALO library SHALL provide a thread-safe query function that returns the detected MPI thread support level as an integer value matching the MPI threading constants (MPI_THREAD_SINGLE, MPI_THREAD_FUNNELED, MPI_THREAD_SERIALIZED, or MPI_THREAD_MULTIPLE).
5. IF MPI_Init_thread has not been called before HALO initialization, THEN THE HALO library SHALL throw a std::runtime_error indicating that MPI must be initialized before HALO.
6. IF HALO initialization is called more than once, THEN THE HALO library SHALL return without re-querying MPI and preserve the previously stored thread support level.

### Requirement 10: CMake Build System

**User Story:** As a build engineer, I want HALO to provide a standalone CMake build system, so that the library can be built independently with proper dependency discovery for OpenMPI and Kokkos.

#### Acceptance Criteria

1. THE HALO build system SHALL use CMake version 3.21 or later and require the C++20 standard.
2. THE HALO build system SHALL locate MPI using find_package(MPI REQUIRED COMPONENTS CXX).
3. THE HALO build system SHALL locate Kokkos using find_package(Kokkos REQUIRED).
4. THE HALO build system SHALL produce a shared library target named halo with a namespace alias HELM::HALO, linking MPI and Kokkos as PUBLIC dependencies so that downstream consumers inherit their include paths and link flags transitively.
5. THE HALO build system SHALL export CMake configuration files (HALOConfig.cmake, HALOConfigVersion.cmake, and HALOTargets.cmake) so downstream projects can consume HALO via find_package(HALO), using SameMajorVersion compatibility for version matching.
6. THE HALO build system SHALL provide a BUILD_TESTING option defaulting to OFF that, when set to ON, builds the Google Test suite.
7. IF BUILD_TESTING is set to ON, THEN THE HALO build system SHALL locate Google Test using find_package(GTest REQUIRED).
8. IF MPI is not found during configuration, THEN THE HALO build system SHALL emit a fatal error indicating that MPI with CXX component is required.
9. IF Kokkos is not found during configuration, THEN THE HALO build system SHALL emit a fatal error indicating that Kokkos is required.

### Requirement 11: RAII Destructor Correctness Under Exceptions

**User Story:** As a library developer, I want a Google Test suite that proves RAII destructors execute correctly when exceptions unwind the stack, so that resource leaks are prevented in all error paths.

#### Acceptance Criteria

1. THE test suite SHALL contain a test that constructs a Communicator (via duplicate or split) inside a try block, throws a std::runtime_error, and after the catch block verifies the Communicator's internal handle equals MPI_COMM_NULL, confirming MPI_Comm_free was invoked during stack unwinding.
2. THE test suite SHALL contain a test that constructs a Request_Guard with a pending non-blocking operation inside a try block, throws a std::runtime_error, and after the catch block verifies the Request_Guard's internal handle equals MPI_REQUEST_NULL, confirming the request was completed or cancelled during stack unwinding.
3. THE test suite SHALL contain a test that constructs at least 2 nested scopes with a Communicator in the outer scope and a Request_Guard in the inner scope, throws from the innermost scope, and verifies via a destruction-order tracking mechanism that resources are freed in reverse construction order.
4. THE test suite SHALL contain a test that move-constructs a Communicator from a source, then verifies the source object's handle equals MPI_COMM_NULL and that destroying the source does not invoke MPI_Comm_free.
5. THE test suite SHALL contain a test that verifies Communicator wrapping MPI_COMM_WORLD does not call MPI_Comm_free on destruction.
6. THE test suite SHALL contain a test that creates a Communicator via split or duplicate, destroys it, and verifies the handle equals MPI_COMM_NULL afterward, confirming no leaked MPI handles remain (round-trip property).
7. THE test suite SHALL use a mock, spy, or instrumented MPI interposition layer to intercept and record calls to MPI_Comm_free, MPI_Cancel, and MPI_Request_free, enabling deterministic verification of destructor behavior without relying on MPI runtime side effects.

### Requirement 12: Repository and Container Structure

**User Story:** As a build engineer, I want HALO to live in its own dedicated repository added as a Git submodule within the HELM project, so that Tier 1 libraries maintain independent version histories and CI pipelines while remaining composable within the monorepo.

#### Acceptance Criteria

1. THE HALO source code SHALL reside in a dedicated Git repository that is added as a Git submodule at the path `libs/halo` within the HELM project workspace root.
2. THE HALO build and test workflow SHALL use the Docker image built from the HELM project DockerFile as its development and CI container environment.
3. THE HALO repository SHALL contain a standalone CMakeLists.txt at its root that, when configured and built inside the helm-project Docker container without any other HELM source trees present, produces the HELM::HALO library target without build errors.
4. THE HALO repository SHALL include a README at its root that documents prerequisites, the Docker container launch command, the CMake configure and build commands, and the command to run the test suite.
5. WHEN the HELM project workspace is cloned with `--recurse-submodules`, THE HALO submodule SHALL be checked out at a pinned commit so that the HELM project build is reproducible without additional manual steps.

### Requirement 13: Tier 1 Isolation Compliance

**User Story:** As an architect, I want HALO to have zero compile-time dependencies on other HELM Tier 1 libraries, so that the no-circular-dependency invariant of the HELM architecture is preserved.

#### Acceptance Criteria

1. THE HALO library SHALL NOT include any header files from TICK, LOGS, AXIS, AMIO, SPAN, or DAGR in any source file, public header, or internal header.
2. THE HALO library SHALL NOT link against any other HELM library target (HELM::TICK, HELM::LOGS, HELM::AXIS, HELM::AMIO, HELM::SPAN, or HELM::DAGR) at build time.
3. THE HALO library public headers SHALL only contain `#include` directives referencing C++ standard library headers, MPI headers, and Kokkos headers.
4. THE HALO CMakeLists.txt SHALL NOT reference any HELM:: namespace targets other than HELM::HALO in its target_link_libraries, add_dependencies, or find_package directives.
5. WHEN the HALO CI pipeline runs, THE build system SHALL execute a static verification step that scans all HALO source and header files for `#include` directives matching other HELM component header paths and fails the build if any are found.

### Requirement 14: Fortran C-Interop Interface

**User Story:** As a legacy Fortran model developer, I want a Fortran-callable interface to HALO using iso_c_binding, so that existing NUOPC/ESMF-based Fortran models can adopt HALO halo exchanges incrementally without rewriting their communication layers.

#### Acceptance Criteria

1. THE C_Interop_Layer SHALL expose a flat C API consisting of `extern "C"` functions for all public HALO operations (initialization, communicator creation, plan creation, blocking exchange, non-blocking exchange, handle wait, handle test, and resource destruction).
2. THE halo_mod Fortran module SHALL use iso_c_binding to wrap every C_Interop_Layer function, providing explicit Fortran interfaces with `bind(c)` attribute and appropriate `type(c_ptr)`, `integer(c_int)`, and `value` attributes on arguments.
3. WHEN the C_Interop_Layer creates a Communicator, Halo_Plan, or Halo_Handle, THE C_Interop_Layer SHALL return an Opaque_Handle as an `integer(c_int)` token to the Fortran caller rather than exposing raw C++ pointers across the language boundary.
4. WHEN the halo_mod receives an Opaque_Handle from the C_Interop_Layer, THE halo_mod SHALL store the handle as an `integer(c_int)` and pass it back to subsequent C_Interop_Layer calls for resource identification.
5. WHEN halo_mod::halo_exchange_blocking is called with a Halo_Plan handle and a Fortran_Contiguous_Array, THE halo_mod SHALL pass the array base address via `c_loc` to the C_Interop_Layer, which SHALL construct a non-owning view and execute the blocking exchange.
6. WHEN halo_mod::halo_exchange_async is called with a Halo_Plan handle and a Fortran_Contiguous_Array, THE halo_mod SHALL pass the array base address via `c_loc` to the C_Interop_Layer, which SHALL initiate the non-blocking exchange and return a Halo_Handle Opaque_Handle to the Fortran caller.
7. WHEN halo_mod::halo_wait is called with a Halo_Handle Opaque_Handle, THE C_Interop_Layer SHALL invoke Halo_Handle::wait on the corresponding C++ object and return an integer error code to the Fortran caller.
8. WHEN halo_mod::halo_test is called with a Halo_Handle Opaque_Handle, THE C_Interop_Layer SHALL invoke Halo_Handle::test on the corresponding C++ object, return an integer error code, and set an output integer flag indicating completion status (1 for complete, 0 for pending).
9. THE C_Interop_Layer SHALL return integer error codes from all functions where 0 indicates success and non-zero values indicate specific error conditions, rather than throwing C++ exceptions across the language boundary.
10. IF a C++ exception is thrown within any C_Interop_Layer function, THEN THE C_Interop_Layer SHALL catch the exception, map it to a non-zero integer error code, and return the error code to the Fortran caller without propagating the exception.
11. THE halo_mod SHALL accept MPI communicator handles as `integer(c_int)` values compatible with both the Fortran 2008 MPI binding (`type(MPI_Comm)%mpi_val`) and the legacy ESMF integer handle convention (`call ESMF_VMGet(vm, mpiCommunicator=int_comm)`).
12. THE halo_mod SHALL provide a `halo_plan_create` subroutine that accepts integer arrays of send-neighbor ranks, send-neighbor counts, receive-neighbor ranks, and receive-neighbor counts, and returns a Halo_Plan Opaque_Handle and an integer error code.
13. THE C_Interop_Layer SHALL provide a `halo_destroy_plan` and `halo_destroy_comm` function that releases the C++ resources associated with an Opaque_Handle and invalidates the handle token.
14. THE halo_mod and C_Interop_Layer SHALL compile and link correctly with Fortran 2008-compliant compilers including gfortran 9 or later, Intel ifort 2021 or later, and NVIDIA nvfortran 21.1 or later.
15. THE halo_mod derived types used for configuration structures SHALL use the `bind(c)` attribute with only `integer(c_int)` and `real(c_double)` members, and SHALL NOT contain allocatable, pointer, or polymorphic members.
16. WHEN halo_mod::halo_init is called with an integer MPI communicator handle, THE C_Interop_Layer SHALL call Environment::initialize and construct a root Communicator from the provided handle, returning an Opaque_Handle for the Communicator and an integer error code.

