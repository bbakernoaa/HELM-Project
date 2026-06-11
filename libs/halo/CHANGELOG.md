# Changelog

All notable changes to the HALO library will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-01

### Added

- `halo::Communicator` — RAII wrapper for `MPI_Comm` with rank/size queries, split, and duplicate.
- `halo::Environment` — Singleton managing MPI initialization and finalization (RAII).
- `halo::Halo_Plan` — Precomputed flat-buffer exchange descriptor (neighbor list, send/recv counts).
- `halo::Halo_Handle` — Asynchronous exchange handle with wait/test semantics.
- `halo::Request_Guard` — RAII guard for `MPI_Request` (auto-wait on destruction).
- `halo::Window_Guard` — RAII guard for `MPI_Win` (auto-free on destruction).
- `halo::exchange_blocking()` — Blocking halo exchange on flat Kokkos buffers.
- `halo::exchange_async()` — Non-blocking halo exchange returning a `Halo_Handle`.
- Compile-time GPU-aware MPI support via `HALO_GPU_AWARE_MPI` flag.
- Host-to-device staging for non-GPU-aware MPI implementations.
- Fortran C-interop layer (`halo_mod`) exposing the full exchange API.
- CMake install/export with `HELM::HALO` and `HELM::HALO_Fortran` targets.
- GTest + RapidCheck test suite (22 tests covering all public API surfaces).
- Tier 1 isolation check (zero dependencies on other HELM components).
