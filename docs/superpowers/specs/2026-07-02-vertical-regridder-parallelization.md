# Spec: AXIS Vertical Regridding 3D Parallelization (Gap B)

## 1. Overview & Motivation
Currently, vertical regridding (`VerticalRegridder::interpolate`) is parallelized only across columns (`n_col`) using `Kokkos::RangePolicy`. Inside each column's thread, all vertical level computations (derivative solutions and spline evaluations) run sequentially. Furthermore, the temporary variables are allocated as raw arrays of size `256` on the thread's local stack, which creates a critical risk of stack overflow on GPUs.

This specification defines the high-performance refactoring of `VerticalRegridder` using **Kokkos `TeamPolicy` and Shared Scratch Memory**. This eliminates the GPU stack allocation risk and parallelizes vertical levels cooperatively within each execution team.

---

## 2. Design & Architecture

1.  **Kokkos TeamPolicy & Shared Memory Allocation:**
    *   Initialize `Kokkos::TeamPolicy` with league size equal to `n_col` and team size determined automatically (`Kokkos::AUTO`).
    *   Request dynamic shared scratch memory (L1-level speed) for each team:
        $$\text{Required Scratch Space} = 4 \times n\_src \times \text{sizeof(double)} \text{ bytes}$$
        This holds `src_x` (coordinates), `src_y` (values), `d` (computed derivatives), and `scratch` (tridiagonal solver workspace).
    *   Configure this using `.set_scratch_size(0, Kokkos::PerTeam(scratch_bytes))`.

2.  **Team Callback Flow (`KOKKOS_LAMBDA(const member_type& team)`):**
    *   **Step A: Fetch Scratch Memory:**
        Obtain pointers to the dynamic shared memory segments directly inside the device kernel.
    *   **Step B: Cooperative Ingestion (`TeamThreadRange`):**
        Use all threads in the team to read values and level coordinates from global memory and write them into the shared scratch space in parallel, followed by a `team_barrier()`.
    *   **Step C: Sequential Tridiagonal Solve:**
        Thread 0 of the team executes `solve_column_spline` sequentially over the shared arrays, followed by a `team_barrier()`.
    *   **Step D: Cooperative Spline Evaluation (`TeamThreadRange`):**
        Parallelize the spline evaluations for all `n_dst` destination levels across the team threads in parallel, writing results directly to the global destination field.

---

## 3. Verification & Testing Strategy

*   **Unit Tests:**
    *   Ensure all pre-existing vertical regridding unit tests in `test_axis_regridder` compile cleanly and pass with 100% agreement.
*   **Property-Based Tests:**
    *   Verify that any vertical interpolation property checks remain fully valid and pass with bitwise agreement to the previous sequential implementation.
