# HALO Thread Safety Model

This document describes how HALO behaves under different MPI thread support
levels and provides guidance for calling HALO from multi-threaded contexts
(OpenMP, C++ `std::thread`, etc.).

---

## Background: MPI Thread Levels

MPI defines four thread support levels, negotiated at `MPI_Init_thread` time:

| Level | Meaning |
| --- | --- |
| `MPI_THREAD_SINGLE` | Only one thread may call MPI |
| `MPI_THREAD_FUNNELED` | Only the main thread may call MPI |
| `MPI_THREAD_SERIALIZED` | Any thread may call MPI, but not concurrently |
| `MPI_THREAD_MULTIPLE` | Any thread may call MPI concurrently |

HALO queries the granted level at `Environment::initialize()` via
`MPI_Query_thread` and caches it for the lifetime of the process.

```cpp
halo::Environment::initialize();                       // call once after MPI_Init_thread
int level = halo::Environment::thread_support_level(); // MPI_THREAD_* constant
bool safe  = halo::Environment::is_thread_multiple();  // true if MULTIPLE granted
```

---

## The Serialized_MPI_Guard

All HALO exchange functions (blocking, async, structured, persistent) protect
their internal MPI calls with `detail::Serialized_MPI_Guard`. This RAII guard
implements a simple adaptive policy:

| Detected level | Guard behavior |
| --- | --- |
| `MPI_THREAD_MULTIPLE` | **No-op.** No lock acquired; concurrent exchanges are safe. |
| Anything below MULTIPLE | Locks a process-global `std::mutex` for the duration of the MPI call sequence. |

The guard is constructed at the start of each exchange path and destroyed at
the end, so the critical section covers the full send/recv cycle.

```cpp
// Simplified internal pattern (do NOT call directly — this is library internals)
{
    halo::detail::Serialized_MPI_Guard guard;  // acquires mutex if needed
    MPI_Isend(...);
    MPI_Irecv(...);
    MPI_Waitall(...);
}  // guard destructor releases mutex
```

---

## MPI_THREAD_MULTIPLE — Full Concurrency

When the MPI runtime grants `MPI_THREAD_MULTIPLE`:

- The guard is a **complete no-op** (no mutex, no atomic, no overhead).
- Multiple threads may call any HALO exchange function concurrently on
  different plans/views without synchronization.
- This is the recommended mode for GPU-accelerated systems where MPI
  implementations (e.g., CUDA-aware Open MPI, Cray MPICH with GPU support)
  typically provide full thread safety.

### Usage pattern (OpenMP)

```cpp
halo::Environment::initialize();
assert(halo::Environment::is_thread_multiple());

#pragma omp parallel for
for (int field = 0; field < num_fields; ++field) {
    // Each thread exchanges a different field — safe without locks.
    halo::exchange_structured_blocking(plan, fields[field]);
}
```

### What is safe to call concurrently

| API | Concurrent-safe under MULTIPLE? |
| --- | --- |
| `exchange_blocking` | Yes |
| `exchange_async` + `Halo_Handle::wait()` | Yes (distinct handles) |
| `exchange_structured_blocking` | Yes |
| `exchange_structured_async` | Yes |
| `Persistent_Halo_Handle::start()` / `wait()` | Yes (distinct handles) |
| `exchange_neighbor_collective` | Yes |
| `Environment::is_thread_multiple()` | Yes (read-only) |
| `Environment::is_gpu_aware_mpi()` | Yes (read-only) |
| `Diagnostics::set_callback()` | Not safe concurrently with `emit()` — set before exchanges begin |

---

## MPI_THREAD_SERIALIZED — Manual Serialization Required

When only `MPI_THREAD_SERIALIZED` is available (common on Cray systems with
`cray-mpich` default configuration):

- The guard acquires a process-global mutex on entry, ensuring only one thread
  at a time executes MPI calls through HALO.
- This means HALO calls **will not deadlock**, but they **will serialize**
  against each other, eliminating any benefit of concurrent exchange calls.

### Constraints

1. **All HALO exchange calls must be issued from a serial context** — outside
   `#pragma omp parallel`, or inside `#pragma omp single` / `#pragma omp master`.
2. Calling HALO exchange from multiple OpenMP threads concurrently is
   **technically safe** (the mutex prevents data races), but **pointless** — the
   mutex forces sequential execution anyway, and the thread contention adds
   overhead compared to simply calling from one thread.
3. `Environment::initialize()` must be called from the main thread before any
   parallel region.

### Recommended pattern (serial context)

```cpp
halo::Environment::initialize();
// MPI_THREAD_SERIALIZED detected — issue exchanges from serial context.

// Exchange all fields sequentially (serial context)
for (int field = 0; field < num_fields; ++field) {
    halo::exchange_structured_blocking(plan, fields[field]);
}

// THEN enter parallel region for computation
#pragma omp parallel for
for (int field = 0; field < num_fields; ++field) {
    compute_interior(fields[field]);
}
```

### Anti-pattern (avoid this)

```cpp
// BAD: exchanges from parallel region under SERIALIZED — safe but wasteful.
#pragma omp parallel for
for (int field = 0; field < num_fields; ++field) {
    // All threads contend on the same mutex; effectively serial with overhead.
    halo::exchange_structured_blocking(plan, fields[field]);
}
```

---

## MPI_THREAD_FUNNELED and MPI_THREAD_SINGLE

Under `MPI_THREAD_FUNNELED` or `MPI_THREAD_SINGLE`, the same mutex-based guard
applies. However, the MPI standard imposes additional constraints:

- **FUNNELED**: Only the main thread (the one that called `MPI_Init_thread`)
  may make MPI calls. HALO's guard serializes but does **not** enforce thread
  identity — it is the caller's responsibility to ensure HALO is only called
  from the main thread.
- **SINGLE**: Only one thread exists in the program. HALO works correctly but
  the mutex is never contended.

> **Recommendation:** Request at least `MPI_THREAD_SERIALIZED` when using HALO.
> If your MPI implementation supports it, prefer `MPI_THREAD_MULTIPLE` for best
> performance.

---

## Optional: Dedicated Communication Thread Mode

For systems limited to `MPI_THREAD_SERIALIZED` that still need overlap between
communication and computation, HALO supports a **dedicated communication
thread** pattern. In this mode, a single background thread owns all MPI calls,
freeing worker threads to compute without contention on the serialization mutex.

### Concept

```
┌──────────────────────────────────────┐
│  Worker threads (OpenMP)             │
│  ┌────────┐ ┌────────┐ ┌────────┐   │
│  │compute │ │compute │ │compute │   │
│  └───┬────┘ └───┬────┘ └───┬────┘   │
│      │          │          │         │
│      └──────────┴──────────┘         │
│              enqueue                  │
│                │                     │
├────────────────┼─────────────────────┤
│  Communication thread (background)   │
│                ▼                     │
│  ┌─────────────────────────────┐     │
│  │ dequeue → pack → MPI → unpack │   │
│  └─────────────────────────────┘     │
└──────────────────────────────────────┘
```

### Design considerations

- The communication thread is the **only** thread that calls MPI, satisfying
  `MPI_THREAD_SERIALIZED` semantics (single thread at a time, guaranteed).
- Worker threads post exchange requests to a thread-safe queue and receive
  a future/handle signaling completion.
- This eliminates mutex contention on the critical path — worker threads never
  block on the serialization mutex.
- The communication thread can batch multiple exchange requests for efficiency.

### Implementation status

The dedicated communication thread mode is a **planned extension**. The current
HALO release provides the `Serialized_MPI_Guard` mechanism which is correct and
safe under all thread levels. The communication thread mode would be a
performance optimization for `MPI_THREAD_SERIALIZED` environments that need
communication/computation overlap.

To implement this pattern at the application level today:

```cpp
#include <halo/halo.hpp>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

// Application-level communication thread (sketch)
class Comm_Thread {
public:
    void enqueue(std::function<void()> exchange_fn) {
        {
            std::lock_guard lk(mtx_);
            queue_.push(std::move(exchange_fn));
        }
        cv_.notify_one();
    }

    void run() {
        while (!done_) {
            std::unique_lock lk(mtx_);
            cv_.wait(lk, [&] { return !queue_.empty() || done_; });
            while (!queue_.empty()) {
                auto fn = std::move(queue_.front());
                queue_.pop();
                lk.unlock();
                fn();  // HALO exchange runs here — single thread, safe
                lk.lock();
            }
        }
    }

    void stop() { done_ = true; cv_.notify_one(); }

private:
    std::queue<std::function<void()>> queue_;
    std::mutex mtx_;
    std::condition_variable cv_;
    bool done_ = false;
};
```

---

## Summary Table

| MPI Thread Level | HALO behavior | Recommended usage |
| --- | --- | --- |
| `MPI_THREAD_MULTIPLE` | No serialization; concurrent exchanges safe | Call from any thread freely |
| `MPI_THREAD_SERIALIZED` | Mutex serializes MPI calls | Call from serial context; consider comm thread for overlap |
| `MPI_THREAD_FUNNELED` | Mutex serializes; caller must be main thread | Call only from main thread |
| `MPI_THREAD_SINGLE` | Mutex present but uncontended | Single-threaded use only |

---

## Querying Thread Support at Runtime

```cpp
#include <halo/environment.hpp>

halo::Environment::initialize();

if (halo::Environment::is_thread_multiple()) {
    // Safe to call HALO from any thread concurrently
} else {
    int level = halo::Environment::thread_support_level();
    // Adapt calling pattern to the detected level
}
```

---

## Further Reading

- [MPI Standard §12.4 — MPI and Threads](https://www.mpi-forum.org/docs/)
- [HALO overlap pattern guide](overlap_pattern.md) — async exchange with computation overlap
- [HALO CI pipeline](CI_PIPELINE.md) — thread-safety tests under sanitizers
