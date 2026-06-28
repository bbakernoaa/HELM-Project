// SPDX-License-Identifier: Apache-2.0
// SPAN — TripleBuffer<T>: lock-free I/O isolation for AMIO background streaming
// Copyright (c) HELM Project Contributors

#ifndef SPAN_TRIPLE_BUFFER_HPP
#define SPAN_TRIPLE_BUFFER_HPP

/// @file span/triple_buffer.hpp
/// @brief Triple-buffer pointer rotation for race-free asynchronous I/O.
///
/// TripleBuffer manages three logical buffer tracks (Write, Read, IO) to
/// enable AMIO background threads to stream Zarr/HDF5 chunks from a stable
/// pointer while the model continues writing to a separate buffer without
/// data races.
///
/// @par Thread Safety
/// - swap_buffers_for_amio() is mutex-protected (safe from any thread).
/// - The Write buffer is accessed ONLY by the model thread (no contention).
/// - The IO buffer is accessed ONLY by the AMIO thread after swap.
///
/// @par Memory Model
/// Each track holds a raw pointer to externally-owned memory (Fortran arrays).
/// TripleBuffer does NOT allocate or free the underlying storage — it only
/// rotates pointer assignments. HELM Law #1 is fully preserved.

#include <cstddef>
#include <mutex>
#include <stdexcept>

namespace span {

/// @brief Triple-buffer pointer rotation for race-free asynchronous I/O.
///
/// Manages three logical buffer tracks:
///   - **Write** — The model physics actively writes to this buffer every timestep.
///   - **Read**  — Contains the most recently completed field snapshot.
///   - **IO**    — Locked by the AMIO background thread for streaming to storage.
///
/// The `swap_buffers_for_amio()` operation is an O(1) pointer rotation that
/// moves pointers in the cycle Write→Read→IO→Write, allowing background
/// threads to serialize data without data races against model writes.
///
/// @tparam T  Element type of the buffered arrays.
template <typename T>
class TripleBuffer {
   public:
    /// @brief Construct a triple buffer from three pre-allocated pointer tracks.
    ///
    /// @param buf_write   Initial Write buffer pointer.
    /// @param buf_read    Initial Read buffer pointer.
    /// @param buf_io      Initial IO buffer pointer.
    /// @param n_elements  Number of elements in each buffer (all must be equal).
    ///
    /// @pre buf_write, buf_read, buf_io are non-null and point to at least n_elements of T.
    /// @pre All three buffers are distinct (no aliasing).
    /// @pre n_elements > 0.
    ///
    /// @throws std::invalid_argument if any pointer is null, n_elements is 0,
    ///         or pointers are not distinct.
    TripleBuffer(T *buf_write, T *buf_read, T *buf_io, std::size_t n_elements) : n_elements_(n_elements), io_locked_(false) {
        // Validate non-null
        if (buf_write == nullptr) {
            throw std::invalid_argument("TripleBuffer: buf_write must not be null");
        }
        if (buf_read == nullptr) {
            throw std::invalid_argument("TripleBuffer: buf_read must not be null");
        }
        if (buf_io == nullptr) {
            throw std::invalid_argument("TripleBuffer: buf_io must not be null");
        }

        // Validate n_elements > 0
        if (n_elements == 0) {
            throw std::invalid_argument("TripleBuffer: n_elements must be > 0");
        }

        // Validate distinct pointers
        if (buf_write == buf_read || buf_write == buf_io || buf_read == buf_io) {
            throw std::invalid_argument("TripleBuffer: all three buffer pointers must be distinct");
        }

        buffers_[WRITE] = buf_write;
        buffers_[READ] = buf_read;
        buffers_[IO] = buf_io;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Model-thread interface
    // ─────────────────────────────────────────────────────────────────────────

    /// @brief Get the current write buffer pointer.
    ///
    /// The model physics kernel writes new field values here every timestep.
    /// This pointer is stable until swap_buffers_for_amio() rotates it.
    ///
    /// @return Pointer to the active Write buffer.
    [[nodiscard]] T *write_ptr() const noexcept {
        return buffers_[WRITE];
    }

    /// @brief Get the current read buffer pointer.
    ///
    /// Contains the most recently completed snapshot (the last Write before
    /// a swap). Available for read-only consumption.
    ///
    /// @return Pointer to the Read buffer.
    [[nodiscard]] T *read_ptr() const noexcept {
        return buffers_[READ];
    }

    // ─────────────────────────────────────────────────────────────────────────
    // AMIO-thread interface
    // ─────────────────────────────────────────────────────────────────────────

    /// @brief Get the current IO buffer pointer.
    ///
    /// After a successful swap_buffers_for_amio(), this points to the buffer
    /// locked for AMIO background consumption.
    ///
    /// @return Pointer to the IO buffer.
    [[nodiscard]] T *io_ptr() const noexcept {
        return buffers_[IO];
    }

    /// @brief Rotate pointer tracks for AMIO consumption.
    ///
    /// Performs the rotation: Write→Read→IO→Write (circular shift).
    /// After success:
    ///   - Old Write becomes new Read (latest snapshot)
    ///   - Old Read becomes new IO (locked for AMIO streaming)
    ///   - Old IO becomes new Write (recycled for model)
    ///   - io_locked_ is set to true
    ///
    /// If IO is already locked, returns false without modifying anything.
    ///
    /// @par Complexity: O(1) — three pointer assignments under mutex.
    /// @par Thread Safety: Safe to call from any thread.
    ///
    /// @return true on successful swap, false if IO is already locked.
    [[nodiscard]] bool swap_buffers_for_amio() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);

        if (io_locked_) {
            return false;  // Req 6.2: return false without modifying state
        }

        // Rotation: Write→Read→IO→Write
        // Save old pointers
        T *old_write = buffers_[WRITE];
        T *old_read = buffers_[READ];
        T *old_io = buffers_[IO];

        // Apply rotation
        buffers_[READ] = old_write;  // old Write → new Read
        buffers_[IO] = old_read;     // old Read  → new IO
        buffers_[WRITE] = old_io;    // old IO    → new Write

        io_locked_ = true;

        return true;  // Req 6.3: return true on success
    }

    /// @brief Release the IO buffer back to the rotation pool.
    ///
    /// Must be called by the AMIO thread after it finishes streaming the
    /// IO buffer to storage. After this call, subsequent swap_buffers_for_amio()
    /// calls will succeed.
    ///
    /// @par Thread Safety: Safe to call from any thread.
    void release_io() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        io_locked_ = false;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // State queries
    // ─────────────────────────────────────────────────────────────────────────

    /// @brief Query whether the IO track is currently locked by AMIO.
    ///
    /// Returns true if swap_buffers_for_amio() was called and release_io()
    /// has not yet been called. Otherwise returns false.
    [[nodiscard]] bool is_io_locked() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return io_locked_;
    }

    /// @brief Number of elements in each buffer track.
    [[nodiscard]] std::size_t size() const noexcept {
        return n_elements_;
    }

   private:
    // Track indices
    static constexpr int WRITE = 0;
    static constexpr int READ = 1;
    static constexpr int IO = 2;

    T *buffers_[3] = {};
    std::size_t n_elements_ = 0;
    mutable std::mutex mutex_;
    bool io_locked_ = false;
};

}  // namespace span

#endif  // SPAN_TRIPLE_BUFFER_HPP
