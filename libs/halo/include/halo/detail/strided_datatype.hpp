#ifndef HALO_DETAIL_STRIDED_DATATYPE_HPP
#define HALO_DETAIL_STRIDED_DATATYPE_HPP

/// @file halo/detail/strided_datatype.hpp
/// @brief Move-only RAII wrapper owning a derived MPI_Datatype.
///
/// Owns a committed MPI_Datatype (e.g. a resized strided receive type built for
/// the replicated-gather plan) and frees it exactly once on destruction. The
/// free is guarded by MPI_Finalized so that a wrapper outliving MPI finalization
/// does not call MPI_Type_free after the library has shut down. Copying is
/// deleted; moving transfers ownership and leaves the moved-from wrapper holding
/// MPI_DATATYPE_NULL (which frees nothing).

#include <mpi.h>

namespace halo::detail {

/// @brief Move-only RAII owner of a single MPI_Datatype.
///
/// A default-constructed wrapper holds MPI_DATATYPE_NULL and owns nothing. The
/// explicit constructor takes ownership of a caller-committed datatype. The
/// destructor frees the owned datatype exactly once (skipped when MPI has been
/// finalized). Move operations transfer ownership; the source is reset to
/// MPI_DATATYPE_NULL so its later destruction is a no-op.
class Strided_Datatype {
   public:
    /// @brief Construct an empty wrapper holding MPI_DATATYPE_NULL.
    Strided_Datatype() noexcept = default;

    /// @brief Take ownership of an already-committed MPI_Datatype.
    /// @param dt The datatype to own. May be MPI_DATATYPE_NULL.
    explicit Strided_Datatype(MPI_Datatype dt) noexcept : dt_(dt) {}

    /// @brief Free the owned datatype (guarded by MPI_Finalized).
    ~Strided_Datatype() {
        free();
    }

    /// @brief Move-construct, transferring ownership from @p o.
    Strided_Datatype(Strided_Datatype &&o) noexcept : dt_(o.dt_) {
        o.dt_ = MPI_DATATYPE_NULL;
    }

    /// @brief Move-assign, freeing any current datatype then taking @p o's.
    Strided_Datatype &operator=(Strided_Datatype &&o) noexcept {
        if (this != &o) {
            free();
            dt_ = o.dt_;
            o.dt_ = MPI_DATATYPE_NULL;
        }
        return *this;
    }

    Strided_Datatype(const Strided_Datatype &) = delete;
    Strided_Datatype &operator=(const Strided_Datatype &) = delete;

    /// @brief Access the owned datatype without transferring ownership.
    /// @return The owned MPI_Datatype (MPI_DATATYPE_NULL if empty).
    [[nodiscard]] MPI_Datatype get() const noexcept {
        return dt_;
    }

   private:
    /// @brief Free the owned datatype exactly once, unless MPI is finalized.
    void free() noexcept {
        if (dt_ != MPI_DATATYPE_NULL) {
            int finalized = 0;
            MPI_Finalized(&finalized);
            if (!finalized) {
                MPI_Type_free(&dt_);
            }
            dt_ = MPI_DATATYPE_NULL;
        }
    }

    MPI_Datatype dt_{MPI_DATATYPE_NULL};
};

}  // namespace halo::detail

#endif  // HALO_DETAIL_STRIDED_DATATYPE_HPP
