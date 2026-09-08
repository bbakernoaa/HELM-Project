#ifndef HALO_REPLICATED_GATHER_PLAN_HPP
#define HALO_REPLICATED_GATHER_PLAN_HPP

/// @file replicated_gather_plan.hpp
/// @brief Tier 2 precomputed, reusable plan for a multi-level replicated gather.
///
/// Replicated_Gather_Plan<T> is built once from this rank's per-level band
/// element count and the number of outer-axis levels. At construction it
/// allgathers every rank's per-level band count, prefix-sums those counts into
/// per-rank displacements, records the total per-level element count, and builds
/// a committed, resized strided receive MPI_Datatype describing how a single
/// MPI_Allgatherv scatters each rank's [level][band] contribution directly into
/// the replicated [level][j][i] layout with ZERO host reorder.
///
/// The plan mirrors Structured_Halo_Plan: it holds a non-owning
/// const Communicator* (which must outlive the plan), deletes copy, supports a
/// move that transfers ownership of the derived datatype and leaves the source
/// safe to destroy, and frees the datatype exactly once (RAII, MPI_Finalized
/// guarded) via its detail::Strided_Datatype member.

#include <mpi.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <halo/collectives.hpp>
#include <halo/communicator.hpp>
#include <halo/detail/mpi_datatype.hpp>
#include <halo/detail/strided_datatype.hpp>
#include <halo/environment.hpp>
#include <halo/error_policy.hpp>

namespace halo {

/// @brief Precomputed, reusable plan for a multi-level replicated gather.
///
/// Built once from this rank's per-level band element count and the number of
/// levels. Assembling the replicated field is then a single MPI_Allgatherv
/// (issued by gather_replicated) using the derived receive datatype so all
/// levels land in [level][j][i] with no host reorder.
///
/// Templated on the element type T because the derived receive datatype is
/// built from detail::mpi_datatype_for<T>() and the per-level stride is in
/// units of T.
///
/// @tparam T Element type of the gathered field.
template <typename T>
class Replicated_Gather_Plan {
   public:
    /// @brief Construct a replicated-gather plan.
    ///
    /// Validation runs BEFORE any MPI call or datatype creation (Req 4.5, 4.6).
    /// The constructor then:
    ///   1. Allgathers every rank's per-level band count into band_counts_ via
    ///      the Tier 1 allgather (Req 4.1).
    ///   2. Prefix-sums those counts into per-rank displacements_ (Req 4.2).
    ///   3. Records per_level_total_ = sum of all band counts (Req 4.4).
    ///   4. Builds and commits the resized strided receive datatype describing
    ///      the multi-level scatter and stores it in recv_type_ (Req 4.3).
    ///
    /// @param comm             Communicator (non-owning; MUST outlive the plan).
    /// @param local_band_count This rank's band element count for ONE level.
    /// @param num_levels       Number of outer-axis levels (> 0).
    ///
    /// @throws std::invalid_argument if @p num_levels == 0 (Req 4.5) or
    ///         @p local_band_count < 0 (Req 4.6), before any MPI call.
    /// @throws std::runtime_error via detail::handle_mpi_error if the internal
    ///         allgather fails.
    Replicated_Gather_Plan(const Communicator &comm, int local_band_count, int num_levels) : comm_(&comm) {
        // ── Validation BEFORE any MPI / datatype creation (Req 4.5, 4.6) ──
        if (num_levels == 0) {
            throw std::invalid_argument("Replicated_Gather_Plan: num_levels must be > 0 (got num_levels = " +
                                        std::to_string(num_levels) + ")");
        }
        if (local_band_count < 0) {
            throw std::invalid_argument("Replicated_Gather_Plan: local_band_count must be >= 0 (got " +
                                        std::to_string(local_band_count) + ")");
        }

        num_levels_ = num_levels;
        local_band_count_ = local_band_count;

        // 1. Allgather every rank's per-level band count (Req 4.1).
        band_counts_ = allgather<int>(*comm_, &local_band_count, 1);

        // 2. Prefix-sum into per-rank displacements (Req 4.2).
        displacements_ = detail::prefix_sum(band_counts_);

        // 3. Total per-level element count == sum of band counts
        //    == displacements_.back() + band_counts_.back() (Req 4.4).
        per_level_total_ = displacements_.back() + band_counts_.back();

        // 4. Build + commit the resized strided send/receive datatypes (Req 4.3).
        build_datatypes();
    }

    /// @brief Destructor. Frees the owned datatype exactly once via the
    ///        detail::Strided_Datatype member (RAII, MPI_Finalized guarded).
    ~Replicated_Gather_Plan() = default;

    /// @brief Move-construct, transferring datatype ownership (Req 4.7, 8.5).
    ///
    /// The detail::Strided_Datatype member transfers ownership of the committed
    /// datatype; the source's comm_ is nulled so the moved-from plan is safe to
    /// destroy.
    Replicated_Gather_Plan(Replicated_Gather_Plan &&other) noexcept
        : comm_(other.comm_),
          num_levels_(other.num_levels_),
          per_level_total_(other.per_level_total_),
          local_band_count_(other.local_band_count_),
          band_counts_(std::move(other.band_counts_)),
          displacements_(std::move(other.displacements_)),
          send_type_(std::move(other.send_type_)),
          recv_type_(std::move(other.recv_type_)) {
        other.comm_ = nullptr;
        other.num_levels_ = 0;
        other.per_level_total_ = 0;
        other.local_band_count_ = 0;
    }

    /// @brief Move-assign, freeing any current datatype then taking @p other's.
    Replicated_Gather_Plan &operator=(Replicated_Gather_Plan &&other) noexcept {
        if (this != &other) {
            comm_ = other.comm_;
            num_levels_ = other.num_levels_;
            per_level_total_ = other.per_level_total_;
            local_band_count_ = other.local_band_count_;
            band_counts_ = std::move(other.band_counts_);
            displacements_ = std::move(other.displacements_);
            send_type_ = std::move(other.send_type_);  // frees this->send_type_ first
            recv_type_ = std::move(other.recv_type_);  // frees this->recv_type_ first
            other.comm_ = nullptr;
            other.num_levels_ = 0;
            other.per_level_total_ = 0;
            other.local_band_count_ = 0;
        }
        return *this;
    }

    // Copy is deleted — the derived datatype is not duplicated (Req 8.5).
    Replicated_Gather_Plan(const Replicated_Gather_Plan &) = delete;
    Replicated_Gather_Plan &operator=(const Replicated_Gather_Plan &) = delete;

    // ─── Immutable accessors (all const, noexcept) ──────────────────────────

    /// @brief Number of outer-axis levels.
    [[nodiscard]] int num_levels() const noexcept { return num_levels_; }

    /// @brief Total per-level element count (sum of all ranks' band counts).
    [[nodiscard]] int per_level_total() const noexcept { return per_level_total_; }

    /// @brief Per-rank band element counts for one level (ascending rank order).
    [[nodiscard]] const std::vector<int> &band_counts() const noexcept { return band_counts_; }

    /// @brief Per-rank displacements for one level (prefix sum of band_counts()).
    [[nodiscard]] const std::vector<int> &displacements() const noexcept { return displacements_; }

    /// @brief This rank's per-level band element count (as passed to the ctor).
    [[nodiscard]] int local_band_count() const noexcept { return local_band_count_; }

    /// @brief The committed, resized strided send datatype.
    ///
    /// Presents this rank's contiguous [level][band] source in [band][level]
    /// order so ONE instance is a single j-column across all levels; sending
    /// local_band_count() instances feeds the receive datatype in the order it
    /// scatters (see build_datatypes / the design Data Models "crux").
    [[nodiscard]] MPI_Datatype send_type() const noexcept { return send_type_.get(); }

    /// @brief The committed, resized strided receive datatype.
    [[nodiscard]] MPI_Datatype recv_type() const noexcept { return recv_type_.get(); }

    /// @brief Access the communicator.
    [[nodiscard]] const Communicator &communicator() const noexcept { return *comm_; }

   private:
    /// @brief Build and commit the resized strided send + receive datatypes
    ///        (Req 4.3).
    ///
    /// Receive datatype (per the design Data Models "crux"): a vector type of
    /// num_levels_ blocks, blocklength 1, stride per_level_total_ elements over
    /// mpi_datatype_for<T>(), resized to a 1-element (sizeof(T)) extent so the
    /// Allgatherv rdispls[r] = displacements_[r] address the j-axis directly and
    /// the stride carries each rank's band across levels. recvcounts[r] =
    /// band_counts_[r] selects that rank's band width.
    ///
    /// Send datatype (the factoring pinned against the container's OpenMPI):
    /// because the shared receive datatype forces the per-rank band width to be
    /// carried by recvcounts (band = the outer, instance axis MPI consumes
    /// first), the receive side consumes each sender's stream in [band][level]
    /// order. This rank's source is contiguous in [level][band] order, so a
    /// plain contiguous send would scatter diagonally. The send datatype
    /// therefore reads the [level][band] source as [band][level]: a vector of
    /// num_levels_ blocks, blocklength 1, stride local_band_count_ (one
    /// j-column across all levels), resized to a 1-element extent. gather_
    /// replicated sends local_band_count_ instances of it, presenting one
    /// j-column per instance in the exact order the receive datatype scatters.
    /// One committed receive datatype + the (band_counts_, displacements_) pair
    /// then reproduce the naive per-level layout exactly, with no host reorder.
    void build_datatypes() {
        detail::Serialized_MPI_Guard guard;
        const MPI_Datatype base = detail::mpi_datatype_for<T>();

        // ── Receive datatype: num_levels_ blocks, stride per_level_total_ ──
        recv_type_ = make_resized_column(base, /*stride=*/per_level_total_);

        // ── Send datatype: num_levels_ blocks, stride local_band_count_. When
        //    this rank contributes no band (local_band_count_ == 0) it sends
        //    zero instances, so the send type is never dereferenced; build it
        //    with a benign unit stride to avoid a zero stride. ─────────────
        const int send_stride = local_band_count_ > 0 ? local_band_count_ : 1;
        send_type_ = make_resized_column(base, /*stride=*/send_stride);
    }

    /// @brief Build one committed, resized "column" datatype: num_levels_ blocks
    ///        of one base element, spaced @p stride base elements apart, resized
    ///        to a single-element (sizeof(T)) extent. The intermediate vector
    ///        type is freed before returning.
    [[nodiscard]] detail::Strided_Datatype make_resized_column(MPI_Datatype base, int stride) {
        MPI_Datatype level_strided = MPI_DATATYPE_NULL;
        int rc = MPI_Type_vector(num_levels_, /*blocklength=*/1, stride, base, &level_strided);
        if (rc != MPI_SUCCESS) {
            detail::handle_mpi_error(rc, comm_->rank(), "MPI_Type_vector", comm_->handle());
        }

        MPI_Datatype resized = MPI_DATATYPE_NULL;
        rc = MPI_Type_create_resized(level_strided, /*lb=*/0,
                                     /*extent=*/static_cast<MPI_Aint>(sizeof(T)), &resized);
        if (rc != MPI_SUCCESS) {
            MPI_Type_free(&level_strided);
            detail::handle_mpi_error(rc, comm_->rank(), "MPI_Type_create_resized", comm_->handle());
        }

        rc = MPI_Type_commit(&resized);
        if (rc != MPI_SUCCESS) {
            MPI_Type_free(&level_strided);
            detail::handle_mpi_error(rc, comm_->rank(), "MPI_Type_commit", comm_->handle());
        }

        MPI_Type_free(&level_strided);
        return detail::Strided_Datatype{resized};
    }

    const Communicator *comm_;                   ///< Non-owning (Req 8.6).
    int num_levels_{0};                          ///< Immutable after ctor (Req 4.4).
    int per_level_total_{0};                      ///< Immutable after ctor (Req 4.4).
    int local_band_count_{0};                     ///< This rank's band width (send type).
    std::vector<int> band_counts_;                ///< Per-rank, one level (Req 4.1).
    std::vector<int> displacements_;              ///< Per-rank, one level (Req 4.2).
    detail::Strided_Datatype send_type_;          ///< RAII, move-only (Req 4.7).
    detail::Strided_Datatype recv_type_;          ///< RAII, move-only (Req 4.7).
};

}  // namespace halo

#endif  // HALO_REPLICATED_GATHER_PLAN_HPP
