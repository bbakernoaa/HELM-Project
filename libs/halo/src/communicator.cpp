#include "halo/communicator.hpp"

#include <mpi.h>
#include <stdexcept>
#include <string>
#include <utility>

namespace halo {

// ─── Construction ────────────────────────────────────────────────────────────

Communicator::Communicator(MPI_Comm comm) noexcept
    : comm_{comm}
{}

// ─── Destruction ─────────────────────────────────────────────────────────────

Communicator::~Communicator() {
    if (comm_ == MPI_COMM_NULL) {
        return;
    }
    if (is_predefined()) {
        return;
    }

    // Do not call MPI_Comm_free if MPI has already been finalized.
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (finalized) {
        return;
    }

    MPI_Comm_free(&comm_);
}

// ─── Move Semantics ──────────────────────────────────────────────────────────

Communicator::Communicator(Communicator&& other) noexcept
    : comm_{std::exchange(other.comm_, MPI_COMM_NULL)}
{}

Communicator& Communicator::operator=(Communicator&& other) noexcept {
    if (this != &other) {
        // Free current handle if owned (same logic as destructor)
        if (comm_ != MPI_COMM_NULL && !is_predefined()) {
            int finalized = 0;
            MPI_Finalized(&finalized);
            if (!finalized) {
                MPI_Comm_free(&comm_);
            }
        }
        comm_ = std::exchange(other.comm_, MPI_COMM_NULL);
    }
    return *this;
}

// ─── Handle Accessor ─────────────────────────────────────────────────────────

MPI_Comm Communicator::handle() const noexcept {
    return comm_;
}

// ─── Rank and Size ───────────────────────────────────────────────────────────

int Communicator::rank() const {
    int r = 0;
    int rc = MPI_Comm_rank(comm_, &r);
    if (rc != MPI_SUCCESS) {
        char err_str[MPI_MAX_ERROR_STRING];
        int len = 0;
        MPI_Error_string(rc, err_str, &len);
        throw std::runtime_error(
            std::string("MPI_Comm_rank failed: ") + std::string(err_str, len));
    }
    return r;
}

int Communicator::size() const {
    int s = 0;
    int rc = MPI_Comm_size(comm_, &s);
    if (rc != MPI_SUCCESS) {
        char err_str[MPI_MAX_ERROR_STRING];
        int len = 0;
        MPI_Error_string(rc, err_str, &len);
        throw std::runtime_error(
            std::string("MPI_Comm_size failed: ") + std::string(err_str, len));
    }
    return s;
}

// ─── Split and Duplicate ─────────────────────────────────────────────────────

Communicator Communicator::split(int color, int key) const {
    MPI_Comm new_comm = MPI_COMM_NULL;
    int rc = MPI_Comm_split(comm_, color, key, &new_comm);
    if (rc != MPI_SUCCESS) {
        char err_str[MPI_MAX_ERROR_STRING];
        int len = 0;
        MPI_Error_string(rc, err_str, &len);
        throw std::runtime_error(
            std::string("MPI_Comm_split failed: ") + std::string(err_str, len));
    }
    return Communicator{new_comm};
}

Communicator Communicator::duplicate() const {
    MPI_Comm new_comm = MPI_COMM_NULL;
    int rc = MPI_Comm_dup(comm_, &new_comm);
    if (rc != MPI_SUCCESS) {
        char err_str[MPI_MAX_ERROR_STRING];
        int len = 0;
        MPI_Error_string(rc, err_str, &len);
        throw std::runtime_error(
            std::string("MPI_Comm_dup failed: ") + std::string(err_str, len));
    }
    return Communicator{new_comm};
}

// ─── Private Helpers ─────────────────────────────────────────────────────────

bool Communicator::is_predefined() const noexcept {
    return comm_ == MPI_COMM_WORLD || comm_ == MPI_COMM_SELF;
}

} // namespace halo
