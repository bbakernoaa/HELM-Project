#include "halo/request_guard.hpp"

#include <mpi.h>

#include <exception>
#include <utility>

namespace halo {

// ─── Construction ────────────────────────────────────────────────────────────

Request_Guard::Request_Guard(MPI_Request &req) noexcept : req_{req}, uncaught_on_entry_{std::uncaught_exceptions()} {
    req = MPI_REQUEST_NULL;
}

// ─── Destruction ─────────────────────────────────────────────────────────────

Request_Guard::~Request_Guard() {
    if (req_ == MPI_REQUEST_NULL) {
        return;
    }

    if (std::uncaught_exceptions() > uncaught_on_entry_) {
        // Stack unwinding: cancel the operation and free the request.
        MPI_Cancel(&req_);
        MPI_Request_free(&req_);
    } else {
        // Normal exit: complete the operation.
        MPI_Wait(&req_, MPI_STATUS_IGNORE);
    }

    req_ = MPI_REQUEST_NULL;
}

// ─── Move Semantics ──────────────────────────────────────────────────────────

Request_Guard::Request_Guard(Request_Guard &&other) noexcept
    : req_{std::exchange(other.req_, MPI_REQUEST_NULL)}, uncaught_on_entry_{other.uncaught_on_entry_} {}

Request_Guard &Request_Guard::operator=(Request_Guard &&other) noexcept {
    if (this != &other) {
        // Complete or cancel the currently held request before taking the new one.
        if (req_ != MPI_REQUEST_NULL) {
            if (std::uncaught_exceptions() > uncaught_on_entry_) {
                MPI_Cancel(&req_);
                MPI_Request_free(&req_);
            } else {
                MPI_Wait(&req_, MPI_STATUS_IGNORE);
            }
        }
        req_ = std::exchange(other.req_, MPI_REQUEST_NULL);
        uncaught_on_entry_ = other.uncaught_on_entry_;
    }
    return *this;
}

// ─── Test ────────────────────────────────────────────────────────────────────

bool Request_Guard::test() {
    if (req_ == MPI_REQUEST_NULL) {
        return true;
    }

    int flag = 0;
    MPI_Test(&req_, &flag, MPI_STATUS_IGNORE);
    return flag != 0;
}

// ─── Wait ────────────────────────────────────────────────────────────────────

void Request_Guard::wait() {
    if (req_ == MPI_REQUEST_NULL) {
        return;
    }

    MPI_Wait(&req_, MPI_STATUS_IGNORE);
}

// ─── Handle Accessor ─────────────────────────────────────────────────────────

MPI_Request *Request_Guard::handle() noexcept {
    return &req_;
}

}  // namespace halo
