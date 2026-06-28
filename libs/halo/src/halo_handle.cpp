#include <halo/halo_handle.hpp>
#include <utility>

namespace halo {

// ─── Destructor ─────────────────────────────────────────────────────────────

Halo_Handle::~Halo_Handle() {
    // Ensure all pending operations complete before resources are released.
    // This guarantees that MPI requests are not left dangling and that
    // post-receive deep-copies execute even if the user forgets to call wait().
    if (!empty()) {
        wait();
    }
}

// ─── Move Constructor ───────────────────────────────────────────────────────

Halo_Handle::Halo_Handle(Halo_Handle &&other) noexcept : requests_(std::move(other.requests_)), staged_recv_(std::move(other.staged_recv_)) {
    // Source is now in a valid empty state:
    // - other.requests_ is moved-from (valid but unspecified, typically empty)
    // - other.staged_recv_ is nullptr after move
    // Explicitly clear the source's vector to guarantee empty state.
    other.requests_.clear();
}

// ─── Move Assignment ────────────────────────────────────────────────────────

Halo_Handle &Halo_Handle::operator=(Halo_Handle &&other) noexcept {
    if (this != &other) {
        // Complete any pending operations on the current handle before
        // taking ownership of the other handle's operations.
        if (!empty()) {
            wait();
        }

        requests_ = std::move(other.requests_);
        staged_recv_ = std::move(other.staged_recv_);

        // Ensure source is in a valid empty state.
        other.requests_.clear();
    }
    return *this;
}

// ─── test() ─────────────────────────────────────────────────────────────────

bool Halo_Handle::test() {
    if (empty()) {
        return true;
    }

    // Check all requests non-blockingly.
    for (auto &req : requests_) {
        if (!req.test()) {
            // At least one operation is still pending.
            return false;
        }
    }

    // All operations complete — perform post-receive deep-copy if needed.
    finalize_staged_recv_();
    return true;
}

// ─── wait() ─────────────────────────────────────────────────────────────────

void Halo_Handle::wait() {
    if (empty()) {
        return;
    }

    // Block until all requests complete.
    for (auto &req : requests_) {
        req.wait();
    }

    // All operations complete — perform post-receive deep-copy if needed.
    finalize_staged_recv_();
}

// ─── empty() ────────────────────────────────────────────────────────────────

bool Halo_Handle::empty() const noexcept {
    return requests_.empty();
}

// ─── finalize_staged_recv_() ────────────────────────────────────────────────

void Halo_Handle::finalize_staged_recv_() {
    if (staged_recv_ && !staged_recv_->completed) {
        staged_recv_->post_recv_copy();
        staged_recv_->completed = true;
    }
}

}  // namespace halo
