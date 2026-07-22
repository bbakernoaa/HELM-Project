// ─── HALO exchange_indexed MPI-failure unit test (mocked MPI spy) ────────────
// Feature: cpp-dycore-halo-exchange, Task 2.7
//
// Example-based unit test for the exchange_indexed error path:
//   - When an MPI operation returns a failure code, exchange_indexed raises an
//     error identifying the failing operation and the neighbor rank involved
//                                                                      (Req 2.5)
//
// This assertion needs a deterministic MPI failure, so it links the MPI
// interposition spy (mpi_interposition.cpp) rather than the real MPI runtime:
// the spy records intercepted calls and injects a chosen error code into the
// next MPI call. The value-moving infrastructure checks (diagnostics, tag
// equivalence, no-op) live in the sibling real-MPI target
// (test_exchange_indexed_infra), because the spy does not move data.
//
// The spy overrides MPI_Comm_rank -> 0 and MPI_Comm_size -> 4, so ranks in
// [0, 4) are valid without launching mpirun. exchange_indexed routes failures
// through the communicator-aware handle_mpi_error overload, which queries the
// communicator name; the spy does not override MPI_Comm_get_name, so this
// translation unit supplies a local override returning an empty name to keep
// the test self-contained (no real MPI runtime required).
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <mpi.h>

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "halo/communicator.hpp"
#include "halo/exchange_indexed.hpp"
#include "halo/indexed_halo_plan.hpp"
#include "mpi_interposition.hpp"

// Local override: the spy does not intercept MPI_Comm_get_name, and no real MPI
// runtime is initialized in this target. handle_mpi_error queries the comm name
// on the failure path, so provide a benign empty-name override here.
extern "C" int MPI_Comm_get_name(MPI_Comm /*comm*/, char *comm_name, int *resultlen) {
    if (comm_name != nullptr) comm_name[0] = '\0';
    if (resultlen != nullptr) *resultlen = 0;
    return MPI_SUCCESS;
}

namespace {

using HostView = Kokkos::View<double *, Kokkos::HostSpace>;

/// The mocked MPI_Comm_size returns 4, so valid ranks are [0, 4).
constexpr int MOCK_COMM_SIZE = 4;

}  // namespace

// ─── Req 2.5: MPI receive failure names the operation and neighbor rank ──────
TEST(ExchangeIndexedMpiFail, IrecvFailureNamesOperationAndNeighborRank) {
    auto &spy = halo::testing::MPI_Spy::instance();
    spy.reset();

    halo::Communicator comm(MPI_COMM_WORLD);

    // Receive neighbor rank 2, send neighbor rank 3 (both valid in [0, 4) and
    // distinct from the local rank 0 so the neighbor rank is unambiguous).
    const int recv_rank = 2;
    const int send_rank = 3;
    halo::Indexed_Neighbor send_nbr{send_rank, {{0, 1}}};
    halo::Indexed_Neighbor recv_nbr{recv_rank, {{10, 11}}};
    halo::Indexed_Halo_Plan plan(comm, halo::Element_Kind::cell, {send_nbr}, {recv_nbr});

    HostView field("mpifail_r1", 20);
    for (std::size_t i = 0; i < field.extent(0); ++i) field(i) = static_cast<double>(i);

    const int layer0[] = {0};

    // exchange_indexed posts all receives before any send, so the first MPI
    // call is MPI_Irecv for recv_rank; inject the failure there.
    spy.reset();
    spy.set_next_error(MPI_ERR_OTHER);

    bool caught = false;
    std::string msg;
    try {
        halo::exchange_indexed(plan, field, std::span<const int>(layer0, 1));
    } catch (const std::runtime_error &e) {
        caught = true;
        msg = e.what();
    }

    ASSERT_TRUE(caught) << "expected std::runtime_error on MPI failure";
    EXPECT_NE(msg.find("MPI_Irecv"), std::string::npos) << "error must name the failing operation: " << msg;
    EXPECT_NE(msg.find("rank " + std::to_string(recv_rank)), std::string::npos) << "error must name the neighbor rank: " << msg;
}

// ─── Req 2.5: MPI send failure names the operation and neighbor rank ─────────
// A send-only plan makes MPI_Isend the first (and failing) MPI call, exercising
// the send-side branch of the error path with a distinct operation name.
TEST(ExchangeIndexedMpiFail, IsendFailureNamesOperationAndNeighborRank) {
    auto &spy = halo::testing::MPI_Spy::instance();
    spy.reset();

    halo::Communicator comm(MPI_COMM_WORLD);

    const int send_rank = 3;
    halo::Indexed_Neighbor send_nbr{send_rank, {{0, 1}}};
    halo::Indexed_Halo_Plan plan(comm, halo::Element_Kind::cell, {send_nbr}, /*recv=*/{});

    HostView field("mpifail_send_only", 20);
    for (std::size_t i = 0; i < field.extent(0); ++i) field(i) = static_cast<double>(i);

    const int layer0[] = {0};

    spy.reset();
    spy.set_next_error(MPI_ERR_OTHER);

    bool caught = false;
    std::string msg;
    try {
        halo::exchange_indexed(plan, field, std::span<const int>(layer0, 1));
    } catch (const std::runtime_error &e) {
        caught = true;
        msg = e.what();
    }

    ASSERT_TRUE(caught) << "expected std::runtime_error on MPI failure";
    EXPECT_NE(msg.find("MPI_Isend"), std::string::npos) << "error must name the failing operation: " << msg;
    EXPECT_NE(msg.find("rank " + std::to_string(send_rank)), std::string::npos) << "error must name the neighbor rank: " << msg;
}

// ─── Kokkos environment (mocked MPI needs no MPI_Init) ───────────────────────
namespace {

class KokkosEnvironment : public ::testing::Environment {
   public:
    void SetUp() override {
        if (!Kokkos::is_initialized()) Kokkos::initialize();
    }
    void TearDown() override {
        if (Kokkos::is_initialized()) Kokkos::finalize();
    }
};

}  // namespace

static ::testing::Environment *const kokkos_env = ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
