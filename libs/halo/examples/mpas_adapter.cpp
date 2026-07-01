/// @file mpas_adapter.cpp
/// @brief Example: MPAS unstructured mesh cell halo exchange using HALO.
///
/// This example demonstrates how to replace MPAS's legacy `exchange_halo`
/// routine with HALO's flat-buffer API (Halo_Plan + exchange_blocking).
///
/// ## Why the flat-buffer API?
///
/// MPAS uses an unstructured Voronoi mesh where each partition has an
/// irregular set of neighbors with variable-length halos. This maps
/// naturally to Halo_Plan's flat-buffer model:
///
///   MPAS concept               → HALO equivalent
///   ─────────────────────────────────────────────────────────────
///   exchangeList%endPointID    → Neighbor_Info::rank
///   exchangeList%nList         → Neighbor_Info::count
///   exchange_halo(field)       → exchange_blocking(plan, view)
///
/// ## Buffer layout (matches MPAS convention)
///
///   |<-- send to nbr 0 -->|<-- send to 1 -->|<-- owned -->|<-- recv 0 -->|<-- recv 1 -->|
///   |      count[0]       |    count[1]     |             |   count[0]   |   count[1]   |
///
/// ## Build
///
/// See the accompanying CMakeLists.txt for standalone build instructions
/// using find_package(HALO).

#include <Kokkos_Core.hpp>
#include <halo/communicator.hpp>
#include <halo/environment.hpp>
#include <halo/exchange.hpp>
#include <halo/halo_plan.hpp>
#include <iostream>
#include <vector>

/// @brief Simulated MPAS exchange-list entry (mirrors mpas_dmpar types).
struct MPAS_Exchange_Entry {
    int endpoint_rank;  ///< MPI rank of the neighbor partition.
    int n_elements;     ///< Number of cells to send/receive.
};

/// @brief Build a HALO plan from MPAS-style exchange lists.
///
/// Translates MPAS's exchangeList arrays into HALO's Neighbor_Info
/// vectors and constructs an immutable Halo_Plan.
halo::Halo_Plan build_mpas_halo_plan(const halo::Communicator &comm, const std::vector<MPAS_Exchange_Entry> &send_list,
                                     const std::vector<MPAS_Exchange_Entry> &recv_list) {
    std::vector<halo::Neighbor_Info> send_neighbors;
    send_neighbors.reserve(send_list.size());
    for (const auto &entry : send_list) {
        send_neighbors.push_back({entry.endpoint_rank, static_cast<std::size_t>(entry.n_elements)});
    }

    std::vector<halo::Neighbor_Info> recv_neighbors;
    recv_neighbors.reserve(recv_list.size());
    for (const auto &entry : recv_list) {
        recv_neighbors.push_back({entry.endpoint_rank, static_cast<std::size_t>(entry.n_elements)});
    }

    return halo::Halo_Plan(comm, std::move(send_neighbors), std::move(recv_neighbors));
}

int main(int argc, char *argv[]) {
    // ── MPI + Kokkos initialization ─────────────────────────────────────────
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    Kokkos::initialize(argc, argv);
    {
        halo::Environment::initialize();

        // Wrap MPI_COMM_WORLD (predefined comms are not freed by HALO).
        halo::Communicator comm(MPI_COMM_WORLD);
        const int my_rank = comm.rank();
        const int num_ranks = comm.size();

        // ── Simulated MPAS mesh connectivity ────────────────────────────────
        //
        // In a real MPAS integration, these come from:
        //   domain % exchangeList_send / exchangeList_recv
        // populated during mpas_dmpar_init_multihalo_exchange().
        //
        // Here we simulate a ring topology: each rank sends to its right
        // neighbor and receives from its left, exchanging 100 cells each.

        const int left_neighbor = (my_rank - 1 + num_ranks) % num_ranks;
        const int right_neighbor = (my_rank + 1) % num_ranks;
        const int cells_per_halo = 100;

        std::vector<MPAS_Exchange_Entry> send_list = {{right_neighbor, cells_per_halo}};
        std::vector<MPAS_Exchange_Entry> recv_list = {{left_neighbor, cells_per_halo}};

        // ── Build the HALO plan (done once during model setup) ──────────────
        auto plan = build_mpas_halo_plan(comm, send_list, recv_list);

        // ── Allocate the cell field as a Kokkos::View ───────────────────────
        //
        // Layout: [send_halo | owned_cells | recv_halo]
        //
        // MPAS convention: nCellsSolve = owned, nCells = owned + halo.
        // The flat-buffer API expects send data at the front and receive
        // slots appended after owned.

        const std::size_t n_owned = 1000;
        const std::size_t total_cells = plan.total_send_elements() + n_owned + plan.total_recv_elements();

        using view_t = Kokkos::View<double *, Kokkos::HostSpace>;
        view_t temperature("temperature", total_cells);

        // Fill owned cells with rank-specific data (simulates model physics).
        const std::size_t send_end = plan.total_send_elements();
        const std::size_t own_end = send_end + n_owned;
        for (std::size_t i = send_end; i < own_end; ++i) {
            temperature(i) = static_cast<double>(my_rank * 1000 + i);
        }

        // Copy owned boundary cells into the send region.
        // In MPAS, the send region contains copies of owned cells that
        // neighbors need. Here we just mirror the first cells_per_halo
        // owned cells into the send slot.
        for (std::size_t i = 0; i < plan.total_send_elements(); ++i) {
            temperature(i) = temperature(send_end + i);
        }

        // ── Execute the halo exchange ───────────────────────────────────────
        //
        // This single call replaces the MPAS pattern:
        //   call mpas_dmpar_exch_halo_field(field)
        //
        // Under the hood, HALO posts MPI_Irecv for each recv neighbor,
        // then MPI_Isend for each send neighbor, then MPI_Waitall.
        halo::exchange_blocking(plan, temperature);

        // ── Verify: recv region now contains data from left neighbor ─────────
        const std::size_t recv_start = own_end;
        if (num_ranks > 1) {
            std::cout << "[Rank " << my_rank << "] " << "First received cell value = " << temperature(recv_start) << "\n";
        } else {
            std::cout << "[Rank 0] Single-rank: recv mirrors own data = " << temperature(recv_start) << "\n";
        }
    }
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
