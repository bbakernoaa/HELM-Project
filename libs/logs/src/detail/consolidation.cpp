/// @file detail/consolidation.cpp
/// @brief Consolidation engine + range compaction.
///
/// Requirements: 4.2, 4.5, 4.7, 4.8, 4.9
///
/// consolidate_collective:
///   Each rank serializes its buffered keys (severity as int + message string).
///   Uses MPI_Gather to collect sizes, then MPI_Gatherv to collect serialized
///   key data + ranks on root rank 0. Root groups by key, calls compact_ranges
///   on contributing ranks per key, and produces Consolidated_Records. Non-root
///   ranks return empty. All MPI calls serialized through Serialized_MPI_Guard.
///   Transports ONLY keys and ranks (no source location, stack trace, or context).
///
/// consolidate_local:
///   Groups records by consolidation key locally. Annotates each representative
///   with a single-rank Rank_Range bearing the sentinel rank -1. Issues NO MPI.

#include "logs/detail/consolidation.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <vector>

#include "logs/detail/serialized_mpi_guard.hpp"

namespace logs::detail {

std::string Rank_Range::to_string() const {
    return std::to_string(first) + "-" + std::to_string(last);
}

std::vector<Rank_Range> compact_ranges(std::vector<int> ranks) {
    if (ranks.empty()) {
        return {};
    }

    std::sort(ranks.begin(), ranks.end());
    ranks.erase(std::unique(ranks.begin(), ranks.end()), ranks.end());

    std::vector<Rank_Range> out;
    int first = ranks[0];
    int last = first;

    for (std::size_t i = 1; i < ranks.size(); ++i) {
        if (ranks[i] == last + 1) {
            last = ranks[i];
        } else {
            out.push_back(Rank_Range{first, last});
            first = ranks[i];
            last = first;
        }
    }
    out.push_back(Rank_Range{first, last});

    return out;
}

// ─── Serialization helpers (length-prefixed format) ─────────────────────────
//
// Per-record wire format:
//   [4 bytes: severity as int32]
//   [4 bytes: message length as int32]
//   [N bytes: message characters]
//   [4 bytes: contributing rank as int32]
//
// This transports ONLY the consolidation key (severity + message) and the
// originating rank. No source location, stack trace, or context labels.

namespace {

struct KeyCompare {
    bool operator()(const Consolidation_Key &a, const Consolidation_Key &b) const {
        if (a.severity != b.severity) {
            return static_cast<int>(a.severity) < static_cast<int>(b.severity);
        }
        return a.message < b.message;
    }
};

/// Serialize a single record's key + rank into a byte buffer.
void serialize_record(const Log_Record &record, int rank, std::vector<char> &buffer) {
    int severity = static_cast<int>(record.severity());
    int msg_len = static_cast<int>(record.message().size());

    // Append severity (4 bytes)
    buffer.insert(buffer.end(), reinterpret_cast<const char *>(&severity), reinterpret_cast<const char *>(&severity) + sizeof(int));

    // Append message length (4 bytes)
    buffer.insert(buffer.end(), reinterpret_cast<const char *>(&msg_len), reinterpret_cast<const char *>(&msg_len) + sizeof(int));

    // Append message characters
    buffer.insert(buffer.end(), record.message().data(), record.message().data() + msg_len);

    // Append rank (4 bytes)
    buffer.insert(buffer.end(), reinterpret_cast<const char *>(&rank), reinterpret_cast<const char *>(&rank) + sizeof(int));
}

/// Deserialize all records from a gathered buffer into key->ranks map.
void deserialize_records(const std::vector<char> &buffer, std::map<Consolidation_Key, std::vector<int>, KeyCompare> &groups) {
    std::size_t offset = 0;
    while (offset + 3 * sizeof(int) <= buffer.size()) {
        // Read severity
        int severity = 0;
        std::memcpy(&severity, buffer.data() + offset, sizeof(int));
        offset += sizeof(int);

        // Read message length
        int msg_len = 0;
        std::memcpy(&msg_len, buffer.data() + offset, sizeof(int));
        offset += sizeof(int);

        // Bounds check
        if (msg_len < 0 || offset + static_cast<std::size_t>(msg_len) + sizeof(int) > buffer.size()) {
            break;
        }

        // Read message
        std::string message(buffer.data() + offset, static_cast<std::size_t>(msg_len));
        offset += static_cast<std::size_t>(msg_len);

        // Read rank
        int rank = 0;
        std::memcpy(&rank, buffer.data() + offset, sizeof(int));
        offset += sizeof(int);

        // Group by key
        Consolidation_Key key{static_cast<Severity_Level>(severity), std::move(message)};
        groups[key].push_back(rank);
    }
}

}  // anonymous namespace

std::vector<Consolidated_Record> Consolidation_Engine::consolidate_collective(std::span<const Log_Record> buffered, Mpi_Environment &env) noexcept {
    try {
        // Acquire the serialization guard for all MPI calls.
        Serialized_MPI_Guard guard(env);

        MPI_Comm comm = env.communicator();
        int my_rank = env.rank();

        // Get communicator size.
        int comm_size = 0;
        if (MPI_Comm_size(comm, &comm_size) != MPI_SUCCESS) {
            return {};
        }

        // Step 1: Serialize local buffered records (key + rank only).
        std::vector<char> local_data;
        for (const auto &record : buffered) {
            serialize_record(record, my_rank, local_data);
        }

        int local_size = static_cast<int>(local_data.size());

        // Step 2: Gather sizes on root (rank 0).
        std::vector<int> all_sizes;
        if (my_rank == 0) {
            all_sizes.resize(static_cast<std::size_t>(comm_size));
        }

        if (MPI_Gather(&local_size, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, 0, comm) != MPI_SUCCESS) {
            return {};
        }

        // Step 3: Gather serialized data on root using MPI_Gatherv.
        std::vector<int> displacements;
        std::vector<char> all_data;

        if (my_rank == 0) {
            displacements.resize(static_cast<std::size_t>(comm_size));
            int total = 0;
            for (int i = 0; i < comm_size; ++i) {
                displacements[static_cast<std::size_t>(i)] = total;
                total += all_sizes[static_cast<std::size_t>(i)];
            }
            all_data.resize(static_cast<std::size_t>(total));
        }

        if (MPI_Gatherv(local_data.data(), local_size, MPI_CHAR, all_data.data(), all_sizes.data(), displacements.data(), MPI_CHAR, 0, comm) !=
            MPI_SUCCESS) {
            return {};
        }

        // Step 4: Root deserializes and groups by key, then compacts ranges.
        if (my_rank != 0) {
            // Non-root returns empty — only root produces representatives.
            return {};
        }

        // Deserialize all gathered data.
        std::map<Consolidation_Key, std::vector<int>, KeyCompare> groups;
        deserialize_records(all_data, groups);

        // Build consolidated records.
        std::vector<Consolidated_Record> result;
        result.reserve(groups.size());

        for (auto &[key, ranks] : groups) {
            auto ranges = compact_ranges(std::move(ranks));
            int count = 0;
            for (const auto &r : ranges) {
                count += (r.last - r.first + 1);
            }
            result.push_back(Consolidated_Record{std::move(key), count, std::move(ranges)});
        }

        return result;
    } catch (...) {
        // Exception boundary: absorb any failure, return empty.
        return {};
    }
}

std::vector<Consolidated_Record> Consolidation_Engine::consolidate_local(std::span<const Log_Record> buffered) noexcept {
    // Local (no-communicator) path: consolidate locally, annotate each
    // representative with a single-rank Rank_Range bearing the sentinel
    // rank -1. Issues NO MPI communication (Requirement 4.9).
    try {
        struct LocalKeyCompare {
            bool operator()(const Consolidation_Key &a, const Consolidation_Key &b) const {
                if (a.severity != b.severity) {
                    return static_cast<int>(a.severity) < static_cast<int>(b.severity);
                }
                return a.message < b.message;
            }
        };

        std::map<Consolidation_Key, bool, LocalKeyCompare> seen;

        for (const auto &record : buffered) {
            Consolidation_Key key{record.severity(), std::string(record.message())};
            seen[key] = true;
        }

        std::vector<Consolidated_Record> result;
        result.reserve(seen.size());

        for (auto &[key, _] : seen) {
            // Annotate with sentinel rank -1 Rank_Range (Requirement 4.9).
            // Count how many records matched this key.
            int count = 0;
            for (const auto &record : buffered) {
                if (record.severity() == key.severity && record.message() == key.message) {
                    ++count;
                }
            }
            result.push_back(Consolidated_Record{
                std::move(key), count, std::vector<Rank_Range>{{-1, -1}}  // Sentinel rank -1
            });
        }

        return result;
    } catch (...) {
        return {};
    }
}

}  // namespace logs::detail
