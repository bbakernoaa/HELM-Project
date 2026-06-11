#ifndef LOGS_DETAIL_CONSOLIDATION_HPP
#define LOGS_DETAIL_CONSOLIDATION_HPP

/// @file detail/consolidation.hpp
/// @brief Consolidation_Key, Rank_Range, compact_ranges, and Consolidation_Engine.

#include "logs/log_record.hpp"
#include "logs/detail/mpi_environment.hpp"
#include "logs/detail/serialized_mpi_guard.hpp"

#include <span>
#include <string>
#include <vector>

namespace logs::detail {

/// Identity used for consolidation: severity + payload, EXCLUDING rank.
struct Consolidation_Key {
    Severity_Level severity;
    std::string    message;

    bool operator==(const Consolidation_Key&) const = default;
};

/// A maximal contiguous span of contributing ranks.
struct Rank_Range {
    int first;
    int last;
    [[nodiscard]] std::string to_string() const;
};

/// Partition a set of contributing ranks into maximal contiguous spans.
[[nodiscard]] std::vector<Rank_Range> compact_ranges(std::vector<int> ranks);

/// Representative consolidated record.
struct Consolidated_Record {
    Consolidation_Key       key;
    int                     rank_count;
    std::vector<Rank_Range> ranges;
};

/// Cross-rank consolidation engine.
class Consolidation_Engine {
public:
    [[nodiscard]] std::vector<Consolidated_Record>
    consolidate_collective(std::span<const Log_Record> buffered,
                           Mpi_Environment& env) noexcept;

    [[nodiscard]] std::vector<Consolidated_Record>
    consolidate_local(std::span<const Log_Record> buffered) noexcept;
};

} // namespace logs::detail

#endif // LOGS_DETAIL_CONSOLIDATION_HPP
