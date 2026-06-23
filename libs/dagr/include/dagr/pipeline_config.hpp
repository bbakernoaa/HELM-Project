// DAGR — pipeline_config.hpp
// Pipeline_Config, Stream_Descriptor, enums, and parse_pipeline declaration.
// Requirements: 8.1, 8.2, 8.3, 8.8, 2.12
#pragma once

#include <cstdint>
#include <compare>
#include <filesystem>
#include <string>
#include <vector>

#include "tick/duration.hpp"

namespace dagr {

/// Temporal interpolation strategy for a stream's bookend blending.
enum class Temporal_Profile : std::uint8_t {
    linear = 0,  ///< Linear interpolation between bookends
    step   = 1   ///< Step function (nearest-neighbor select)
};

/// Out-of-bounds remapping policy when simulation time exceeds dataset coverage.
enum class OutOfBounds_Policy : std::uint8_t {
    clamp = 0,   ///< Clamp to boundary snapshot
    cycle = 1    ///< Cycle within the last year of coverage
};

/// Parsed representation of a single I/O stream from the pipeline YAML.
/// Value type: copyable, movable, equality-comparable.
struct Stream_Descriptor {
    std::string            name;               ///< Stream identifier (max 128 chars)
    Temporal_Profile       temporal_profile;    ///< Interpolation strategy
    OutOfBounds_Policy     oob_policy;         ///< Out-of-bounds handling
    std::filesystem::path  dataset_path;       ///< Path to dataset (validated at runtime by AMIO)
    tick::Duration         snapshot_interval;   ///< Duration between consecutive snapshots

    bool operator==(const Stream_Descriptor&) const = default;
};

/// Adjacency list edge: from producer_id → consumer_id.
struct Dependency_Edge {
    std::uint32_t producer_id;
    std::uint32_t consumer_id;
};

/// Complete parsed pipeline configuration.
struct Pipeline_Config {
    std::vector<Stream_Descriptor> streams;         ///< Ordered stream declarations
    std::vector<std::string>       task_names;      ///< Task identifiers (index = node ID)
    std::vector<Dependency_Edge>   edges;           ///< DAG adjacency list
    std::uint32_t                  max_concurrency{64};    ///< [1, 1024]
    std::uint32_t                  deadlock_timeout_s{30}; ///< [1, 3600] seconds
    std::uint32_t                  shutdown_timeout_s{30}; ///< Rank reclamation timeout
};

/// Parse a pipeline YAML file into a Pipeline_Config.
/// @param yaml_path  Path to the YAML pipeline configuration file
/// @return Fully validated Pipeline_Config (acyclic, no duplicates, all fields present)
/// @throws std::invalid_argument on validation failure (cycles, duplicates, missing fields)
/// @throws conf::Conf_Error on YAML parse failure (propagated from conf::Config::from_file)
[[nodiscard]] Pipeline_Config parse_pipeline(const std::filesystem::path& yaml_path);

} // namespace dagr
