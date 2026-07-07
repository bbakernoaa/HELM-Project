// DAGR — parse_pipeline.cpp
// YAML pipeline parser implementation.
// Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 2.11,
//               2.12, 2.13, 8.4, 8.5, 8.6, 8.7, 8.9, 10.7

#include <algorithm>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "conf/conf.hpp"
#include "dagr/pipeline_config.hpp"
#include "logs/logs.hpp"
#include "shared_logger.hpp"

namespace dagr {

namespace {

/// parse_pipeline diagnostics route through DAGR's single shared LOGS logger
/// (configured once via dagr::configure_logging).
logs::Logger &logger() {
    return detail::shared_logger();
}

/// Parse a snapshot_interval string like "86400s", "3600s", "100s" into a
/// tick::Duration. Supported suffix: 's' (seconds). The value before the
/// suffix must be a positive integer.
/// Returns tick::Duration{0} on parse failure (caller validates positivity).
tick::Duration parse_snapshot_interval(const std::string &raw) {
    if (raw.empty()) {
        return tick::Duration{0};
    }

    std::string_view sv{raw};
    // Strip trailing 's' suffix if present
    if (sv.back() == 's' || sv.back() == 'S') {
        sv.remove_suffix(1);
    }

    if (sv.empty()) {
        return tick::Duration{0};
    }

    // Parse integer seconds
    std::int64_t value = 0;
    bool negative = false;
    std::size_t start = 0;

    if (sv[0] == '-') {
        negative = true;
        start = 1;
    } else if (sv[0] == '+') {
        start = 1;
    }

    if (start >= sv.size()) {
        return tick::Duration{0};
    }

    for (std::size_t i = start; i < sv.size(); ++i) {
        if (sv[i] < '0' || sv[i] > '9') {
            return tick::Duration{0};  // non-numeric character
        }
        value = value * 10 + (sv[i] - '0');
    }

    if (negative) {
        value = -value;
    }

    return tick::seconds(value);
}

/// Map a temporal_profile string to the enum. Returns nullopt on unrecognized.
std::optional<Temporal_Profile> parse_temporal_profile(const std::string &str) {
    if (str == "linear") return Temporal_Profile::linear;
    if (str == "step") return Temporal_Profile::step;
    return std::nullopt;
}

/// Map an out_of_bounds_policy string to the enum. Returns nullopt on unrecognized.
std::optional<OutOfBounds_Policy> parse_oob_policy(const std::string &str) {
    if (str == "clamp") return OutOfBounds_Policy::clamp;
    if (str == "cycle") return OutOfBounds_Policy::cycle;
    return std::nullopt;
}

}  // anonymous namespace

Pipeline_Config parse_pipeline(const std::filesystem::path &yaml_path) {
    // Req 2.1, 2.2: Load YAML via conf::Config::from_file.
    // Allow conf::Conf_Error to propagate on missing file or malformed YAML.
    auto cfg = conf::Config::from_file(yaml_path.string());

    Pipeline_Config result;

    // ── Settings extraction ──────────────────────────────────────────────────
    if (cfg.has("settings.max_concurrency")) {
        result.max_concurrency = static_cast<std::uint32_t>(cfg.get_int("settings.max_concurrency"));
    }
    if (cfg.has("settings.deadlock_timeout")) {
        result.deadlock_timeout_s = static_cast<std::uint32_t>(cfg.get_int("settings.deadlock_timeout"));
    }
    if (cfg.has("settings.shutdown_timeout")) {
        result.shutdown_timeout_s = static_cast<std::uint32_t>(cfg.get_int("settings.shutdown_timeout"));
    }

    // ── Streams extraction (Req 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9) ─────────
    std::unordered_set<std::string> seen_names;
    const std::size_t stream_count = cfg.size("streams");

    for (std::size_t i = 0; i < stream_count; ++i) {
        const std::string prefix = "streams." + std::to_string(i);

        // Extract stream name
        if (!cfg.has(prefix + ".name")) {
            logger().log(logs::Severity_Level::WARNING, "parse_pipeline: stream at index " + std::to_string(i) + " is missing required field 'name'");
            throw std::invalid_argument("stream at index " + std::to_string(i) + ": missing required field 'name'");
        }
        const std::string name = cfg.get_string(prefix + ".name");

        // Req 2.9: Validate no duplicate stream names
        if (seen_names.count(name) > 0) {
            logger().log(logs::Severity_Level::WARNING, "parse_pipeline: duplicate stream name '" + name + "'");
            throw std::invalid_argument("duplicate stream name: '" + name + "'");
        }
        seen_names.insert(name);

        // Req 2.8: Check required fields exist
        if (!cfg.has(prefix + ".temporal_profile")) {
            logger().log(logs::Severity_Level::WARNING, "parse_pipeline: stream '" + name + "' missing required field 'temporal_profile'");
            throw std::invalid_argument("stream '" + name + "': missing required field 'temporal_profile'");
        }
        if (!cfg.has(prefix + ".out_of_bounds_policy")) {
            logger().log(logs::Severity_Level::WARNING, "parse_pipeline: stream '" + name + "' missing required field 'out_of_bounds_policy'");
            throw std::invalid_argument("stream '" + name + "': missing required field 'out_of_bounds_policy'");
        }
        if (!cfg.has(prefix + ".dataset_path")) {
            logger().log(logs::Severity_Level::WARNING, "parse_pipeline: stream '" + name + "' missing required field 'dataset_path'");
            throw std::invalid_argument("stream '" + name + "': missing required field 'dataset_path'");
        }
        if (!cfg.has(prefix + ".snapshot_interval")) {
            logger().log(logs::Severity_Level::WARNING, "parse_pipeline: stream '" + name + "' missing required field 'snapshot_interval'");
            throw std::invalid_argument("stream '" + name + "': missing required field 'snapshot_interval'");
        }

        // Req 2.6, 8.4, 8.5: Parse temporal_profile
        const std::string tp_str = cfg.get_string(prefix + ".temporal_profile");
        auto tp = parse_temporal_profile(tp_str);
        if (!tp.has_value()) {
            logger().log(logs::Severity_Level::WARNING, "parse_pipeline: stream '" + name + "' has unrecognized temporal_profile '" + tp_str + "'");
            throw std::invalid_argument("stream '" + name + "': unrecognized temporal_profile '" + tp_str + "'");
        }

        // Req 2.7, 8.6, 8.7: Parse out_of_bounds_policy
        const std::string oob_str = cfg.get_string(prefix + ".out_of_bounds_policy");
        auto oob = parse_oob_policy(oob_str);
        if (!oob.has_value()) {
            logger().log(logs::Severity_Level::WARNING,
                         "parse_pipeline: stream '" + name + "' has unrecognized out_of_bounds_policy '" + oob_str + "'");
            throw std::invalid_argument("stream '" + name + "': unrecognized out_of_bounds_policy '" + oob_str + "'");
        }

        // Req 2.5: Extract dataset_path
        const std::string ds_path = cfg.get_string(prefix + ".dataset_path");
        // Req 8.9: Validate dataset_path is not empty
        if (ds_path.empty()) {
            logger().log(logs::Severity_Level::WARNING, "parse_pipeline: stream '" + name + "' has empty dataset_path");
            throw std::invalid_argument("stream '" + name + "': dataset_path must not be empty");
        }

        // Req 2.5: Extract snapshot_interval as opaque duration token
        const std::string interval_str = cfg.get_string(prefix + ".snapshot_interval");
        tick::Duration interval = parse_snapshot_interval(interval_str);
        // Req 8.9: Validate snapshot_interval is positive (> 0 nanos)
        if (interval.nanos() <= 0) {
            logger().log(logs::Severity_Level::WARNING, "parse_pipeline: stream '" + name + "' has invalid snapshot_interval '" + interval_str + "'");
            throw std::invalid_argument("stream '" + name + "': snapshot_interval must be positive (got '" + interval_str + "')");
        }

        // Req 2.13: Do NOT validate dataset_path existence at parse time.

        // Build Stream_Descriptor (Req 2.10: preserve declaration order)
        Stream_Descriptor desc;
        desc.name = name;
        desc.temporal_profile = tp.value();
        desc.oob_policy = oob.value();
        desc.dataset_path = std::filesystem::path{ds_path};
        desc.snapshot_interval = interval;

        result.streams.push_back(std::move(desc));
    }

    // ── Tasks extraction ─────────────────────────────────────────────────────
    const std::size_t task_count = cfg.size("tasks");
    std::unordered_map<std::string, std::uint32_t> task_index;

    for (std::size_t i = 0; i < task_count; ++i) {
        const std::string prefix = "tasks." + std::to_string(i);
        const std::string task_name = cfg.get_string(prefix + ".name");
        task_index[task_name] = static_cast<std::uint32_t>(i);
        result.task_names.push_back(task_name);
    }

    // ── Dependencies extraction and DAG construction ─────────────────────────
    const std::size_t dep_count = cfg.size("dependencies");

    // Adjacency list for topological sort: adj[producer] -> list of consumers
    std::vector<std::vector<std::uint32_t>> adj(task_count);
    std::vector<std::uint32_t> in_degree(task_count, 0);

    for (std::size_t i = 0; i < dep_count; ++i) {
        const std::string prefix = "dependencies." + std::to_string(i);
        const std::string from_name = cfg.get_string(prefix + ".from");
        const std::string to_name = cfg.get_string(prefix + ".to");

        // Look up task indices
        auto from_it = task_index.find(from_name);
        auto to_it = task_index.find(to_name);

        if (from_it == task_index.end()) {
            logger().log(logs::Severity_Level::WARNING, "parse_pipeline: dependency references unknown task '" + from_name + "'");
            throw std::invalid_argument("dependency references unknown task '" + from_name + "'");
        }
        if (to_it == task_index.end()) {
            logger().log(logs::Severity_Level::WARNING, "parse_pipeline: dependency references unknown task '" + to_name + "'");
            throw std::invalid_argument("dependency references unknown task '" + to_name + "'");
        }

        const std::uint32_t producer_id = from_it->second;
        const std::uint32_t consumer_id = to_it->second;

        Dependency_Edge edge{producer_id, consumer_id};
        result.edges.push_back(edge);

        adj[producer_id].push_back(consumer_id);
        in_degree[consumer_id] += 1;
    }

    // ── Req 2.11: Kahn's topological sort to validate acyclicity ─────────────
    if (task_count > 0) {
        std::deque<std::uint32_t> queue;
        for (std::uint32_t n = 0; n < static_cast<std::uint32_t>(task_count); ++n) {
            if (in_degree[n] == 0) {
                queue.push_back(n);
            }
        }

        std::uint32_t sorted_count = 0;
        while (!queue.empty()) {
            std::uint32_t node = queue.front();
            queue.pop_front();
            ++sorted_count;

            for (std::uint32_t downstream : adj[node]) {
                in_degree[downstream] -= 1;
                if (in_degree[downstream] == 0) {
                    queue.push_back(downstream);
                }
            }
        }

        if (sorted_count != static_cast<std::uint32_t>(task_count)) {
            // Cycle detected — find at least two nodes still with in_degree > 0
            // to report in the error message.
            std::string cycle_path;
            for (std::uint32_t n = 0; n < static_cast<std::uint32_t>(task_count); ++n) {
                if (in_degree[n] > 0) {
                    if (!cycle_path.empty()) {
                        cycle_path += " -> ";
                    }
                    cycle_path += result.task_names[n];
                    // Report at least two nodes involved in the cycle
                    if (cycle_path.find(" -> ") != std::string::npos) {
                        break;
                    }
                }
            }

            logger().log(logs::Severity_Level::WARNING, "parse_pipeline: cycle detected in task dependencies: " + cycle_path);
            throw std::invalid_argument("cycle detected in task dependencies: " + cycle_path);
        }
    }

    return result;
}

}  // namespace dagr
