// DAGR — prop_missing_field.cpp
// Property 11: Missing Required Field Rejection
//
// For any stream entry missing one or more of the four required fields
// (temporal_profile, out_of_bounds_policy, dataset_path, snapshot_interval),
// parse_pipeline SHALL throw std::invalid_argument reporting the stream name
// and the name of the missing field.
//
// **Validates: Requirements 2.8**

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "dagr/pipeline_config.hpp"

namespace {

/// The four required fields per stream entry.
const std::vector<std::string> REQUIRED_FIELDS = {"temporal_profile", "out_of_bounds_policy", "dataset_path", "snapshot_interval"};

/// Generate a random stream name (1–32 lowercase alpha characters).
rc::Gen<std::string> gen_stream_name() {
    return rc::gen::exec([] {
        auto len = *rc::gen::inRange(1, 33);
        std::string name;
        name.reserve(len);
        for (int i = 0; i < len; ++i) {
            name.push_back(*rc::gen::inRange<char>('a', 'z' + 1));
        }
        return name;
    });
}

/// Generate a non-empty subset of field indices to remove (at least 1 field missing).
/// Returns a sorted vector of indices into REQUIRED_FIELDS.
rc::Gen<std::vector<std::size_t>> gen_fields_to_remove() {
    return rc::gen::exec([] {
        // Each of the 4 fields has a chance of being removed
        std::vector<std::size_t> removed;
        for (std::size_t i = 0; i < REQUIRED_FIELDS.size(); ++i) {
            if (*rc::gen::inRange(0, 2) == 1) {
                removed.push_back(i);
            }
        }
        // Guarantee at least one field is removed
        if (removed.empty()) {
            removed.push_back(*rc::gen::inRange<std::size_t>(0, REQUIRED_FIELDS.size()));
        }
        return removed;
    });
}

/// Build a YAML string for a single stream with specified fields present/absent.
/// @param stream_name     Name of the stream
/// @param removed_indices Indices into REQUIRED_FIELDS of fields to omit
std::string build_yaml_with_missing_fields(const std::string &stream_name, const std::vector<std::size_t> &removed_indices) {
    // Full field values (valid when present)
    const std::string field_values[] = {
        "    temporal_profile: \"linear\"\n",     // index 0
        "    out_of_bounds_policy: \"clamp\"\n",  // index 1
        "    dataset_path: \"/data/test.nc\"\n",  // index 2
        "    snapshot_interval: \"3600s\"\n"      // index 3
    };

    std::string yaml = "streams:\n";
    yaml += "  - name: \"" + stream_name + "\"\n";

    // Include only fields whose index is NOT in removed_indices
    for (std::size_t i = 0; i < REQUIRED_FIELDS.size(); ++i) {
        bool is_removed = std::find(removed_indices.begin(), removed_indices.end(), i) != removed_indices.end();
        if (!is_removed) {
            yaml += field_values[i];
        }
    }

    yaml += "tasks:\n";
    yaml += "  - name: \"task_0\"\n";
    yaml += "dependencies: []\n";

    return yaml;
}

/// Helper: create a temporary YAML file and return its path.
std::filesystem::path write_temp_yaml(const std::string &content) {
    static int counter = 0;
    auto path = std::filesystem::temp_directory_path() / ("dagr_prop_missing_field_" + std::to_string(++counter) + ".yaml");
    std::ofstream ofs(path);
    ofs << content;
    ofs.close();
    return path;
}

/// RAII guard for temporary files.
struct TempFileGuard {
    std::filesystem::path path;
    ~TempFileGuard() {
        std::filesystem::remove(path);
    }
};

}  // anonymous namespace

// ─── Property 11: Missing required field rejection ───────────────────────────
// **Validates: Requirements 2.8**

RC_GTEST_PROP(MissingFieldRejection, MissingRequiredFieldThrows, ()) {
    // Generate a random stream name
    auto stream_name = *gen_stream_name();

    // Generate a random non-empty subset of required fields to remove
    auto removed_indices = *gen_fields_to_remove();

    // Build YAML with the chosen fields missing
    auto yaml = build_yaml_with_missing_fields(stream_name, removed_indices);

    auto path = write_temp_yaml(yaml);
    TempFileGuard guard{path};

    // parse_pipeline must throw std::invalid_argument
    try {
        (void)dagr::parse_pipeline(path);
        // Build description of what was removed for failure message
        std::string removed_desc;
        for (auto idx : removed_indices) {
            if (!removed_desc.empty()) removed_desc += ", ";
            removed_desc += REQUIRED_FIELDS[idx];
        }
        RC_FAIL("parse_pipeline did not throw for stream '" + stream_name + "' with missing fields: " + removed_desc);
    } catch (const std::invalid_argument &e) {
        std::string msg = e.what();

        // Error message must contain the stream name
        RC_ASSERT(msg.find(stream_name) != std::string::npos);

        // The parser checks fields in order (temporal_profile, out_of_bounds_policy,
        // dataset_path, snapshot_interval). It throws on the FIRST missing field.
        // Identify which field is the first missing one in check order.
        std::size_t first_removed = *std::min_element(removed_indices.begin(), removed_indices.end());
        const std::string &expected_field = REQUIRED_FIELDS[first_removed];

        // Error message must report the name of the (first) missing field
        RC_ASSERT(msg.find(expected_field) != std::string::npos);
    }
}
