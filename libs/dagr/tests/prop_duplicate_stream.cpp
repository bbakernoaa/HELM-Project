// DAGR — prop_duplicate_stream.cpp
// Property 12: Duplicate Stream Name Rejection
//
// For any pipeline configuration containing two or more stream entries with
// the same stream name, parse_pipeline SHALL throw std::invalid_argument
// reporting the duplicate stream name.
//
// Validates: Requirements 2.9

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include "dagr/pipeline_config.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

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

/// Generate a valid temporal_profile string.
rc::Gen<std::string> gen_temporal_profile() {
    return rc::gen::element(std::string{"linear"}, std::string{"step"});
}

/// Generate a valid out_of_bounds_policy string.
rc::Gen<std::string> gen_oob_policy() {
    return rc::gen::element(std::string{"clamp"}, std::string{"cycle"});
}

/// Generate a positive snapshot interval string (e.g. "3600s").
rc::Gen<std::string> gen_snapshot_interval() {
    return rc::gen::exec([] {
        auto seconds = *rc::gen::inRange<int>(1, 86401);
        return std::to_string(seconds) + "s";
    });
}

/// Generate a non-empty dataset path string.
rc::Gen<std::string> gen_dataset_path() {
    return rc::gen::exec([] {
        auto suffix_len = *rc::gen::inRange(1, 20);
        std::string path = "/data/";
        for (int i = 0; i < suffix_len; ++i) {
            path.push_back(*rc::gen::inRange<char>('a', 'z' + 1));
        }
        path += ".nc";
        return path;
    });
}

/// Serialize a single stream entry to YAML text.
std::string stream_to_yaml(const std::string& name,
                           const std::string& temporal_profile,
                           const std::string& oob_policy,
                           const std::string& dataset_path,
                           const std::string& snapshot_interval) {
    std::string yaml;
    yaml += "  - name: " + name + "\n";
    yaml += "    temporal_profile: " + temporal_profile + "\n";
    yaml += "    out_of_bounds_policy: " + oob_policy + "\n";
    yaml += "    dataset_path: " + dataset_path + "\n";
    yaml += "    snapshot_interval: " + snapshot_interval + "\n";
    return yaml;
}

/// Helper: create a temporary YAML file and return its path.
std::filesystem::path write_temp_yaml(const std::string& content) {
    static int counter = 0;
    auto path = std::filesystem::temp_directory_path()
                / ("dagr_prop_duplicate_stream_" + std::to_string(++counter) + ".yaml");
    std::ofstream ofs(path);
    ofs << content;
    ofs.close();
    return path;
}

/// RAII guard for temporary files.
struct TempFileGuard {
    std::filesystem::path path;
    ~TempFileGuard() { std::filesystem::remove(path); }
};

} // anonymous namespace

// ─── Property 12: Duplicate stream name is rejected ──────────────────────────
// **Validates: Requirements 2.9**

RC_GTEST_PROP(DuplicateStreamRejection,
              DuplicateStreamNameThrows,
              ()) {
    // Generate a number of unique base streams (2–8)
    auto stream_count = *rc::gen::inRange<std::uint32_t>(2, 9);

    // Generate unique stream names
    std::vector<std::string> names;
    names.reserve(stream_count);
    for (std::uint32_t i = 0; i < stream_count; ++i) {
        // Prefix with index to guarantee uniqueness across generated names
        auto suffix = *gen_stream_name();
        names.push_back("stream_" + std::to_string(i) + "_" + suffix);
    }

    // Pick a random stream index whose name we will duplicate
    auto dup_index = *rc::gen::inRange<std::uint32_t>(0, stream_count);
    std::string duplicate_name = names[dup_index];

    // Pick a different index to inject the duplicate at
    auto inject_index = *rc::gen::inRange<std::uint32_t>(0, stream_count);
    while (inject_index == dup_index) {
        inject_index = (inject_index + 1) % stream_count;
    }
    // Overwrite the name at inject_index with the duplicate
    names[inject_index] = duplicate_name;

    // Build the YAML content
    std::string yaml = "streams:\n";
    for (std::uint32_t i = 0; i < stream_count; ++i) {
        auto tp = *gen_temporal_profile();
        auto oob = *gen_oob_policy();
        auto ds = *gen_dataset_path();
        auto interval = *gen_snapshot_interval();
        yaml += stream_to_yaml(names[i], tp, oob, ds, interval);
    }

    // Add minimal tasks and dependencies sections
    yaml += "tasks:\n";
    yaml += "  - name: task_0\n";
    yaml += "dependencies: []\n";

    auto path = write_temp_yaml(yaml);
    TempFileGuard guard{path};

    // parse_pipeline must throw std::invalid_argument
    try {
        (void)dagr::parse_pipeline(path);
        RC_FAIL("parse_pipeline did not throw for duplicate stream name: '"
                + duplicate_name + "'");
    } catch (const std::invalid_argument& e) {
        std::string msg = e.what();

        // Error message must report the duplicate stream name
        RC_ASSERT(msg.find(duplicate_name) != std::string::npos);
    }
}

// ─── Property 12b: Two identical streams at arbitrary positions ──────────────
// Ensures the duplicate is detected regardless of positioning in the stream list.
// **Validates: Requirements 2.9**

RC_GTEST_PROP(DuplicateStreamRejection,
              TwoIdenticalStreamsAtAnyPosition,
              ()) {
    // Generate a shared name for the duplicate pair
    auto duplicate_name = *gen_stream_name();

    // Generate number of additional unique streams (0–6)
    auto extra_count = *rc::gen::inRange<std::uint32_t>(0, 7);

    // Total stream count = extra_count + 2 (the two duplicates)
    std::uint32_t total = extra_count + 2;

    // Build names vector with unique names for extras
    std::vector<std::string> names;
    names.reserve(total);
    for (std::uint32_t i = 0; i < extra_count; ++i) {
        names.push_back("unique_" + std::to_string(i) + "_" + *gen_stream_name());
    }

    // Insert two duplicate entries at random positions
    auto pos1 = *rc::gen::inRange<std::uint32_t>(0, static_cast<std::uint32_t>(names.size()) + 1);
    names.insert(names.begin() + pos1, duplicate_name);

    auto pos2 = *rc::gen::inRange<std::uint32_t>(0, static_cast<std::uint32_t>(names.size()) + 1);
    // Ensure pos2 != pos1 so they are distinct entries
    if (pos2 == pos1) {
        pos2 = (pos2 + 1) % (static_cast<std::uint32_t>(names.size()) + 1);
    }
    names.insert(names.begin() + pos2, duplicate_name);

    // Build the YAML content
    std::string yaml = "streams:\n";
    for (std::uint32_t i = 0; i < names.size(); ++i) {
        auto tp = *gen_temporal_profile();
        auto oob = *gen_oob_policy();
        auto ds = *gen_dataset_path();
        auto interval = *gen_snapshot_interval();
        yaml += stream_to_yaml(names[i], tp, oob, ds, interval);
    }

    // Add minimal tasks and dependencies sections
    yaml += "tasks:\n";
    yaml += "  - name: task_0\n";
    yaml += "dependencies: []\n";

    auto path = write_temp_yaml(yaml);
    TempFileGuard guard{path};

    // parse_pipeline must throw std::invalid_argument
    try {
        (void)dagr::parse_pipeline(path);
        RC_FAIL("parse_pipeline did not throw for duplicate stream name: '"
                + duplicate_name + "'");
    } catch (const std::invalid_argument& e) {
        std::string msg = e.what();

        // Error message must report the duplicate stream name
        RC_ASSERT(msg.find(duplicate_name) != std::string::npos);
    }
}
