// DAGR — prop_yaml_roundtrip.cpp
// Property 8: YAML Parsing Round-Trip (Stream_Descriptor)
//
// Validates: Requirements 2.3, 2.4, 2.5, 2.10, 8.4, 8.5, 8.6, 8.7, 8.8
//
// For any generated set of valid Stream_Descriptors (1–32 streams with unique
// names, valid temporal profiles, valid out-of-bounds policies, non-empty paths,
// and positive snapshot intervals), serializing them to YAML format and parsing
// with parse_pipeline SHALL produce a Pipeline_Config whose streams vector
// contains Stream_Descriptors that compare equal to the originals, in the same
// declaration order.

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <dagr/pipeline_config.hpp>
#include "generators.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

/// Generate a valid Stream_Descriptor whose snapshot_interval is an exact
/// multiple of whole seconds (since the YAML format only supports "Ns"
/// second-precision intervals). Uses unique names via index prefix.
auto roundtrip_stream_descriptor(int index) {
    return rc::gen::exec([index] {
        // Generate a stream name: 1–20 lowercase alphabetic characters
        auto name_len = *rc::gen::inRange(1, 21);
        std::string name;
        name.reserve(name_len + 10);
        for (int i = 0; i < name_len; ++i) {
            name.push_back(*rc::gen::inRange<char>('a', 'z' + 1));
        }
        // Prefix with index to guarantee uniqueness
        name = "s" + std::to_string(index) + "_" + name;

        // Random temporal profile
        auto profile = *rc::gen::inRange<int>(0, 2) == 0
                           ? dagr::Temporal_Profile::linear
                           : dagr::Temporal_Profile::step;

        // Random out-of-bounds policy
        auto oob = *rc::gen::inRange<int>(0, 2) == 0
                       ? dagr::OutOfBounds_Policy::clamp
                       : dagr::OutOfBounds_Policy::cycle;

        // Generate a non-empty dataset path (alphanumeric)
        auto path_len = *rc::gen::inRange(3, 30);
        std::string path_str = "/data/";
        for (int i = 0; i < path_len; ++i) {
            path_str.push_back(*rc::gen::inRange<char>('a', 'z' + 1));
        }
        path_str += ".zarr";

        // Generate a positive snapshot interval in whole seconds [1, 86400]
        // (parse_pipeline only supports "Ns" format with integer seconds)
        auto interval_seconds = *rc::gen::inRange<std::int64_t>(1, 86401);
        auto snapshot_interval = tick::seconds(interval_seconds);

        return dagr::Stream_Descriptor{
            std::move(name),
            profile,
            oob,
            std::filesystem::path{std::move(path_str)},
            snapshot_interval
        };
    });
}

/// Serialize a vector of Stream_Descriptors to YAML in the format expected
/// by parse_pipeline (via conf::Config::from_file).
std::string serialize_to_yaml(const std::vector<dagr::Stream_Descriptor>& streams) {
    std::string yaml;
    yaml += "settings:\n";
    yaml += "  max_concurrency: 32\n";
    yaml += "  deadlock_timeout: 60\n";
    yaml += "  shutdown_timeout: 30\n";

    yaml += "streams:\n";
    for (const auto& sd : streams) {
        yaml += "  - name: " + sd.name + "\n";

        // temporal_profile enum → string
        yaml += "    temporal_profile: ";
        yaml += (sd.temporal_profile == dagr::Temporal_Profile::linear) ? "linear" : "step";
        yaml += "\n";

        // out_of_bounds_policy enum → string
        yaml += "    out_of_bounds_policy: ";
        yaml += (sd.oob_policy == dagr::OutOfBounds_Policy::clamp) ? "clamp" : "cycle";
        yaml += "\n";

        // dataset_path
        yaml += "    dataset_path: " + sd.dataset_path.string() + "\n";

        // snapshot_interval: convert nanos → seconds, emit as "Ns"
        auto seconds = sd.snapshot_interval.nanos() / 1'000'000'000LL;
        yaml += "    snapshot_interval: " + std::to_string(seconds) + "s\n";
    }

    // Minimal tasks section (required by parse_pipeline)
    yaml += "tasks:\n";
    yaml += "  - name: task_0\n";

    // Empty dependencies
    yaml += "dependencies: []\n";

    return yaml;
}

/// RAII temp file helper: creates a unique temp file, writes content, and
/// removes the file on destruction.
class Temp_YAML_File {
public:
    explicit Temp_YAML_File(const std::string& content) {
        path_ = std::filesystem::temp_directory_path() /
                ("dagr_roundtrip_" + std::to_string(counter_++) + ".yaml");
        std::ofstream ofs(path_);
        ofs << content;
        ofs.close();
    }

    ~Temp_YAML_File() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

    Temp_YAML_File(const Temp_YAML_File&) = delete;
    Temp_YAML_File& operator=(const Temp_YAML_File&) = delete;

private:
    std::filesystem::path path_;
    static inline int counter_ = 0;
};

} // anonymous namespace

/// **Validates: Requirements 2.3, 2.4, 2.5, 2.10, 8.4, 8.5, 8.6, 8.7, 8.8**
RC_GTEST_PROP(YAMLRoundTrip, StreamDescriptorsPreserved, ()) {
    // Generate 1–32 stream descriptors with unique names
    auto stream_count = *rc::gen::inRange(1, 33);

    std::vector<dagr::Stream_Descriptor> original_streams;
    original_streams.reserve(stream_count);

    for (int i = 0; i < stream_count; ++i) {
        original_streams.push_back(*roundtrip_stream_descriptor(i));
    }

    // Serialize to YAML
    std::string yaml = serialize_to_yaml(original_streams);

    // Write to temp file
    Temp_YAML_File tmp(yaml);

    // Parse with parse_pipeline
    dagr::Pipeline_Config config = dagr::parse_pipeline(tmp.path());

    // Verify stream count matches
    RC_ASSERT(config.streams.size() == static_cast<std::size_t>(stream_count));

    // Verify each stream descriptor matches in order (Req 2.10: order preservation)
    for (std::size_t i = 0; i < config.streams.size(); ++i) {
        const auto& parsed = config.streams[i];
        const auto& expected = original_streams[i];

        // Req 2.3: stream name preserved
        RC_ASSERT(parsed.name == expected.name);

        // Req 8.4, 8.5: temporal_profile preserved
        RC_ASSERT(parsed.temporal_profile == expected.temporal_profile);

        // Req 8.6, 8.7: out_of_bounds_policy preserved
        RC_ASSERT(parsed.oob_policy == expected.oob_policy);

        // Req 2.5: dataset_path preserved
        RC_ASSERT(parsed.dataset_path == expected.dataset_path);

        // Req 2.5, 8.8: snapshot_interval preserved
        RC_ASSERT(parsed.snapshot_interval == expected.snapshot_interval);

        // Full equality check (Req 2.4)
        RC_ASSERT(parsed == expected);
    }
}
