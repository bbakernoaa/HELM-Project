// DAGR — test_yaml_parser.cpp
// Unit tests for parse_pipeline edge cases.
// Requirements: 2.1, 2.2, 2.8, 2.10, 8.9

#include <gtest/gtest.h>

#include "dagr/pipeline_config.hpp"
#include "conf/conf.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <stdexcept>

namespace {

/// Helper: create a temporary YAML file with the given content and return its path.
/// Uses std::filesystem::temp_directory_path() + a unique name based on test info.
std::filesystem::path write_temp_yaml(const std::string& content,
                                       const std::string& suffix = ".yaml") {
    static int counter = 0;
    auto path = std::filesystem::temp_directory_path()
                / ("dagr_test_" + std::to_string(++counter) + suffix);
    std::ofstream ofs(path);
    ofs << content;
    ofs.close();
    return path;
}

/// Cleanup helper — removes a temp file if it exists.
struct TempFileGuard {
    std::filesystem::path path;
    ~TempFileGuard() {
        std::filesystem::remove(path);
    }
};

} // anonymous namespace

// ─── Req 2.1, 2.2: Missing file → conf::Conf_Error propagation ──────────────

TEST(YamlParser, MissingFileThrowsConfError) {
    // A non-existent path should cause conf::Config::from_file to throw,
    // and parse_pipeline must let it propagate without catching or wrapping.
    const std::filesystem::path bogus_path = "/tmp/dagr_nonexistent_file_xyz_42.yaml";

    EXPECT_THROW(dagr::parse_pipeline(bogus_path), conf::Conf_Error);
}

// ─── Req 2.1, 2.2: Malformed YAML → conf::Conf_Error propagation ────────────

TEST(YamlParser, MalformedYamlThrowsConfError) {
    // Write invalid YAML content — unbalanced braces / bad syntax
    const std::string bad_yaml = R"(
streams:
  - name: "test_stream"
    temporal_profile: "linear"
    [[[invalid yaml content here
    this: is: not: valid: yaml:::{}}
)";
    auto path = write_temp_yaml(bad_yaml);
    TempFileGuard guard{path};

    EXPECT_THROW(dagr::parse_pipeline(path), conf::Conf_Error);
}

// ─── Req 8.9: Empty dataset_path throws with stream name ─────────────────────

TEST(YamlParser, EmptyDatasetPathThrowsWithStreamName) {
    const std::string yaml = R"(
streams:
  - name: "sst_forcing"
    temporal_profile: "linear"
    out_of_bounds_policy: "clamp"
    dataset_path: ""
    snapshot_interval: "3600s"
tasks: []
dependencies: []
)";
    auto path = write_temp_yaml(yaml);
    TempFileGuard guard{path};

    try {
        auto cfg = dagr::parse_pipeline(path);
        (void)cfg;
        FAIL() << "Expected std::invalid_argument to be thrown";
    } catch (const std::invalid_argument& e) {
        // Error message should contain the stream name
        std::string msg = e.what();
        EXPECT_NE(msg.find("sst_forcing"), std::string::npos)
            << "Error message should contain the stream name. Got: " << msg;
        // Should also mention dataset_path
        EXPECT_NE(msg.find("dataset_path"), std::string::npos)
            << "Error message should mention 'dataset_path'. Got: " << msg;
    }
}

// ─── Req 8.9: Zero snapshot_interval throws with stream name ─────────────────

TEST(YamlParser, ZeroSnapshotIntervalThrowsWithStreamName) {
    const std::string yaml = R"(
streams:
  - name: "wind_data"
    temporal_profile: "step"
    out_of_bounds_policy: "cycle"
    dataset_path: "/data/wind.zarr"
    snapshot_interval: "0s"
tasks: []
dependencies: []
)";
    auto path = write_temp_yaml(yaml);
    TempFileGuard guard{path};

    try {
        auto cfg = dagr::parse_pipeline(path);
        (void)cfg;
        FAIL() << "Expected std::invalid_argument to be thrown";
    } catch (const std::invalid_argument& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("wind_data"), std::string::npos)
            << "Error message should contain the stream name. Got: " << msg;
        EXPECT_NE(msg.find("snapshot_interval"), std::string::npos)
            << "Error message should mention 'snapshot_interval'. Got: " << msg;
    }
}

// ─── Req 8.9: Negative snapshot_interval throws with stream name ─────────────

TEST(YamlParser, NegativeSnapshotIntervalThrowsWithStreamName) {
    const std::string yaml = R"(
streams:
  - name: "precip_stream"
    temporal_profile: "linear"
    out_of_bounds_policy: "clamp"
    dataset_path: "/data/precip.zarr"
    snapshot_interval: "-100s"
tasks: []
dependencies: []
)";
    auto path = write_temp_yaml(yaml);
    TempFileGuard guard{path};

    try {
        auto cfg = dagr::parse_pipeline(path);
        (void)cfg;
        FAIL() << "Expected std::invalid_argument to be thrown";
    } catch (const std::invalid_argument& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("precip_stream"), std::string::npos)
            << "Error message should contain the stream name. Got: " << msg;
        EXPECT_NE(msg.find("snapshot_interval"), std::string::npos)
            << "Error message should mention 'snapshot_interval'. Got: " << msg;
    }
}

// ─── Req 2.10: Valid minimal config parses successfully ──────────────────────

TEST(YamlParser, ValidMinimalConfigParsesSuccessfully) {
    const std::string yaml = R"(
streams:
  - name: "temperature"
    temporal_profile: "linear"
    out_of_bounds_policy: "clamp"
    dataset_path: "/data/temp.zarr"
    snapshot_interval: "86400s"
tasks:
  - name: "fetch_temp"
  - name: "regrid_temp"
dependencies:
  - from: "fetch_temp"
    to: "regrid_temp"
)";
    auto path = write_temp_yaml(yaml);
    TempFileGuard guard{path};

    dagr::Pipeline_Config cfg = dagr::parse_pipeline(path);

    // Verify stream was parsed
    ASSERT_EQ(cfg.streams.size(), 1u);
    EXPECT_EQ(cfg.streams[0].name, "temperature");
    EXPECT_EQ(cfg.streams[0].temporal_profile, dagr::Temporal_Profile::linear);
    EXPECT_EQ(cfg.streams[0].oob_policy, dagr::OutOfBounds_Policy::clamp);
    EXPECT_EQ(cfg.streams[0].dataset_path, std::filesystem::path{"/data/temp.zarr"});
    // 86400 seconds in nanoseconds
    EXPECT_EQ(cfg.streams[0].snapshot_interval, tick::seconds(86400));

    // Verify tasks
    ASSERT_EQ(cfg.task_names.size(), 2u);
    EXPECT_EQ(cfg.task_names[0], "fetch_temp");
    EXPECT_EQ(cfg.task_names[1], "regrid_temp");

    // Verify edges
    ASSERT_EQ(cfg.edges.size(), 1u);
    EXPECT_EQ(cfg.edges[0].producer_id, 0u);
    EXPECT_EQ(cfg.edges[0].consumer_id, 1u);
}

// ─── Req 2.9: Duplicate stream names throw ───────────────────────────────────

TEST(YamlParser, DuplicateStreamNamesThrow) {
    const std::string yaml = R"(
streams:
  - name: "sst_forcing"
    temporal_profile: "linear"
    out_of_bounds_policy: "clamp"
    dataset_path: "/data/sst.zarr"
    snapshot_interval: "3600s"
  - name: "sst_forcing"
    temporal_profile: "step"
    out_of_bounds_policy: "cycle"
    dataset_path: "/data/sst2.zarr"
    snapshot_interval: "7200s"
tasks: []
dependencies: []
)";
    auto path = write_temp_yaml(yaml);
    TempFileGuard guard{path};

    try {
        auto cfg = dagr::parse_pipeline(path);
        (void)cfg;
        FAIL() << "Expected std::invalid_argument for duplicate stream names";
    } catch (const std::invalid_argument& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("sst_forcing"), std::string::npos)
            << "Error message should contain the duplicate stream name. Got: " << msg;
    }
}

// ─── Req 2.6: Invalid temporal_profile throws ────────────────────────────────

TEST(YamlParser, InvalidTemporalProfileThrows) {
    const std::string yaml = R"(
streams:
  - name: "wind_stream"
    temporal_profile: "cubic_spline"
    out_of_bounds_policy: "clamp"
    dataset_path: "/data/wind.zarr"
    snapshot_interval: "3600s"
tasks: []
dependencies: []
)";
    auto path = write_temp_yaml(yaml);
    TempFileGuard guard{path};

    try {
        auto cfg = dagr::parse_pipeline(path);
        (void)cfg;
        FAIL() << "Expected std::invalid_argument for invalid temporal_profile";
    } catch (const std::invalid_argument& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("wind_stream"), std::string::npos)
            << "Error message should contain the stream name. Got: " << msg;
        EXPECT_NE(msg.find("cubic_spline"), std::string::npos)
            << "Error message should contain the invalid value. Got: " << msg;
    }
}

// ─── Req 2.7: Invalid out_of_bounds_policy throws ────────────────────────────

TEST(YamlParser, InvalidOutOfBoundsPolicyThrows) {
    const std::string yaml = R"(
streams:
  - name: "pressure_data"
    temporal_profile: "linear"
    out_of_bounds_policy: "extrapolate"
    dataset_path: "/data/pressure.zarr"
    snapshot_interval: "3600s"
tasks: []
dependencies: []
)";
    auto path = write_temp_yaml(yaml);
    TempFileGuard guard{path};

    try {
        auto cfg = dagr::parse_pipeline(path);
        (void)cfg;
        FAIL() << "Expected std::invalid_argument for invalid out_of_bounds_policy";
    } catch (const std::invalid_argument& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("pressure_data"), std::string::npos)
            << "Error message should contain the stream name. Got: " << msg;
        EXPECT_NE(msg.find("extrapolate"), std::string::npos)
            << "Error message should contain the invalid value. Got: " << msg;
    }
}

// ─── Req 2.8: Missing required fields throw ──────────────────────────────────

TEST(YamlParser, MissingTemporalProfileThrows) {
    const std::string yaml = R"(
streams:
  - name: "humidity"
    out_of_bounds_policy: "clamp"
    dataset_path: "/data/humidity.zarr"
    snapshot_interval: "3600s"
tasks: []
dependencies: []
)";
    auto path = write_temp_yaml(yaml);
    TempFileGuard guard{path};

    try {
        auto cfg = dagr::parse_pipeline(path);
        (void)cfg;
        FAIL() << "Expected std::invalid_argument for missing temporal_profile";
    } catch (const std::invalid_argument& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("humidity"), std::string::npos)
            << "Error message should contain stream name. Got: " << msg;
        EXPECT_NE(msg.find("temporal_profile"), std::string::npos)
            << "Error message should mention the missing field. Got: " << msg;
    }
}

TEST(YamlParser, MissingOutOfBoundsPolicyThrows) {
    const std::string yaml = R"(
streams:
  - name: "ozone"
    temporal_profile: "step"
    dataset_path: "/data/ozone.zarr"
    snapshot_interval: "86400s"
tasks: []
dependencies: []
)";
    auto path = write_temp_yaml(yaml);
    TempFileGuard guard{path};

    try {
        auto cfg = dagr::parse_pipeline(path);
        (void)cfg;
        FAIL() << "Expected std::invalid_argument for missing out_of_bounds_policy";
    } catch (const std::invalid_argument& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("ozone"), std::string::npos)
            << "Error message should contain stream name. Got: " << msg;
        EXPECT_NE(msg.find("out_of_bounds_policy"), std::string::npos)
            << "Error message should mention the missing field. Got: " << msg;
    }
}

TEST(YamlParser, MissingDatasetPathThrows) {
    const std::string yaml = R"(
streams:
  - name: "aerosol"
    temporal_profile: "linear"
    out_of_bounds_policy: "cycle"
    snapshot_interval: "3600s"
tasks: []
dependencies: []
)";
    auto path = write_temp_yaml(yaml);
    TempFileGuard guard{path};

    try {
        auto cfg = dagr::parse_pipeline(path);
        (void)cfg;
        FAIL() << "Expected std::invalid_argument for missing dataset_path";
    } catch (const std::invalid_argument& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("aerosol"), std::string::npos)
            << "Error message should contain stream name. Got: " << msg;
        EXPECT_NE(msg.find("dataset_path"), std::string::npos)
            << "Error message should mention the missing field. Got: " << msg;
    }
}

TEST(YamlParser, MissingSnapshotIntervalThrows) {
    const std::string yaml = R"(
streams:
  - name: "radiation"
    temporal_profile: "step"
    out_of_bounds_policy: "clamp"
    dataset_path: "/data/radiation.zarr"
tasks: []
dependencies: []
)";
    auto path = write_temp_yaml(yaml);
    TempFileGuard guard{path};

    try {
        auto cfg = dagr::parse_pipeline(path);
        (void)cfg;
        FAIL() << "Expected std::invalid_argument for missing snapshot_interval";
    } catch (const std::invalid_argument& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("radiation"), std::string::npos)
            << "Error message should contain stream name. Got: " << msg;
        EXPECT_NE(msg.find("snapshot_interval"), std::string::npos)
            << "Error message should mention the missing field. Got: " << msg;
    }
}

// ─── Req 2.11: Cycle detection throws ────────────────────────────────────────

TEST(YamlParser, CycleDetectionThrows) {
    const std::string yaml = R"(
streams: []
tasks:
  - name: "task_a"
  - name: "task_b"
  - name: "task_c"
dependencies:
  - from: "task_a"
    to: "task_b"
  - from: "task_b"
    to: "task_c"
  - from: "task_c"
    to: "task_a"
)";
    auto path = write_temp_yaml(yaml);
    TempFileGuard guard{path};

    try {
        auto cfg = dagr::parse_pipeline(path);
        (void)cfg;
        FAIL() << "Expected std::invalid_argument for cyclic dependencies";
    } catch (const std::invalid_argument& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("cycle"), std::string::npos)
            << "Error message should mention 'cycle'. Got: " << msg;
    }
}

// ─── Req 2.10: Stream ordering preservation ──────────────────────────────────

TEST(YamlParser, StreamOrderingPreserved) {
    const std::string yaml = R"(
streams:
  - name: "alpha_stream"
    temporal_profile: "linear"
    out_of_bounds_policy: "clamp"
    dataset_path: "/data/alpha.zarr"
    snapshot_interval: "3600s"
  - name: "beta_stream"
    temporal_profile: "step"
    out_of_bounds_policy: "cycle"
    dataset_path: "/data/beta.zarr"
    snapshot_interval: "7200s"
  - name: "gamma_stream"
    temporal_profile: "linear"
    out_of_bounds_policy: "cycle"
    dataset_path: "/data/gamma.zarr"
    snapshot_interval: "86400s"
tasks: []
dependencies: []
)";
    auto path = write_temp_yaml(yaml);
    TempFileGuard guard{path};

    dagr::Pipeline_Config cfg = dagr::parse_pipeline(path);

    // Streams must appear in declaration order (Req 2.10)
    ASSERT_EQ(cfg.streams.size(), 3u);
    EXPECT_EQ(cfg.streams[0].name, "alpha_stream");
    EXPECT_EQ(cfg.streams[1].name, "beta_stream");
    EXPECT_EQ(cfg.streams[2].name, "gamma_stream");

    // Verify individual stream properties are correctly assigned
    EXPECT_EQ(cfg.streams[0].temporal_profile, dagr::Temporal_Profile::linear);
    EXPECT_EQ(cfg.streams[0].oob_policy, dagr::OutOfBounds_Policy::clamp);
    EXPECT_EQ(cfg.streams[0].snapshot_interval, tick::seconds(3600));

    EXPECT_EQ(cfg.streams[1].temporal_profile, dagr::Temporal_Profile::step);
    EXPECT_EQ(cfg.streams[1].oob_policy, dagr::OutOfBounds_Policy::cycle);
    EXPECT_EQ(cfg.streams[1].snapshot_interval, tick::seconds(7200));

    EXPECT_EQ(cfg.streams[2].temporal_profile, dagr::Temporal_Profile::linear);
    EXPECT_EQ(cfg.streams[2].oob_policy, dagr::OutOfBounds_Policy::cycle);
    EXPECT_EQ(cfg.streams[2].snapshot_interval, tick::seconds(86400));
}
