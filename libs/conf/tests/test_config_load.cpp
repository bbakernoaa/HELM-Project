// ─── CONF Config Loading Unit Tests ──────────────────────────────────────────
// Validates Requirements 1.1, 1.2, 1.3, 1.4:
//   - Config::from_file loads a valid YAML file and queries return expected values
//   - Config::from_string loads valid YAML text and queries return expected values
//   - Config::from_file on a missing path raises Conf_Error with File_Not_Found
//   - Query results are immune to later mutation of the originating source
// ──────────────────────────────────────────────────────────────────────────────

#include <conf/config.hpp>
#include <conf/error.hpp>

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <string>

namespace {

// Simple YAML content used across tests
constexpr const char* kSimpleYaml =
    "model:\n"
    "  layers: 42\n"
    "  name: spherical\n";

// Helper: write content to a temporary file and return its path.
// The caller is responsible for removing the file.
std::string write_temp_yaml(const std::string& content) {
    const std::string path = "/tmp/conf_test_load.yaml";
    std::ofstream ofs(path, std::ios::out | std::ios::trunc);
    ofs << content;
    ofs.close();
    return path;
}

} // namespace

// ─── Requirement 1.2: Config::from_string parses valid YAML and queries succeed

TEST(ConfigLoad, FromString_ValidYaml_LoadsSuccessfully) {
    const auto cfg = conf::Config::from_string(kSimpleYaml);

    // Verify known keys are queryable with expected values
    EXPECT_TRUE(cfg.has("model"));
    EXPECT_TRUE(cfg.has("model.layers"));
    EXPECT_TRUE(cfg.has("model.name"));

    EXPECT_EQ(cfg.get_int("model.layers"), 42);
    EXPECT_EQ(cfg.get_string("model.name"), "spherical");
}

// ─── Requirement 1.2: Config::from_string with deeply nested maps

TEST(ConfigLoad, FromString_NestedMaps_DeepPathQuerySucceeds) {
    const char* nested_yaml =
        "simulation:\n"
        "  physics:\n"
        "    gravity:\n"
        "      constant: 9810\n"
        "      units: mm_per_s2\n"
        "    layers:\n"
        "      - thickness: 100\n"
        "      - thickness: 200\n";

    const auto cfg = conf::Config::from_string(nested_yaml);

    // Deep path into nested maps
    EXPECT_TRUE(cfg.has("simulation.physics.gravity.constant"));
    EXPECT_EQ(cfg.get_int("simulation.physics.gravity.constant"), 9810);
    EXPECT_EQ(cfg.get_string("simulation.physics.gravity.units"), "mm_per_s2");

    // Deep path into a sequence element
    EXPECT_EQ(cfg.get_int("simulation.physics.layers.0.thickness"), 100);
    EXPECT_EQ(cfg.get_int("simulation.physics.layers.1.thickness"), 200);
}

// ─── Requirement 1.1: Config::from_file loads a valid YAML file

TEST(ConfigLoad, FromFile_ValidFile_LoadsSuccessfully) {
    const std::string path = write_temp_yaml(kSimpleYaml);

    const auto cfg = conf::Config::from_file(path);

    EXPECT_TRUE(cfg.has("model.layers"));
    EXPECT_EQ(cfg.get_int("model.layers"), 42);
    EXPECT_EQ(cfg.get_string("model.name"), "spherical");

    // Clean up
    std::remove(path.c_str());
}

// ─── Requirement 1.4: from_file on a missing path raises Conf_Error{File_Not_Found}

TEST(ConfigLoad, FromFile_MissingPath_ThrowsFileNotFound) {
    const std::string missing = "/tmp/conf_test_nonexistent_98765.yaml";

    try {
        auto cfg = conf::Config::from_file(missing);
        FAIL() << "Expected Conf_Error to be thrown for missing file";
    } catch (const conf::Conf_Error& e) {
        EXPECT_EQ(e.code(), conf::Error_Code::File_Not_Found);
    } catch (...) {
        FAIL() << "Expected conf::Conf_Error, got a different exception type";
    }
}

// ─── Requirement 1.3: Query results are immune to later mutation of the source

TEST(ConfigLoad, QueryResults_ImmuneToSourceMutation) {
    // Load from a mutable string
    std::string yaml_source =
        "model:\n"
        "  layers: 42\n"
        "  name: spherical\n";

    const auto cfg = conf::Config::from_string(yaml_source);

    // Verify initial values
    EXPECT_EQ(cfg.get_int("model.layers"), 42);
    EXPECT_EQ(cfg.get_string("model.name"), "spherical");

    // Mutate the original source string completely
    yaml_source = "model:\n  layers: 99\n  name: cuboid\n";

    // Queries must still return the values captured at construction time
    EXPECT_EQ(cfg.get_int("model.layers"), 42);
    EXPECT_EQ(cfg.get_string("model.name"), "spherical");
}
