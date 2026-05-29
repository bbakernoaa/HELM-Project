// ─── CONF Config Loading Unit Tests ──────────────────────────────────────────
// Exercises the two public factory constructors of conf::Config and the loading
// contract that backs every later query:
//   * from_string parses valid YAML and a dotted-path query returns the value.
//   * from_file parses a valid file written to a temp path and a query returns
//     the value.
//   * from_file on a nonexistent path raises Conf_Error{File_Not_Found}.
//   * A Config snapshots its tree at construction: mutating the originating
//     string afterward never changes a query result (each instance owns an
//     independent parsed tree).
//
// Feature: conf-config-parser
// Requirements: 32.1, 1.1, 1.2, 1.3, 1.4
// ─────────────────────────────────────────────────────────────────────────────

#include <conf/config.hpp>
#include <conf/error.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

using conf::Config;
using conf::Conf_Error;
using conf::Error_Code;

namespace {

// A small, representative model-config document with a nested map and a scalar
// integer leaf addressable as the dotted path "model.layers".
constexpr const char* kValidYaml =
    "model:\n"
    "  layers: 42\n"
    "  name: spherical\n"
    "grid:\n"
    "  resolution: 3.5\n";

// RAII helper: writes `contents` to a uniquely named file under the system temp
// directory on construction and removes it on destruction, so a failing
// assertion never leaks a stray fixture file.
class Temp_Yaml_File {
public:
    explicit Temp_Yaml_File(const std::string& contents) {
        // A process-unique, monotonically increasing suffix keeps concurrent
        // tests from colliding without relying on any POSIX-only API.
        static std::atomic<unsigned> counter{0};
        const auto unique =
            "conf_test_" + std::to_string(counter.fetch_add(1)) + "_" +
            std::to_string(reinterpret_cast<std::uintptr_t>(this)) + ".yaml";
        path_ = std::filesystem::temp_directory_path() / unique;

        std::ofstream out(path_, std::ios::binary | std::ios::trunc);
        out << contents;
        out.close();
    }

    Temp_Yaml_File(const Temp_Yaml_File&)            = delete;
    Temp_Yaml_File& operator=(const Temp_Yaml_File&) = delete;

    ~Temp_Yaml_File() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);  // best-effort cleanup; never throws
    }

    [[nodiscard]] std::string string() const { return path_.string(); }

private:
    std::filesystem::path path_;
};

} // namespace

// ─── from_string loads and answers a query (Req 1.2) ─────────────────────────

TEST(ConfigLoadFromString, ValidYamlLoadsAndQueryReturnsExpectedValue) {
    Config cfg = Config::from_string(kValidYaml);

    EXPECT_EQ(cfg.get_int("model.layers"), 42);
    EXPECT_EQ(cfg.get_string("model.name"), "spherical");
    EXPECT_DOUBLE_EQ(cfg.get_double("grid.resolution"), 3.5);
}

// ─── from_file loads from a real file and answers a query (Req 1.1) ──────────

TEST(ConfigLoadFromFile, ValidFileLoadsAndQueryReturnsExpectedValue) {
    Temp_Yaml_File fixture(kValidYaml);

    Config cfg = Config::from_file(fixture.string());

    EXPECT_EQ(cfg.get_int("model.layers"), 42);
    EXPECT_EQ(cfg.get_string("model.name"), "spherical");
}

// ─── from_file on a missing path raises File_Not_Found (Req 1.4) ─────────────

TEST(ConfigLoadFromFile, MissingPathRaisesFileNotFound) {
    // A path under the temp directory that is overwhelmingly unlikely to exist.
    const auto missing =
        (std::filesystem::temp_directory_path() /
         "conf_definitely_missing_3f1c9e7b.yaml")
            .string();
    ASSERT_FALSE(std::filesystem::exists(missing));

    try {
        Config cfg = Config::from_file(missing);
        FAIL() << "from_file on a nonexistent path must throw Conf_Error";
    } catch (const Conf_Error& e) {
        EXPECT_EQ(e.code(), Error_Code::File_Not_Found);
    } catch (...) {
        FAIL() << "from_file must throw conf::Conf_Error, not another exception";
    }
}

// ─── Query results are immune to later mutation of the source (Req 1.3) ──────
// The Config snapshots its parsed tree at construction time; the originating
// std::string is no longer referenced afterward, so mutating it must not change
// any query result.

TEST(ConfigLoadFromString, QueryResultsImmuneToSourceMutation) {
    std::string source = kValidYaml;

    Config cfg = Config::from_string(source);
    ASSERT_EQ(cfg.get_int("model.layers"), 42);

    // Mutate the original buffer the Config was built from.
    source = "model:\n  layers: 999\n";
    source.clear();
    source.shrink_to_fit();

    // The Config still answers from its own snapshot of the tree.
    EXPECT_EQ(cfg.get_int("model.layers"), 42);
    EXPECT_EQ(cfg.get_string("model.name"), "spherical");
}
