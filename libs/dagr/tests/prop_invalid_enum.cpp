// DAGR — prop_invalid_enum.cpp
// Property 10: Invalid Enum Rejection
//
// Validates: Requirements 2.6, 2.7
//
// For any stream entry whose temporal_profile field is a non-empty string other
// than "linear" or "step", OR whose out_of_bounds_policy field is a non-empty
// string other than "clamp" or "cycle", parse_pipeline SHALL throw
// std::invalid_argument reporting the stream name and the invalid value.

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <cstdio>
#include <dagr/pipeline_config.hpp>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {

/// Set of valid temporal_profile values.
const std::string VALID_TEMPORAL_PROFILES[] = {"linear", "step"};

/// Set of valid out_of_bounds_policy values.
const std::string VALID_OOB_POLICIES[] = {"clamp", "cycle"};

/// Check if a string is a valid temporal_profile value.
bool is_valid_temporal_profile(const std::string &s) {
    return s == "linear" || s == "step";
}

/// Check if a string is a valid out_of_bounds_policy value.
bool is_valid_oob_policy(const std::string &s) {
    return s == "clamp" || s == "cycle";
}

/// Generate a random non-empty YAML-safe string that is NOT a valid temporal_profile.
/// Uses only lowercase/uppercase alphanumeric characters to avoid YAML structural issues.
auto invalid_temporal_profile() {
    return rc::gen::exec([] {
        // Generate a string of 1–15 alphanumeric characters
        auto len = *rc::gen::inRange(1, 16);
        std::string result;
        result.reserve(len);
        for (int i = 0; i < len; ++i) {
            // Mix of lowercase, uppercase, and digits
            auto choice = *rc::gen::inRange(0, 3);
            if (choice == 0) {
                result.push_back(*rc::gen::inRange<char>('a', 'z' + 1));
            } else if (choice == 1) {
                result.push_back(*rc::gen::inRange<char>('A', 'Z' + 1));
            } else {
                result.push_back(*rc::gen::inRange<char>('0', '9' + 1));
            }
        }
        // Discard if it happens to be a valid value (extremely unlikely but possible)
        RC_PRE(!is_valid_temporal_profile(result));
        return result;
    });
}

/// Generate a random non-empty YAML-safe string that is NOT a valid out_of_bounds_policy.
/// Uses only lowercase/uppercase alphanumeric characters to avoid YAML structural issues.
auto invalid_oob_policy() {
    return rc::gen::exec([] {
        // Generate a string of 1–15 alphanumeric characters
        auto len = *rc::gen::inRange(1, 16);
        std::string result;
        result.reserve(len);
        for (int i = 0; i < len; ++i) {
            // Mix of lowercase, uppercase, and digits
            auto choice = *rc::gen::inRange(0, 3);
            if (choice == 0) {
                result.push_back(*rc::gen::inRange<char>('a', 'z' + 1));
            } else if (choice == 1) {
                result.push_back(*rc::gen::inRange<char>('A', 'Z' + 1));
            } else {
                result.push_back(*rc::gen::inRange<char>('0', '9' + 1));
            }
        }
        // Discard if it happens to be a valid value (extremely unlikely but possible)
        RC_PRE(!is_valid_oob_policy(result));
        return result;
    });
}

/// Generate a random stream name (1–30 lowercase alphanumeric chars).
auto stream_name_gen() {
    return rc::gen::exec([] {
        auto len = *rc::gen::inRange(1, 31);
        std::string name;
        name.reserve(len);
        for (int i = 0; i < len; ++i) {
            name.push_back(*rc::gen::inRange<char>('a', 'z' + 1));
        }
        return name;
    });
}

/// Serialize a minimal pipeline YAML with the given stream fields.
/// temporal_profile and out_of_bounds_policy are injected as-is.
std::string make_yaml(const std::string &stream_name, const std::string &temporal_profile, const std::string &oob_policy) {
    std::string yaml;
    yaml += "streams:\n";
    yaml += "  - name: " + stream_name + "\n";
    yaml += "    temporal_profile: " + temporal_profile + "\n";
    yaml += "    out_of_bounds_policy: " + oob_policy + "\n";
    yaml += "    dataset_path: /data/test.nc\n";
    yaml += "    snapshot_interval: 3600s\n";
    yaml += "tasks:\n";
    yaml += "  - name: task_0\n";
    yaml += "dependencies: []\n";
    return yaml;
}

/// RAII temp file helper.
class Temp_YAML_File {
   public:
    explicit Temp_YAML_File(const std::string &content) {
        path_ = std::filesystem::temp_directory_path() / ("dagr_invalid_enum_" + std::to_string(counter_++) + ".yaml");
        std::ofstream ofs(path_);
        ofs << content;
        ofs.close();
    }

    ~Temp_YAML_File() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }

    [[nodiscard]] const std::filesystem::path &path() const noexcept {
        return path_;
    }

    Temp_YAML_File(const Temp_YAML_File &) = delete;
    Temp_YAML_File &operator=(const Temp_YAML_File &) = delete;

   private:
    std::filesystem::path path_;
    static inline int counter_ = 0;
};

}  // anonymous namespace

/// **Validates: Requirements 2.6**
///
/// For any stream entry whose temporal_profile field is a non-empty string
/// other than "linear" or "step", parse_pipeline SHALL throw
/// std::invalid_argument reporting the stream name and the invalid value.
RC_GTEST_PROP(InvalidEnumRejection, InvalidTemporalProfile, ()) {
    auto name = *stream_name_gen();
    auto bad_tp = *invalid_temporal_profile();

    // Use a valid oob_policy so we isolate the temporal_profile rejection
    auto valid_oob = *rc::gen::element(std::string("clamp"), std::string("cycle"));

    std::string yaml = make_yaml(name, bad_tp, valid_oob);
    Temp_YAML_File tmp(yaml);

    // parse_pipeline must throw std::invalid_argument
    bool threw = false;
    std::string error_msg;
    try {
        (void)dagr::parse_pipeline(tmp.path());
    } catch (const std::invalid_argument &e) {
        threw = true;
        error_msg = e.what();
    }

    RC_ASSERT(threw);

    // Error message must contain the stream name
    RC_ASSERT(error_msg.find(name) != std::string::npos);

    // Error message must contain the invalid value
    RC_ASSERT(error_msg.find(bad_tp) != std::string::npos);
}

/// **Validates: Requirements 2.7**
///
/// For any stream entry whose out_of_bounds_policy field is a non-empty string
/// other than "clamp" or "cycle", parse_pipeline SHALL throw
/// std::invalid_argument reporting the stream name and the invalid value.
RC_GTEST_PROP(InvalidEnumRejection, InvalidOutOfBoundsPolicy, ()) {
    auto name = *stream_name_gen();
    auto bad_oob = *invalid_oob_policy();

    // Use a valid temporal_profile so we isolate the oob_policy rejection
    auto valid_tp = *rc::gen::element(std::string("linear"), std::string("step"));

    std::string yaml = make_yaml(name, valid_tp, bad_oob);
    Temp_YAML_File tmp(yaml);

    // parse_pipeline must throw std::invalid_argument
    bool threw = false;
    std::string error_msg;
    try {
        (void)dagr::parse_pipeline(tmp.path());
    } catch (const std::invalid_argument &e) {
        threw = true;
        error_msg = e.what();
    }

    RC_ASSERT(threw);

    // Error message must contain the stream name
    RC_ASSERT(error_msg.find(name) != std::string::npos);

    // Error message must contain the invalid value
    RC_ASSERT(error_msg.find(bad_oob) != std::string::npos);
}
