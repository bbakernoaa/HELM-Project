// AXIS unit test: Proj_Handle and File_Handle release verification
// Verifies that RAII handles properly manage their underlying C resources:
// File_Handle opens, writes, and closes on scope exit. Proj_Handle throws on
// invalid input.

#include <gtest/gtest.h>

#include <axis/detail/raii_handles.hpp>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>

namespace axis::test {

static std::string make_temp_txt_path() {
    namespace fs = std::filesystem;
    static thread_local std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<unsigned long long> dist;
    const fs::path dir = fs::temp_directory_path();

    for (int attempt = 0; attempt < 128; ++attempt) {
        const fs::path candidate = dir / ("axis_raii_" + std::to_string(dist(rng)) + ".txt");
        if (!fs::exists(candidate)) {
            return candidate.string();
        }
    }

    throw std::runtime_error("Failed to generate unique temp .txt path");
}

// Test: File_Handle opens a temp file, writes content, dtor closes it,
// and the content is readable afterward.
TEST(RaiiHandles, FileHandleWritesAndCloses) {
    std::string tmp_path = make_temp_txt_path();
    const char *test_content = "Hello AXIS\n";

    {
        axis::detail::File_Handle fh(tmp_path.c_str(), "w");
        ASSERT_TRUE(static_cast<bool>(fh));
        std::fputs(test_content, fh.get());
    }  // dtor closes the file

    // Verify content is accessible (file was properly closed)
    std::ifstream in(tmp_path);
    ASSERT_TRUE(in.is_open());
    std::string line;
    std::getline(in, line);
    EXPECT_EQ(line, "Hello AXIS");
    in.close();

    std::remove(tmp_path.c_str());
}

// Test: File_Handle with an invalid path throws std::runtime_error
TEST(RaiiHandles, FileHandleInvalidPathThrows) {
    EXPECT_THROW(axis::detail::File_Handle("/nonexistent/dir/file.txt", "r"), std::runtime_error);
}

// Test: File_Handle is move-only — moved-from handle is null
TEST(RaiiHandles, FileHandleMoveTransfersOwnership) {
    std::string tmp_path = make_temp_txt_path();

    axis::detail::File_Handle fh1(tmp_path.c_str(), "w");
    ASSERT_TRUE(static_cast<bool>(fh1));

    axis::detail::File_Handle fh2(std::move(fh1));
    EXPECT_FALSE(static_cast<bool>(fh1));  // moved-from is null
    EXPECT_TRUE(static_cast<bool>(fh2));   // moved-to is valid

    // fh2 destructor will close the file
    std::remove(tmp_path.c_str());
}

#ifdef AXIS_ENABLE_PROJ
// Test: Proj_Handle with invalid proj_string throws std::runtime_error
TEST(RaiiHandles, ProjHandleInvalidStringThrows) {
    EXPECT_THROW(axis::detail::Proj_Handle("not_a_valid_proj_string"), std::runtime_error);
}

// Test: Proj_Handle with valid longlat string succeeds
TEST(RaiiHandles, ProjHandleValidStringSucceeds) {
    axis::detail::Proj_Handle ph("+proj=longlat +datum=WGS84");
    EXPECT_TRUE(static_cast<bool>(ph));
    EXPECT_NE(ph.get(), nullptr);
}

// Test: Proj_Handle move transfers ownership
TEST(RaiiHandles, ProjHandleMoveTransfersOwnership) {
    axis::detail::Proj_Handle ph1("+proj=longlat +datum=WGS84");
    auto *raw = ph1.get();

    axis::detail::Proj_Handle ph2(std::move(ph1));
    EXPECT_FALSE(static_cast<bool>(ph1));
    EXPECT_TRUE(static_cast<bool>(ph2));
    EXPECT_EQ(ph2.get(), raw);
}
#endif  // AXIS_ENABLE_PROJ

}  // namespace axis::test
