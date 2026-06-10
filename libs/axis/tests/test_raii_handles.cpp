// AXIS unit test: Proj_Handle and File_Handle release verification
// Verifies that RAII handles properly manage their underlying C resources:
// File_Handle opens, writes, and closes on scope exit. Proj_Handle throws on
// invalid input.

#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

#include <axis/detail/raii_handles.hpp>

namespace axis::test {

// Test: File_Handle opens a temp file, writes content, dtor closes it,
// and the content is readable afterward.
TEST(RaiiHandles, FileHandleWritesAndCloses) {
    std::string tmp_path = std::string(std::tmpnam(nullptr)) + ".txt";
    const char* test_content = "Hello AXIS\n";

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
    EXPECT_THROW(
        axis::detail::File_Handle("/nonexistent/dir/file.txt", "r"),
        std::runtime_error);
}

// Test: File_Handle is move-only — moved-from handle is null
TEST(RaiiHandles, FileHandleMoveTransfersOwnership) {
    std::string tmp_path = std::string(std::tmpnam(nullptr)) + ".txt";

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
    EXPECT_THROW(
        axis::detail::Proj_Handle("not_a_valid_proj_string"),
        std::runtime_error);
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
    auto* raw = ph1.get();

    axis::detail::Proj_Handle ph2(std::move(ph1));
    EXPECT_FALSE(static_cast<bool>(ph1));
    EXPECT_TRUE(static_cast<bool>(ph2));
    EXPECT_EQ(ph2.get(), raw);
}
#endif  // AXIS_ENABLE_PROJ

}  // namespace axis::test
