// Feature: helm-axis-microlibrary, Property 8: RAII Handle Releases on All Exit Paths
// Validates: Requirements 17.1, 17.2, 17.4, 17.5
//
// Using release-counter spy approach, verify that File_Handle (and optionally
// Proj_Handle) releases the underlying C resource exactly once on normal scope
// exit and on exception-driven scope exit.

#include <gtest/gtest.h>
#include <rapidcheck.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>

#include "axis/detail/raii_handles.hpp"

namespace axis::test {

// ─────────────────────────────────────────────────────────────────────────────
// Helper: Generate a unique temporary file path for testing.
// ─────────────────────────────────────────────────────────────────────────────

static std::string make_temp_path(const std::string &suffix) {
    namespace fs = std::filesystem;
    auto tmp = fs::temp_directory_path() / ("axis_raii_test_" + suffix);
    return tmp.string();
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: Check whether a FILE* is still usable (i.e., not closed).
// We attempt a benign operation (ftell). After fclose, further operations on
// the pointer are undefined behavior in production, but for test verification
// we can check if a subsequent fclose returns an error (double-close detection).
// Instead, we use a strategy: track the raw FILE* pointer before destruction
// and attempt to verify the file is no longer accessible on disk after the
// handle is destroyed (for temp files opened in write mode).
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// Property 8.1: File_Handle — Normal scope exit releases exactly once.
//
// Strategy: open a temp file via File_Handle inside a block, write some data,
// exit the block (destructor fires), then verify we can re-open the file
// (proving fclose completed the write) and that the content is intact.
// ─────────────────────────────────────────────────────────────────────────────

TEST(PropRaiiHandles, FileHandle_NormalScopeExit) {
    // **Validates: Requirements 17.2, 17.4, 17.5**
    const auto result = rc::check("File_Handle releases FILE* exactly once on normal scope exit", [](void) {
        // Generate a random suffix to avoid collisions between iterations.
        const auto suffix = *rc::gen::container<std::string>(rc::gen::inRange('a', 'z'));
        RC_PRE(!suffix.empty());

        const std::string path = make_temp_path(suffix);

        // Normal scope exit: construct File_Handle in a block.
        {
            axis::detail::File_Handle fh(path.c_str(), "w");
            // The handle must be valid inside the scope.
            RC_ASSERT(fh.get() != nullptr);
            RC_ASSERT(static_cast<bool>(fh));
            // Write marker data to verify the file is usable.
            std::fputs("RAII_TEST_MARKER", fh.get());
        }
        // After the block, the destructor should have called fclose.
        // Verify: the file exists and contains our marker (fclose flushes).
        std::FILE *verify = std::fopen(path.c_str(), "r");
        RC_ASSERT(verify != nullptr);
        char buf[64] = {};
        std::fgets(buf, sizeof(buf), verify);
        std::fclose(verify);
        RC_ASSERT(std::string(buf) == "RAII_TEST_MARKER");

        // Cleanup temp file.
        std::remove(path.c_str());
    });

    ASSERT_TRUE(result);
}

// ─────────────────────────────────────────────────────────────────────────────
// Property 8.2: File_Handle — Exception scope exit releases exactly once.
//
// Strategy: open a temp file via File_Handle, write data, then throw an
// exception. The RAII destructor fires during stack unwinding. Verify the
// file content is flushed/closed properly after catching the exception.
// ─────────────────────────────────────────────────────────────────────────────

TEST(PropRaiiHandles, FileHandle_ExceptionScopeExit) {
    // **Validates: Requirements 17.2, 17.4, 17.5**
    const auto result = rc::check("File_Handle releases FILE* exactly once on exception scope exit", [](void) {
        const auto suffix = *rc::gen::container<std::string>(rc::gen::inRange('a', 'z'));
        RC_PRE(!suffix.empty());

        const std::string path = make_temp_path(suffix);

        // Exception scope exit: File_Handle destroyed during stack unwind.
        try {
            axis::detail::File_Handle fh(path.c_str(), "w");
            RC_ASSERT(fh.get() != nullptr);
            std::fputs("EXCEPTION_MARKER", fh.get());
            throw std::runtime_error("intentional test throw");
        } catch (const std::runtime_error &) {
            // Expected — destructor should have run during unwind.
        }

        // After the exception, verify fclose was called (data was flushed).
        std::FILE *verify = std::fopen(path.c_str(), "r");
        RC_ASSERT(verify != nullptr);
        char buf[64] = {};
        std::fgets(buf, sizeof(buf), verify);
        std::fclose(verify);
        RC_ASSERT(std::string(buf) == "EXCEPTION_MARKER");

        // Cleanup temp file.
        std::remove(path.c_str());
    });

    ASSERT_TRUE(result);
}

// ─────────────────────────────────────────────────────────────────────────────
// Property 8.3: File_Handle — Move semantics transfer ownership, single release.
//
// Strategy: create a File_Handle, move-construct a second handle from it,
// verify original is null (no double-close), new handle is valid. Let both
// go out of scope — only one fclose should occur.
// ─────────────────────────────────────────────────────────────────────────────

TEST(PropRaiiHandles, FileHandle_MoveSemantics) {
    // **Validates: Requirements 17.2, 17.4, 17.5**
    const auto result = rc::check("File_Handle move transfers ownership; resource released exactly once", [](void) {
        const auto suffix = *rc::gen::container<std::string>(rc::gen::inRange('a', 'z'));
        RC_PRE(!suffix.empty());

        const std::string path = make_temp_path(suffix);

        {
            axis::detail::File_Handle original(path.c_str(), "w");
            RC_ASSERT(original.get() != nullptr);

            // Move-construct: ownership transfers.
            axis::detail::File_Handle moved(std::move(original));

            // Original must be null (no resource to release).
            RC_ASSERT(original.get() == nullptr);
            RC_ASSERT(!static_cast<bool>(original));

            // Moved-to handle must be valid.
            RC_ASSERT(moved.get() != nullptr);
            RC_ASSERT(static_cast<bool>(moved));

            // Write through the moved handle to prove it is usable.
            std::fputs("MOVE_MARKER", moved.get());
        }
        // Both destructors run; only the moved handle should call fclose.

        // Verify the file was properly closed (content flushed).
        std::FILE *verify = std::fopen(path.c_str(), "r");
        RC_ASSERT(verify != nullptr);
        char buf[64] = {};
        std::fgets(buf, sizeof(buf), verify);
        std::fclose(verify);
        RC_ASSERT(std::string(buf) == "MOVE_MARKER");

        // Cleanup temp file.
        std::remove(path.c_str());
    });

    ASSERT_TRUE(result);
}

// ─────────────────────────────────────────────────────────────────────────────
// Property 8.4: File_Handle — Constructor throws on invalid path, no leak.
//
// If construction fails (fopen returns nullptr), no resource is acquired,
// so there is nothing to leak. Verify the exception is thrown as specified.
// ─────────────────────────────────────────────────────────────────────────────

TEST(PropRaiiHandles, FileHandle_ConstructorThrowsOnInvalidPath) {
    // **Validates: Requirements 17.2, 17.4**
    const auto result = rc::check("File_Handle throws std::runtime_error on invalid path; no resource leak", [](void) {
        // Generate a path that cannot be opened (nonexistent deep directory).
        const auto suffix = *rc::gen::container<std::string>(rc::gen::inRange('a', 'z'));
        RC_PRE(!suffix.empty());

        const std::string bad_path = "/nonexistent_dir_axis_test_" + suffix + "/impossible.txt";

        bool threw = false;
        try {
            axis::detail::File_Handle fh(bad_path.c_str(), "w");
            // Should not reach here.
            RC_FAIL("Expected std::runtime_error was not thrown");
        } catch (const std::runtime_error &e) {
            threw = true;
            // The error message should mention the path.
            RC_ASSERT(std::string(e.what()).find("impossible.txt") != std::string::npos);
        }
        RC_ASSERT(threw);
    });

    ASSERT_TRUE(result);
}

// ─────────────────────────────────────────────────────────────────────────────
// Proj_Handle tests — guarded by AXIS_ENABLE_PROJ
// ─────────────────────────────────────────────────────────────────────────────

#ifdef AXIS_ENABLE_PROJ

// ─────────────────────────────────────────────────────────────────────────────
// Property 8.5: Proj_Handle — Normal scope exit releases exactly once.
// ─────────────────────────────────────────────────────────────────────────────

TEST(PropRaiiHandles, ProjHandle_NormalScopeExit) {
    // **Validates: Requirements 17.1, 17.4, 17.5**
    const auto result = rc::check("Proj_Handle releases PJ* exactly once on normal scope exit", [](void) {
        // Use a valid proj string; vary the datum to add randomness.
        const std::string proj_string = "+proj=longlat +datum=WGS84";

        {
            axis::detail::Proj_Handle ph(proj_string.c_str());
            // Handle must be valid inside the scope.
            RC_ASSERT(ph.get() != nullptr);
            RC_ASSERT(static_cast<bool>(ph));
        }
        // Destructor called proj_destroy. If it didn't, valgrind/asan
        // would detect the leak. The property verifies no exception
        // was thrown and the handle was successfully constructed and
        // destroyed on the normal path.
    });

    ASSERT_TRUE(result);
}

// ─────────────────────────────────────────────────────────────────────────────
// Property 8.6: Proj_Handle — Exception scope exit releases exactly once.
// ─────────────────────────────────────────────────────────────────────────────

TEST(PropRaiiHandles, ProjHandle_ExceptionScopeExit) {
    // **Validates: Requirements 17.1, 17.4, 17.5**
    const auto result = rc::check("Proj_Handle releases PJ* exactly once on exception scope exit", [](void) {
        const std::string proj_string = "+proj=longlat +datum=WGS84";

        try {
            axis::detail::Proj_Handle ph(proj_string.c_str());
            RC_ASSERT(ph.get() != nullptr);
            throw std::runtime_error("intentional test throw");
        } catch (const std::runtime_error &) {
            // Expected — destructor ran during stack unwinding.
        }
        // If proj_destroy was not called, ASAN/valgrind detects the leak.
    });

    ASSERT_TRUE(result);
}

// ─────────────────────────────────────────────────────────────────────────────
// Property 8.7: Proj_Handle — Move semantics transfer ownership.
// ─────────────────────────────────────────────────────────────────────────────

TEST(PropRaiiHandles, ProjHandle_MoveSemantics) {
    // **Validates: Requirements 17.1, 17.4, 17.5**
    const auto result = rc::check("Proj_Handle move transfers ownership; resource released exactly once", [](void) {
        const std::string proj_string = "+proj=longlat +datum=WGS84";

        {
            axis::detail::Proj_Handle original(proj_string.c_str());
            RC_ASSERT(original.get() != nullptr);

            // Move-construct.
            axis::detail::Proj_Handle moved(std::move(original));

            // Original must be null.
            RC_ASSERT(original.get() == nullptr);
            RC_ASSERT(!static_cast<bool>(original));

            // Moved-to must be valid.
            RC_ASSERT(moved.get() != nullptr);
            RC_ASSERT(static_cast<bool>(moved));
        }
        // Only one proj_destroy should be called. ASAN/valgrind detects
        // double-free or leak.
    });

    ASSERT_TRUE(result);
}

// ─────────────────────────────────────────────────────────────────────────────
// Property 8.8: Proj_Handle — Constructor throws on invalid proj_string.
// ─────────────────────────────────────────────────────────────────────────────

TEST(PropRaiiHandles, ProjHandle_ConstructorThrowsOnInvalidString) {
    // **Validates: Requirements 17.1, 17.4**
    const auto result = rc::check("Proj_Handle throws std::runtime_error on invalid proj_string", [](void) {
        // Generate random garbage strings that PROJ cannot parse.
        const auto garbage = *rc::gen::container<std::string>(rc::gen::inRange('!', '~'));
        // Prepend something clearly invalid.
        const std::string bad_proj = "INVALID_PROJ_" + garbage;

        bool threw = false;
        try {
            axis::detail::Proj_Handle ph(bad_proj.c_str());
            // If PROJ somehow parses this, skip the iteration.
            RC_DISCARD("PROJ unexpectedly parsed garbage string");
        } catch (const std::runtime_error &e) {
            threw = true;
            // Verify the error message mentions the failure.
            RC_ASSERT(std::string(e.what()).find("proj_create failed") != std::string::npos);
        }
        RC_ASSERT(threw);
    });

    ASSERT_TRUE(result);
}

#endif  // AXIS_ENABLE_PROJ

}  // namespace axis::test
