// ─── HALO Pack/Unpack Kernel Unit Tests ─────────────────────────────────────
// Verifies the device-side pack/unpack kernels correctly stage data between
// multi-dimensional Kokkos views and contiguous 1D buffers. Tests cover:
//   - Contiguous 1D view pack/unpack (RangePolicy path)
//   - Strided 2D subview pack/unpack (MDRangePolicy path)
//   - LayoutLeft and LayoutRight rank-2 views
//   - Execution space tag dispatch (host backends: Serial/OpenMP)
//
// Feature: halo-production-hardening
// Requirements: 2.1, 2.2, 2.3, 2.5
// ─────────────────────────────────────────────────────────────────────────────

#include <halo/detail/pack_unpack.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

// ─── Test Fixture ────────────────────────────────────────────────────────────

class PackUnpackTest : public ::testing::Test {
protected:
    // Kokkos is initialized/finalized in main() via ScopeGuard.
};

// ─── 1. Pack from a contiguous 1D view into a buffer ────────────────────────

TEST_F(PackUnpackTest, PackContiguous1DView) {
    constexpr int N = 10;

    // Source: contiguous 1D view filled with known pattern
    Kokkos::View<double*> src("src", N);
    Kokkos::parallel_for("fill_src", N, KOKKOS_LAMBDA(int i) {
        src(i) = static_cast<double>(i * 3 + 7);
    });
    Kokkos::fence();

    // Destination buffer
    Kokkos::View<double*> buffer("buffer", N);

    // Pack
    halo::detail::pack(src, buffer);

    // Verify on host
    auto buffer_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, buffer);
    for (int i = 0; i < N; ++i) {
        EXPECT_DOUBLE_EQ(buffer_h(i), static_cast<double>(i * 3 + 7))
            << "Mismatch at index " << i;
    }
}

// ─── 2. Unpack from a buffer into a contiguous 1D view ──────────────────────

TEST_F(PackUnpackTest, UnpackToContiguous1DView) {
    constexpr int N = 8;

    // Source buffer filled with known values
    Kokkos::View<double*> buffer("buffer", N);
    Kokkos::parallel_for("fill_buffer", N, KOKKOS_LAMBDA(int i) {
        buffer(i) = static_cast<double>(i * 5 + 1);
    });
    Kokkos::fence();

    // Destination: contiguous 1D view, initially zero
    Kokkos::View<double*> dst("dst", N);

    // Unpack
    halo::detail::unpack(buffer, dst);

    // Verify on host
    auto dst_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dst);
    for (int i = 0; i < N; ++i) {
        EXPECT_DOUBLE_EQ(dst_h(i), static_cast<double>(i * 5 + 1))
            << "Mismatch at index " << i;
    }
}

// ─── 3. Pack/Unpack roundtrip with a contiguous 1D view ────────────────────

TEST_F(PackUnpackTest, PackUnpackRoundtripContiguous1D) {
    constexpr int N = 16;

    // Source view
    Kokkos::View<double*> src("src", N);
    Kokkos::parallel_for("fill_src", N, KOKKOS_LAMBDA(int i) {
        src(i) = static_cast<double>(i * i);
    });
    Kokkos::fence();

    // Pack into buffer
    Kokkos::View<double*> buffer("buffer", N);
    halo::detail::pack(src, buffer);

    // Unpack into new destination
    Kokkos::View<double*> dst("dst", N);
    halo::detail::unpack(buffer, dst);

    // Verify roundtrip preserves data
    auto src_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, src);
    auto dst_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dst);
    for (int i = 0; i < N; ++i) {
        EXPECT_DOUBLE_EQ(dst_h(i), src_h(i))
            << "Roundtrip mismatch at index " << i;
    }
}

// ─── 4. Pack from a strided 2D subview (non-contiguous) ─────────────────────

TEST_F(PackUnpackTest, PackStrided2DSubview) {
    // Create a 4x6 LayoutRight (row-major) 2D view
    constexpr int nrows = 4;
    constexpr int ncols = 6;
    Kokkos::View<double**, Kokkos::LayoutRight> full("full", nrows, ncols);

    // Fill with a known pattern: value = row*100 + col
    Kokkos::parallel_for("fill_full",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nrows, ncols}),
        KOKKOS_LAMBDA(int r, int c) {
            full(r, c) = static_cast<double>(r * 100 + c);
        });
    Kokkos::fence();

    // Take a column subview: column 2 (this is strided/non-contiguous in row-major)
    auto col_subview = Kokkos::subview(full, Kokkos::ALL, 2);
    // col_subview is rank-1 with stride = ncols (not contiguous)

    // Verify it's non-contiguous
    EXPECT_FALSE(halo::detail::is_contiguous(col_subview));

    // Pack the strided subview
    Kokkos::View<double*> buffer("buffer", nrows);
    halo::detail::pack(col_subview, buffer);

    // Verify buffer contents
    auto buffer_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, buffer);
    for (int r = 0; r < nrows; ++r) {
        EXPECT_DOUBLE_EQ(buffer_h(r), static_cast<double>(r * 100 + 2))
            << "Mismatch at row " << r;
    }
}

// ─── 5. Unpack into a strided 2D subview (non-contiguous) ───────────────────

TEST_F(PackUnpackTest, UnpackToStrided2DSubview) {
    constexpr int nrows = 5;
    constexpr int ncols = 4;
    Kokkos::View<double**, Kokkos::LayoutRight> full("full", nrows, ncols);
    Kokkos::deep_copy(full, 0.0);

    // Buffer with known data
    Kokkos::View<double*> buffer("buffer", nrows);
    Kokkos::parallel_for("fill_buffer", nrows, KOKKOS_LAMBDA(int i) {
        buffer(i) = static_cast<double>(i * 10 + 3);
    });
    Kokkos::fence();

    // Take column 1 subview (strided)
    auto col_subview = Kokkos::subview(full, Kokkos::ALL, 1);
    EXPECT_FALSE(halo::detail::is_contiguous(col_subview));

    // Unpack into the strided subview
    halo::detail::unpack(buffer, col_subview);

    // Verify: only column 1 should have data, rest should be 0
    auto full_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, full);
    for (int r = 0; r < nrows; ++r) {
        EXPECT_DOUBLE_EQ(full_h(r, 1), static_cast<double>(r * 10 + 3))
            << "Column 1 mismatch at row " << r;
        // Other columns should remain zero
        EXPECT_DOUBLE_EQ(full_h(r, 0), 0.0) << "Column 0 should be untouched at row " << r;
        EXPECT_DOUBLE_EQ(full_h(r, 2), 0.0) << "Column 2 should be untouched at row " << r;
        EXPECT_DOUBLE_EQ(full_h(r, 3), 0.0) << "Column 3 should be untouched at row " << r;
    }
}

// ─── 6. Pack/Unpack roundtrip with strided subview ──────────────────────────

TEST_F(PackUnpackTest, PackUnpackRoundtripStridedSubview) {
    constexpr int nrows = 6;
    constexpr int ncols = 8;

    // Source 2D view (LayoutRight)
    Kokkos::View<double**, Kokkos::LayoutRight> src_full("src_full", nrows, ncols);
    Kokkos::parallel_for("fill_src",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nrows, ncols}),
        KOKKOS_LAMBDA(int r, int c) {
            src_full(r, c) = static_cast<double>(r * 10 + c);
        });
    Kokkos::fence();

    // Take column 5 subview from source
    auto src_col = Kokkos::subview(src_full, Kokkos::ALL, 5);

    // Pack
    Kokkos::View<double*> buffer("buffer", nrows);
    halo::detail::pack(src_col, buffer);

    // Destination 2D view — unpack into column 3
    Kokkos::View<double**, Kokkos::LayoutRight> dst_full("dst_full", nrows, ncols);
    Kokkos::deep_copy(dst_full, 0.0);
    auto dst_col = Kokkos::subview(dst_full, Kokkos::ALL, 3);

    // Unpack
    halo::detail::unpack(buffer, dst_col);

    // Verify: dst column 3 == src column 5
    auto src_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, src_full);
    auto dst_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dst_full);
    for (int r = 0; r < nrows; ++r) {
        EXPECT_DOUBLE_EQ(dst_h(r, 3), src_h(r, 5))
            << "Roundtrip mismatch at row " << r;
    }
}

// ─── 7. Pack/Unpack with LayoutLeft (column-major) rank-2 view ──────────────

TEST_F(PackUnpackTest, PackLayoutLeft2DView) {
    constexpr int nrows = 5;
    constexpr int ncols = 3;

    // LayoutLeft (column-major) 2D view
    Kokkos::View<double**, Kokkos::LayoutLeft> src("src_ll", nrows, ncols);
    Kokkos::parallel_for("fill_src_ll",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nrows, ncols}),
        KOKKOS_LAMBDA(int r, int c) {
            src(r, c) = static_cast<double>(r + c * 100);
        });
    Kokkos::fence();

    // A full LayoutLeft 2D view IS contiguous (span == size)
    EXPECT_TRUE(halo::detail::is_contiguous(src));

    // Pack (will take contiguous path)
    Kokkos::View<double*> buffer("buffer", nrows * ncols);
    halo::detail::pack(src, buffer);

    // Verify: contiguous pack uses data() pointer directly (LayoutLeft order)
    auto buffer_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, buffer);
    auto src_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, src);

    for (int i = 0; i < nrows * ncols; ++i) {
        EXPECT_DOUBLE_EQ(buffer_h(i), src_h.data()[i])
            << "Mismatch at linearized index " << i;
    }
}

TEST_F(PackUnpackTest, UnpackLayoutLeft2DView) {
    constexpr int nrows = 4;
    constexpr int ncols = 3;

    // Fill buffer with sequential values
    Kokkos::View<double*> buffer("buffer", nrows * ncols);
    Kokkos::parallel_for("fill_buffer", nrows * ncols, KOKKOS_LAMBDA(int i) {
        buffer(i) = static_cast<double>(i + 1);
    });
    Kokkos::fence();

    // Unpack into LayoutLeft 2D view
    Kokkos::View<double**, Kokkos::LayoutLeft> dst("dst_ll", nrows, ncols);
    halo::detail::unpack(buffer, dst);

    // Verify: contiguous unpack writes via data() pointer (LayoutLeft order)
    auto buffer_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, buffer);
    auto dst_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dst);

    for (int i = 0; i < nrows * ncols; ++i) {
        EXPECT_DOUBLE_EQ(dst_h.data()[i], buffer_h(i))
            << "Mismatch at linearized index " << i;
    }
}

// ─── 8. Pack/Unpack with LayoutRight (row-major) rank-2 view ────────────────

TEST_F(PackUnpackTest, PackLayoutRight2DView) {
    constexpr int nrows = 3;
    constexpr int ncols = 7;

    // LayoutRight (row-major) 2D view
    Kokkos::View<double**, Kokkos::LayoutRight> src("src_lr", nrows, ncols);
    Kokkos::parallel_for("fill_src_lr",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nrows, ncols}),
        KOKKOS_LAMBDA(int r, int c) {
            src(r, c) = static_cast<double>(r * 10 + c);
        });
    Kokkos::fence();

    // A full LayoutRight 2D view IS contiguous
    EXPECT_TRUE(halo::detail::is_contiguous(src));

    // Pack (contiguous path)
    Kokkos::View<double*> buffer("buffer", nrows * ncols);
    halo::detail::pack(src, buffer);

    // Verify: data matches linearized memory order
    auto buffer_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, buffer);
    auto src_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, src);

    for (int i = 0; i < nrows * ncols; ++i) {
        EXPECT_DOUBLE_EQ(buffer_h(i), src_h.data()[i])
            << "Mismatch at linearized index " << i;
    }
}

TEST_F(PackUnpackTest, UnpackLayoutRight2DView) {
    constexpr int nrows = 4;
    constexpr int ncols = 5;

    // Fill buffer
    Kokkos::View<double*> buffer("buffer", nrows * ncols);
    Kokkos::parallel_for("fill_buffer", nrows * ncols, KOKKOS_LAMBDA(int i) {
        buffer(i) = static_cast<double>(i * 2 + 1);
    });
    Kokkos::fence();

    // Unpack into LayoutRight 2D view
    Kokkos::View<double**, Kokkos::LayoutRight> dst("dst_lr", nrows, ncols);
    halo::detail::unpack(buffer, dst);

    // Verify
    auto buffer_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, buffer);
    auto dst_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dst);

    for (int i = 0; i < nrows * ncols; ++i) {
        EXPECT_DOUBLE_EQ(dst_h.data()[i], buffer_h(i))
            << "Mismatch at linearized index " << i;
    }
}

// ─── 9. Strided subview from LayoutLeft 2D ──────────────────────────────────

TEST_F(PackUnpackTest, PackStridedSubviewLayoutLeft) {
    constexpr int nrows = 5;
    constexpr int ncols = 4;

    // LayoutLeft: columns are contiguous, rows are strided
    Kokkos::View<double**, Kokkos::LayoutLeft> full("full_ll", nrows, ncols);
    Kokkos::parallel_for("fill_ll",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nrows, ncols}),
        KOKKOS_LAMBDA(int r, int c) {
            full(r, c) = static_cast<double>(r * 10 + c);
        });
    Kokkos::fence();

    // In LayoutLeft, a row subview (fixing first index) IS strided
    // A column subview (fixing second index) is contiguous
    // Take row 2: full(2, :) — this is strided in LayoutLeft
    auto row_subview = Kokkos::subview(full, 2, Kokkos::ALL);
    EXPECT_FALSE(halo::detail::is_contiguous(row_subview));

    // Pack the strided row
    Kokkos::View<double*> buffer("buffer", ncols);
    halo::detail::pack(row_subview, buffer);

    // Verify buffer contents
    auto buffer_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, buffer);
    for (int c = 0; c < ncols; ++c) {
        EXPECT_DOUBLE_EQ(buffer_h(c), static_cast<double>(2 * 10 + c))
            << "Mismatch at col " << c;
    }
}

// ─── 10. Pack a full rank-2 strided subview (MDRangePolicy path) ────────────

TEST_F(PackUnpackTest, PackStridedRank2Subview) {
    // Create a 6x8 view (LayoutRight), take a 3x4 interior subview
    constexpr int nrows = 6;
    constexpr int ncols = 8;
    Kokkos::View<double**, Kokkos::LayoutRight> full("full_lr", nrows, ncols);
    Kokkos::parallel_for("fill_full",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nrows, ncols}),
        KOKKOS_LAMBDA(int r, int c) {
            full(r, c) = static_cast<double>(r * 100 + c);
        });
    Kokkos::fence();

    // Take a rank-2 subview: rows [1,4), cols [2,6) → 3x4 subview
    auto sub = Kokkos::subview(full,
        Kokkos::make_pair(1, 4),
        Kokkos::make_pair(2, 6));

    // This rank-2 subview is non-contiguous (row stride != ncols_sub)
    EXPECT_FALSE(halo::detail::is_contiguous(sub));

    // Pack using MDRangePolicy (rank-2 strided path)
    constexpr int sub_rows = 3;
    constexpr int sub_cols = 4;
    Kokkos::View<double*> buffer("buffer", sub_rows * sub_cols);
    halo::detail::pack(sub, buffer);

    // Verify: buffer linearized in row-major order of the subview
    auto buffer_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, buffer);
    int idx = 0;
    for (int r = 1; r < 4; ++r) {
        for (int c = 2; c < 6; ++c) {
            EXPECT_DOUBLE_EQ(buffer_h(idx), static_cast<double>(r * 100 + c))
                << "Mismatch at buffer index " << idx << " (src row=" << r << " col=" << c << ")";
            ++idx;
        }
    }
}

// ─── 11. Unpack into a rank-2 strided subview ───────────────────────────────

TEST_F(PackUnpackTest, UnpackStridedRank2Subview) {
    constexpr int nrows = 6;
    constexpr int ncols = 8;
    constexpr int sub_rows = 3;
    constexpr int sub_cols = 4;

    // Fill buffer with known data
    Kokkos::View<double*> buffer("buffer", sub_rows * sub_cols);
    Kokkos::parallel_for("fill_buffer", sub_rows * sub_cols, KOKKOS_LAMBDA(int i) {
        buffer(i) = static_cast<double>(i + 100);
    });
    Kokkos::fence();

    // Destination: 6x8 view initialized to zero
    Kokkos::View<double**, Kokkos::LayoutRight> full("full_lr", nrows, ncols);
    Kokkos::deep_copy(full, 0.0);

    // Take a rank-2 subview: rows [1,4), cols [2,6) → 3x4 subview
    auto sub = Kokkos::subview(full,
        Kokkos::make_pair(1, 4),
        Kokkos::make_pair(2, 6));

    EXPECT_FALSE(halo::detail::is_contiguous(sub));

    // Unpack
    halo::detail::unpack(buffer, sub);

    // Verify: subview region has correct data, rest remains zero
    auto full_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, full);
    int idx = 0;
    for (int r = 1; r < 4; ++r) {
        for (int c = 2; c < 6; ++c) {
            EXPECT_DOUBLE_EQ(full_h(r, c), static_cast<double>(idx + 100))
                << "Mismatch at (" << r << "," << c << ")";
            ++idx;
        }
    }

    // Verify untouched regions remain zero
    for (int r = 0; r < nrows; ++r) {
        for (int c = 0; c < ncols; ++c) {
            if (r >= 1 && r < 4 && c >= 2 && c < 6) continue;
            EXPECT_DOUBLE_EQ(full_h(r, c), 0.0)
                << "Non-subview element (" << r << "," << c << ") should be zero";
        }
    }
}

// ─── 12. Execution space tag dispatch verification ──────────────────────────

TEST_F(PackUnpackTest, ExplicitExecutionSpaceInstance) {
    // Verify that we can pass an explicit execution space instance.
    // On this Docker container, Kokkos::DefaultExecutionSpace is Serial or OpenMP.
    constexpr int N = 12;

    using exec_space = Kokkos::DefaultExecutionSpace;
    exec_space exec_instance{};

    Kokkos::View<double*> src("src", N);
    Kokkos::parallel_for("fill", Kokkos::RangePolicy<exec_space>(exec_instance, 0, N),
        KOKKOS_LAMBDA(int i) { src(i) = static_cast<double>(i); });
    exec_instance.fence("fill_fence");

    Kokkos::View<double*> buffer("buffer", N);
    halo::detail::pack(src, buffer, exec_instance);

    Kokkos::View<double*> dst("dst", N);
    halo::detail::unpack(buffer, dst, exec_instance);

    auto dst_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, dst);
    for (int i = 0; i < N; ++i) {
        EXPECT_DOUBLE_EQ(dst_h(i), static_cast<double>(i));
    }
}

TEST_F(PackUnpackTest, HostExecutionSpaceForHostViews) {
    // Explicitly use Kokkos::DefaultHostExecutionSpace to verify host-side
    // pack/unpack works without requiring GPU hardware (Requirement 2.3).
    constexpr int N = 7;

    using host_space = Kokkos::DefaultHostExecutionSpace;
    using host_view = Kokkos::View<double*, Kokkos::HostSpace>;

    host_view src("src_host", N);
    host_view buffer("buffer_host", N);
    host_view dst("dst_host", N);

    // Fill on host
    for (int i = 0; i < N; ++i) {
        src(i) = static_cast<double>(i * 7 + 2);
    }

    // Pack/unpack on host execution space
    halo::detail::pack(src, buffer, host_space{});
    halo::detail::unpack(buffer, dst, host_space{});

    // Verify
    for (int i = 0; i < N; ++i) {
        EXPECT_DOUBLE_EQ(dst(i), src(i))
            << "Host-space roundtrip mismatch at index " << i;
    }
}

// ─── 13. Contiguity detection utility ───────────────────────────────────────

TEST_F(PackUnpackTest, IsContiguousDetectsCorrectly) {
    // A directly-allocated 1D view is always contiguous
    Kokkos::View<double*> v1d("v1d", 10);
    EXPECT_TRUE(halo::detail::is_contiguous(v1d));

    // A directly-allocated 2D view is contiguous
    Kokkos::View<double**, Kokkos::LayoutRight> v2d("v2d", 4, 5);
    EXPECT_TRUE(halo::detail::is_contiguous(v2d));

    // A column subview from LayoutRight is non-contiguous
    auto col = Kokkos::subview(v2d, Kokkos::ALL, 2);
    EXPECT_FALSE(halo::detail::is_contiguous(col));

    // A row range subview from LayoutRight is contiguous
    auto row_range = Kokkos::subview(v2d, Kokkos::make_pair(0, 2), Kokkos::ALL);
    EXPECT_TRUE(halo::detail::is_contiguous(row_range));
}

// ─── Custom main() for Kokkos initialization ────────────────────────────────

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    Kokkos::ScopeGuard kokkos_scope(argc, argv);
    return RUN_ALL_TESTS();
}
