# Copyright 2024 HELM Contributors
# SPDX-License-Identifier: Apache-2.0

from spack.package import *


class Halo(CMakePackage):
    """Hardware-Abstracted Link Operations (HALO) — RAII MPI with GPU-aware
    halo exchange for structured grids. Part of the HELM ecosystem."""

    homepage = "https://github.com/helm-framework/halo"
    git = "https://github.com/helm-framework/halo.git"

    maintainers("helm-maintainers")

    version("develop", branch="main")
    version("0.1.0", tag="v0.1.0")

    # ─── Variants ────────────────────────────────────────────────────────────
    variant("fortran", default=True, description="Build Fortran C-interop layer (halo_mod)")
    variant(
        "gpu_aware_mpi",
        default=False,
        description="Enable GPU-aware MPI (pass device pointers directly)",
    )
    variant("tests", default=False, description="Build test suite (GTest + RapidCheck)")

    # Kokkos backend variants — mirrors kokkos package variants
    variant("openmp", default=True, description="Enable Kokkos OpenMP backend")
    variant("cuda", default=False, description="Enable Kokkos CUDA backend")
    variant("hip", default=False, description="Enable Kokkos HIP backend")
    variant("serial", default=True, description="Enable Kokkos Serial backend")

    # ─── Dependencies ────────────────────────────────────────────────────────
    depends_on("cmake@3.21:", type="build")
    depends_on("mpi")
    depends_on("kokkos")

    # Forward Kokkos backend selections
    depends_on("kokkos+openmp", when="+openmp")
    depends_on("kokkos+cuda", when="+cuda")
    depends_on("kokkos+hip", when="+hip")
    depends_on("kokkos+serial", when="+serial")

    # Optional test dependencies
    depends_on("googletest", when="+tests", type=("build", "test"))
    depends_on("rapidcheck", when="+tests", type=("build", "test"))

    def cmake_args(self):
        args = [
            self.define_from_variant("BUILD_FORTRAN", "fortran"),
            self.define_from_variant("HALO_GPU_AWARE_MPI", "gpu_aware_mpi"),
            self.define_from_variant("BUILD_TESTING", "tests"),
        ]
        return args

    @run_after("install")
    @on_package_attributes(run_tests=True)
    def check_install(self):
        """Run the test suite after installation when tests are enabled."""
        if "+tests" in self.spec:
            with working_dir(self.build_directory):
                ctest("--output-on-failure")

    @property
    def libs(self):
        """Export the halo library for downstream packages."""
        return find_libraries("libhalo", root=self.prefix, recursive=True)
