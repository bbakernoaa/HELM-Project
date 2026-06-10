! SPDX-License-Identifier: Apache-2.0
! AXIS — Arbitrary eXgrid Interpolation Solver
! Copyright (c) HELM Project Contributors

!> @file fortran/axis_mod.f90
!> @brief Fortran iso_c_binding module for the AXIS interpolation library.
!>
!> Provides type-safe Fortran wrappers around the extern "C" interop functions
!> defined in axis_c_interop.cpp. All AXIS objects (meshes, interpolation
!> matrices) are represented as opaque integer handles. Error codes are
!> propagated via integer return values or intent(out) ierr arguments.
!>
!> Usage example:
!>   use axis_mod
!>   integer :: src_mesh, dst_mesh, matrix, ierr
!>   call axis_init(ierr)
!>   call axis_mesh_from_named("O1280", src_mesh, ierr)
!>   call axis_mesh_from_named("N320", dst_mesh, ierr)
!>   call axis_generate_weights(src_mesh, dst_mesh, AXIS_METHOD_BILINEAR, matrix, ierr)
!>   call axis_apply(matrix, src_field, dst_field, n_src, n_dst, ierr)
!>   call axis_destroy_matrix(matrix, ierr)
!>   call axis_destroy_mesh(src_mesh, ierr)
!>   call axis_destroy_mesh(dst_mesh, ierr)

module axis_mod
  use iso_c_binding
  implicit none
  private

  ! ───────────────────────────────────────────────────────────────────────────
  ! Public subroutines
  ! ───────────────────────────────────────────────────────────────────────────

  public :: axis_init
  public :: axis_mesh_from_named
  public :: axis_mesh_from_descriptor
  public :: axis_generate_weights
  public :: axis_apply
  public :: axis_destroy_mesh
  public :: axis_destroy_matrix

  ! ───────────────────────────────────────────────────────────────────────────
  ! Error code constants
  ! ───────────────────────────────────────────────────────────────────────────

  integer(c_int), parameter, public :: AXIS_SUCCESS = 0
  integer(c_int), parameter, public :: AXIS_ERROR   = -1

  ! ───────────────────────────────────────────────────────────────────────────
  ! Interpolation method constants (must match axis_c_interop.cpp switch)
  ! ───────────────────────────────────────────────────────────────────────────

  integer(c_int), parameter, public :: AXIS_METHOD_BILINEAR     = 0
  integer(c_int), parameter, public :: AXIS_METHOD_NEAREST      = 1
  integer(c_int), parameter, public :: AXIS_METHOD_CONSERVATIVE = 2

  ! ───────────────────────────────────────────────────────────────────────────
  ! C function interfaces (private — called by public wrapper subroutines)
  ! ───────────────────────────────────────────────────────────────────────────

  interface

    integer(c_int) function axis_init_c() bind(C, name="axis_init_c")
      import :: c_int
    end function axis_init_c

    integer(c_int) function axis_mesh_from_named_c(name, mesh_handle) &
        bind(C, name="axis_mesh_from_named_c")
      import :: c_int, c_char
      character(c_char), intent(in) :: name(*)
      integer(c_int), intent(out)   :: mesh_handle
    end function axis_mesh_from_named_c

    integer(c_int) function axis_mesh_from_descriptor_c(descriptor_ptr, mesh_handle) &
        bind(C, name="axis_mesh_from_descriptor_c")
      import :: c_int, c_ptr
      type(c_ptr), value, intent(in) :: descriptor_ptr
      integer(c_int), intent(out)    :: mesh_handle
    end function axis_mesh_from_descriptor_c

    integer(c_int) function axis_generate_weights_c(src_handle, dst_handle, &
        method, matrix_handle) bind(C, name="axis_generate_weights_c")
      import :: c_int
      integer(c_int), value, intent(in) :: src_handle
      integer(c_int), value, intent(in) :: dst_handle
      integer(c_int), value, intent(in) :: method
      integer(c_int), intent(out)       :: matrix_handle
    end function axis_generate_weights_c

    integer(c_int) function axis_apply_c(matrix_handle, src, dst, n_src, n_dst) &
        bind(C, name="axis_apply_c")
      import :: c_int, c_double, c_ptr
      integer(c_int), value, intent(in) :: matrix_handle
      type(c_ptr), value, intent(in)    :: src
      type(c_ptr), value, intent(in)    :: dst
      integer(c_int), value, intent(in) :: n_src
      integer(c_int), value, intent(in) :: n_dst
    end function axis_apply_c

    integer(c_int) function axis_destroy_mesh_c(handle) &
        bind(C, name="axis_destroy_mesh_c")
      import :: c_int
      integer(c_int), value, intent(in) :: handle
    end function axis_destroy_mesh_c

    integer(c_int) function axis_destroy_matrix_c(handle) &
        bind(C, name="axis_destroy_matrix_c")
      import :: c_int
      integer(c_int), value, intent(in) :: handle
    end function axis_destroy_matrix_c

  end interface

contains

  ! ───────────────────────────────────────────────────────────────────────────
  ! Public wrapper subroutines
  ! ───────────────────────────────────────────────────────────────────────────

  !> Initialize the AXIS runtime (Kokkos backend).
  !> Safe to call multiple times — subsequent calls are no-ops.
  subroutine axis_init(ierr)
    integer(c_int), intent(out) :: ierr
    ierr = axis_init_c()
  end subroutine axis_init

  !> Build a mesh from a named grid token (e.g. "O1280", "F128", "N320").
  !> Returns an opaque integer handle for the mesh.
  subroutine axis_mesh_from_named(name, mesh_handle, ierr)
    character(len=*), intent(in)  :: name
    integer(c_int), intent(out)   :: mesh_handle
    integer(c_int), intent(out)   :: ierr

    ! Pass as null-terminated C string
    ierr = axis_mesh_from_named_c(trim(name) // c_null_char, mesh_handle)
  end subroutine axis_mesh_from_named

  !> Build a mesh from a GridDescriptor pointer (advanced use).
  !> The descriptor_ptr should be obtained via c_loc on a GridDescriptor
  !> structure populated by the caller.
  subroutine axis_mesh_from_descriptor(descriptor_ptr, mesh_handle, ierr)
    type(c_ptr), intent(in)     :: descriptor_ptr
    integer(c_int), intent(out) :: mesh_handle
    integer(c_int), intent(out) :: ierr

    ierr = axis_mesh_from_descriptor_c(descriptor_ptr, mesh_handle)
  end subroutine axis_mesh_from_descriptor

  !> Generate interpolation weights between source and destination meshes.
  !> method should be one of AXIS_METHOD_BILINEAR, AXIS_METHOD_NEAREST,
  !> or AXIS_METHOD_CONSERVATIVE.
  subroutine axis_generate_weights(src_handle, dst_handle, method, &
                                   matrix_handle, ierr)
    integer(c_int), intent(in)  :: src_handle
    integer(c_int), intent(in)  :: dst_handle
    integer(c_int), intent(in)  :: method
    integer(c_int), intent(out) :: matrix_handle
    integer(c_int), intent(out) :: ierr

    ierr = axis_generate_weights_c(src_handle, dst_handle, method, matrix_handle)
  end subroutine axis_generate_weights

  !> Apply interpolation weights: dst = S * src.
  !> src and dst are contiguous Fortran arrays passed via c_loc.
  !> Zero-copy: the C++ side wraps the Fortran arrays as non-owning views.
  subroutine axis_apply(matrix_handle, src, dst, n_src, n_dst, ierr)
    integer(c_int), intent(in)             :: matrix_handle
    real(c_double), intent(in), target     :: src(*)
    real(c_double), intent(inout), target  :: dst(*)
    integer(c_int), intent(in)             :: n_src
    integer(c_int), intent(in)             :: n_dst
    integer(c_int), intent(out)            :: ierr

    ierr = axis_apply_c(matrix_handle, c_loc(src(1)), c_loc(dst(1)), n_src, n_dst)
  end subroutine axis_apply

  !> Destroy (release) a mesh handle.
  subroutine axis_destroy_mesh(handle, ierr)
    integer(c_int), intent(in)  :: handle
    integer(c_int), intent(out) :: ierr

    ierr = axis_destroy_mesh_c(handle)
  end subroutine axis_destroy_mesh

  !> Destroy (release) an interpolation matrix handle.
  subroutine axis_destroy_matrix(handle, ierr)
    integer(c_int), intent(in)  :: handle
    integer(c_int), intent(out) :: ierr

    ierr = axis_destroy_matrix_c(handle)
  end subroutine axis_destroy_matrix

end module axis_mod
