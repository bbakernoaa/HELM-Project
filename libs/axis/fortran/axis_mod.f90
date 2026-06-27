! SPDX-License-Identifier: Apache-2.0
! AXIS — Arbitrary eXgrid Interpolation Solver
! Copyright (c) HELM Project Contributors

!> @file fortran/axis_mod.f90
!> @brief Fortran iso_c_binding module for the AXIS interpolation library.
!>
!> @details Provides type-safe Fortran wrappers around the extern "C" interop functions
!> defined in axis_c_interop.cpp. All AXIS objects (meshes, interpolation
!> matrices) are represented as opaque integer handles. Error codes are
!> propagated via integer return values or intent(out) ierr arguments.
!>
!> Usage example:
!> @code
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
!> @endcode
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

  integer(c_int), parameter, public :: AXIS_SUCCESS = 0 !< Operation completed successfully.
  integer(c_int), parameter, public :: AXIS_ERROR   = -1 !< Operation encountered a failure or exception.

  ! ───────────────────────────────────────────────────────────────────────────
  ! Interpolation method constants (must match axis_c_interop.cpp switch)
  ! ───────────────────────────────────────────────────────────────────────────

  integer(c_int), parameter, public :: AXIS_METHOD_BILINEAR     = 0 !< Bilinear interpolation method.
  integer(c_int), parameter, public :: AXIS_METHOD_NEAREST      = 1 !< Nearest neighbor (nearest-point) interpolation method.
  integer(c_int), parameter, public :: AXIS_METHOD_CONSERVATIVE = 2 !< First-order conservative interpolation method.

  ! ───────────────────────────────────────────────────────────────────────────
  ! C function interfaces (private — called by public wrapper subroutines)
  ! ───────────────────────────────────────────────────────────────────────────

  interface

    !> @brief Initialize the AXIS runtime (Kokkos backend) in the C++ layer.
    !> @details Safe to call multiple times; subsequent calls will be no-ops.
    !> @return @c AXIS_SUCCESS (0) on success, or @c AXIS_ERROR (-1) on failure.
    integer(c_int) function axis_init_c() bind(C, name="axis_init_c")
      import :: c_int
    end function axis_init_c

    !> @brief Create a mesh from a named-grid token in the C++ layer.
    !> @param[in] name Null-terminated C-style character string representing the grid name.
    !> @param[out] mesh_handle Opaque integer token/handle for the created mesh.
    !> @return @c AXIS_SUCCESS (0) on success, or @c AXIS_ERROR (-1) on failure.
    integer(c_int) function axis_mesh_from_named_c(name, mesh_handle) &
        bind(C, name="axis_mesh_from_named_c")
      import :: c_int, c_char
      character(c_char), intent(in) :: name(*)
      integer(c_int), intent(out)   :: mesh_handle
    end function axis_mesh_from_named_c

    !> @brief Create a mesh from a GridDescriptor structure in the C++ layer.
    !> @param[in] descriptor_ptr Opaque C pointer to a populated @c GridDescriptor structure.
    !> @param[out] mesh_handle Opaque integer token/handle for the created mesh.
    !> @return @c AXIS_SUCCESS (0) on success, or @c AXIS_ERROR (-1) on failure.
    integer(c_int) function axis_mesh_from_descriptor_c(descriptor_ptr, mesh_handle) &
        bind(C, name="axis_mesh_from_descriptor_c")
      import :: c_int, c_ptr
      type(c_ptr), value, intent(in) :: descriptor_ptr
      integer(c_int), intent(out)    :: mesh_handle
    end function axis_mesh_from_descriptor_c

    !> @brief Generate interpolation weights between source and destination meshes in C++.
    !> @param[in] src_handle Opaque integer token of the source mesh.
    !> @param[in] dst_handle Opaque integer token of the destination mesh.
    !> @param[in] method Integer constant identifying the interpolation method (e.g., bilinear).
    !> @param[out] matrix_handle Opaque integer token for the created interpolation matrix.
    !> @return @c AXIS_SUCCESS (0) on success, or @c AXIS_ERROR (-1) on failure.
    integer(c_int) function axis_generate_weights_c(src_handle, dst_handle, &
        method, matrix_handle) bind(C, name="axis_generate_weights_c")
      import :: c_int
      integer(c_int), value, intent(in) :: src_handle
      integer(c_int), value, intent(in) :: dst_handle
      integer(c_int), value, intent(in) :: method
      integer(c_int), intent(out)       :: matrix_handle
    end function axis_generate_weights_c

    !> @brief Apply interpolation weights using SpMV in the C++ layer.
    !> @param[in] matrix_handle Opaque integer token of the interpolation matrix.
    !> @param[in] src Opaque C pointer to the source array of @c c_double data elements.
    !> @param[in] dst Opaque C pointer to the destination array of @c c_double data elements.
    !> @param[in] n_src Number of elements in the source array.
    !> @param[in] n_dst Number of elements in the destination array.
    !> @return @c AXIS_SUCCESS (0) on success, or @c AXIS_ERROR (-1) on failure.
    integer(c_int) function axis_apply_c(matrix_handle, src, dst, n_src, n_dst) &
        bind(C, name="axis_apply_c")
      import :: c_int, c_double, c_ptr
      integer(c_int), value, intent(in) :: matrix_handle
      type(c_ptr), value, intent(in)    :: src
      type(c_ptr), value, intent(in)    :: dst
      integer(c_int), value, intent(in) :: n_src
      integer(c_int), value, intent(in) :: n_dst
    end function axis_apply_c

    !> @brief Destroy and deallocate a mesh in the C++ layer.
    !> @param[in] handle Opaque integer token of the mesh to destroy.
    !> @return @c AXIS_SUCCESS (0) on success, or @c AXIS_ERROR (-1) on failure.
    integer(c_int) function axis_destroy_mesh_c(handle) &
        bind(C, name="axis_destroy_mesh_c")
      import :: c_int
      integer(c_int), value, intent(in) :: handle
    end function axis_destroy_mesh_c

    !> @brief Destroy and deallocate an interpolation matrix in the C++ layer.
    !> @param[in] handle Opaque integer token of the interpolation matrix to destroy.
    !> @return @c AXIS_SUCCESS (0) on success, or @c AXIS_ERROR (-1) on failure.
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

  !> @brief Initialize the AXIS runtime (Kokkos backend).
  !! @details Safe to call multiple times — subsequent calls are no-ops.
  !!
  !! @param[out] ierr Status code containing @c AXIS_SUCCESS on success, or @c AXIS_ERROR on failure.
  subroutine axis_init(ierr)
    integer(c_int), intent(out) :: ierr
    ierr = axis_init_c()
  end subroutine axis_init

  !> @brief Build a mesh from a named grid token (e.g., "O1280", "F128", "N320").
  !! @details Returns an opaque integer handle representing the built mesh in C++.
  !!
  !! @param[in]  name        Character string of the grid token name.
  !! @param[out] mesh_handle Opaque integer handle for the newly created mesh.
  !! @param[out] ierr        Status code containing @c AXIS_SUCCESS on success, or @c AXIS_ERROR on failure.
  subroutine axis_mesh_from_named(name, mesh_handle, ierr)
    character(len=*), intent(in)  :: name
    integer(c_int), intent(out)   :: mesh_handle
    integer(c_int), intent(out)   :: ierr

    ! Pass as null-terminated C string
    ierr = axis_mesh_from_named_c(trim(name) // c_null_char, mesh_handle)
  end subroutine axis_mesh_from_named

  !> @brief Build a mesh from a GridDescriptor pointer (advanced use).
  !! @details The descriptor_ptr should be obtained via c_loc on a GridDescriptor
  !! structure populated by the caller.
  !!
  !! @param[in]  descriptor_ptr Opaque C pointer to the populated GridDescriptor structure.
  !! @param[out] mesh_handle    Opaque integer handle for the newly created mesh.
  !! @param[out] ierr           Status code containing @c AXIS_SUCCESS on success, or @c AXIS_ERROR on failure.
  subroutine axis_mesh_from_descriptor(descriptor_ptr, mesh_handle, ierr)
    type(c_ptr), intent(in)     :: descriptor_ptr
    integer(c_int), intent(out) :: mesh_handle
    integer(c_int), intent(out) :: ierr

    ierr = axis_mesh_from_descriptor_c(descriptor_ptr, mesh_handle)
  end subroutine axis_mesh_from_descriptor

  !> @brief Generate interpolation weights between source and destination meshes.
  !!
  !! @param[in]  src_handle    Opaque integer handle of the source mesh.
  !! @param[in]  dst_handle    Opaque integer handle of the destination mesh.
  !! @param[in]  method        Interpolation method. Must be @c AXIS_METHOD_BILINEAR,
  !!                           @c AXIS_METHOD_NEAREST, or @c AXIS_METHOD_CONSERVATIVE.
  !! @param[out] matrix_handle Opaque integer handle for the generated interpolation matrix.
  !! @param[out] ierr          Status code containing @c AXIS_SUCCESS on success, or @c AXIS_ERROR on failure.
  subroutine axis_generate_weights(src_handle, dst_handle, method, &
                                   matrix_handle, ierr)
    integer(c_int), intent(in)  :: src_handle
    integer(c_int), intent(in)  :: dst_handle
    integer(c_int), intent(in)  :: method
    integer(c_int), intent(out) :: matrix_handle
    integer(c_int), intent(out) :: ierr

    ierr = axis_generate_weights_c(src_handle, dst_handle, method, matrix_handle)
  end subroutine axis_generate_weights

  !> @brief Apply interpolation weights: dst = S * src.
  !! @details src and dst are contiguous Fortran double-precision arrays passed via c_loc.
  !! Zero-copy: the C++ side wraps the Fortran arrays as non-owning views.
  !!
  !! @param[in]    matrix_handle Opaque integer handle representing the interpolation matrix.
  !! @param[in]    src           Source field double-precision array of size @c n_src.
  !! @param[inout] dst           Destination field double-precision array of size @c n_dst.
  !! @param[in]    n_src         Number of grid points in the source mesh.
  !! @param[in]    n_dst         Number of grid points in the destination mesh.
  !! @param[out]   ierr          Status code containing @c AXIS_SUCCESS on success, or @c AXIS_ERROR on failure.
  subroutine axis_apply(matrix_handle, src, dst, n_src, n_dst, ierr)
    integer(c_int), intent(in)             :: matrix_handle
    real(c_double), intent(in), target     :: src(*)
    real(c_double), intent(inout), target  :: dst(*)
    integer(c_int), intent(in)             :: n_src
    integer(c_int), intent(in)             :: n_dst
    integer(c_int), intent(out)            :: ierr

    ierr = axis_apply_c(matrix_handle, c_loc(src(1)), c_loc(dst(1)), n_src, n_dst)
  end subroutine axis_apply

  !> @brief Destroy (release) a mesh handle.
  !! @details Frees underlying memory and resources associated with the mesh inside C++.
  !!
  !! @param[in]  handle Opaque integer handle of the mesh to destroy.
  !! @param[out] ierr   Status code containing @c AXIS_SUCCESS on success, or @c AXIS_ERROR on failure.
  subroutine axis_destroy_mesh(handle, ierr)
    integer(c_int), intent(in)  :: handle
    integer(c_int), intent(out) :: ierr

    ierr = axis_destroy_mesh_c(handle)
  end subroutine axis_destroy_mesh

  !> @brief Destroy (release) an interpolation matrix handle.
  !! @details Frees underlying memory and resources associated with the interpolation matrix inside C++.
  !!
  !! @param[in]  handle Opaque integer handle of the interpolation matrix to destroy.
  !! @param[out] ierr   Status code containing @c AXIS_SUCCESS on success, or @c AXIS_ERROR on failure.
  subroutine axis_destroy_matrix(handle, ierr)
    integer(c_int), intent(in)  :: handle
    integer(c_int), intent(out) :: ierr

    ierr = axis_destroy_matrix_c(handle)
  end subroutine axis_destroy_matrix

end module axis_mod
