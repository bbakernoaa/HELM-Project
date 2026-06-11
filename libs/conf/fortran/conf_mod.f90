!> @file conf_mod.f90
!! @brief Fortran public API for the CONF micro-library (HELM Tier 1b).
!!
!! This module provides idiomatic Fortran subroutines for legacy NUOPC/ESMF
!! models to load and query YAML configuration files. It is a thin
!! `iso_c_binding` wrapper around the `extern "C"` interop layer implemented in
!! `src/fortran/conf_c_interop.cpp`.
!!
!! Design contract (mirrors the C interop layer):
!!   - Every C function returns an `integer(c_int)` error code where 0 means
!!     success and non-zero values indicate specific error conditions.
!!   - Opaque C++ resources (Config) are exposed to Fortran as plain
!!     `integer(c_int)` handle tokens, never as raw C pointers. The token is
!!     obtained from conf_load and passed back to subsequent calls.
!!   - Fortran strings are passed as `character(len=*)` together with their
!!     length (via `len()`) to the C layer which expects (char*, int) pairs.
!!
!! Requirements: 22.1, 22.2, 22.3, 22.4, 22.5, 22.6, 22.7, 22.8
module conf_mod
  use, intrinsic :: iso_c_binding
  implicit none
  private

  ! ---------------------------------------------------------------------------
  ! Public friendly API (idiomatic Fortran subroutines).
  ! ---------------------------------------------------------------------------
  public :: conf_load, conf_close
  public :: conf_get_int, conf_get_real, conf_get_logical, conf_get_string
  public :: conf_has_key

  ! ---------------------------------------------------------------------------
  ! Public error code constants. These values mirror the Error_Code enum in
  ! include/conf/error.hpp and MUST be kept in sync with it.
  ! ---------------------------------------------------------------------------
  integer(c_int), parameter, public :: CONF_SUCCESS          = 0   !< Operation succeeded.
  integer(c_int), parameter, public :: CONF_INVALID_ARG      = 1   !< Invalid argument.
  integer(c_int), parameter, public :: CONF_FILE_NOT_FOUND   = 2   !< File path does not exist.
  integer(c_int), parameter, public :: CONF_PARSE_ERROR      = 3   !< Malformed YAML syntax.
  integer(c_int), parameter, public :: CONF_KEY_NOT_FOUND    = 4   !< Dotted-path key not found.
  integer(c_int), parameter, public :: CONF_TYPE_MISMATCH    = 5   !< Node cannot convert to type.
  integer(c_int), parameter, public :: CONF_BAD_HANDLE       = 6   !< Invalid or released handle.
  integer(c_int), parameter, public :: CONF_BUFFER_TOO_SMALL = 7   !< Buffer cannot hold value.
  integer(c_int), parameter, public :: CONF_UNKNOWN          = 99  !< Unexpected error.

  ! ---------------------------------------------------------------------------
  ! Explicit interfaces to the extern "C" interop functions. Each function
  ! carries the bind(c, name=...) attribute matching the C symbol exactly.
  ! Every function returns the integer error code as its result.
  ! ---------------------------------------------------------------------------
  interface

    !> Parse a YAML file and register a Config handle.
    function conf_load_c(path, path_len, handle_out) &
        bind(c, name='conf_load_c') result(ierr)
      import :: c_int, c_char
      character(kind=c_char), intent(in) :: path(*)
      integer(c_int), value, intent(in)  :: path_len
      integer(c_int), intent(out)        :: handle_out
      integer(c_int)                     :: ierr
    end function conf_load_c

    !> Destroy a Config and invalidate its handle.
    function conf_close_c(handle) &
        bind(c, name='conf_close_c') result(ierr)
      import :: c_int
      integer(c_int), value, intent(in) :: handle
      integer(c_int)                    :: ierr
    end function conf_close_c

    !> Check if a dotted-path key exists.
    function conf_has_key_c(handle, key, key_len, exists_out) &
        bind(c, name='conf_has_key_c') result(ierr)
      import :: c_int, c_char
      integer(c_int), value, intent(in)  :: handle
      character(kind=c_char), intent(in) :: key(*)
      integer(c_int), value, intent(in)  :: key_len
      integer(c_int), intent(out)        :: exists_out
      integer(c_int)                     :: ierr
    end function conf_has_key_c

    !> Get an integer value by dotted-path key.
    function conf_get_int_c(handle, key, key_len, out) &
        bind(c, name='conf_get_int_c') result(ierr)
      import :: c_int, c_char
      integer(c_int), value, intent(in)  :: handle
      character(kind=c_char), intent(in) :: key(*)
      integer(c_int), value, intent(in)  :: key_len
      integer(c_int), intent(out)        :: out
      integer(c_int)                     :: ierr
    end function conf_get_int_c

    !> Get a double value by dotted-path key.
    function conf_get_double_c(handle, key, key_len, out) &
        bind(c, name='conf_get_double_c') result(ierr)
      import :: c_int, c_char, c_double
      integer(c_int), value, intent(in)  :: handle
      character(kind=c_char), intent(in) :: key(*)
      integer(c_int), value, intent(in)  :: key_len
      real(c_double), intent(out)        :: out
      integer(c_int)                     :: ierr
    end function conf_get_double_c

    !> Get a boolean value (as 0/1 integer) by dotted-path key.
    function conf_get_bool_c(handle, key, key_len, out) &
        bind(c, name='conf_get_bool_c') result(ierr)
      import :: c_int, c_char
      integer(c_int), value, intent(in)  :: handle
      character(kind=c_char), intent(in) :: key(*)
      integer(c_int), value, intent(in)  :: key_len
      integer(c_int), intent(out)        :: out
      integer(c_int)                     :: ierr
    end function conf_get_bool_c

    !> Get the length of a string value (step 1 of length-first protocol).
    function conf_get_string_len_c(handle, key, key_len, str_len_out) &
        bind(c, name='conf_get_string_len_c') result(ierr)
      import :: c_int, c_char
      integer(c_int), value, intent(in)  :: handle
      character(kind=c_char), intent(in) :: key(*)
      integer(c_int), value, intent(in)  :: key_len
      integer(c_int), intent(out)        :: str_len_out
      integer(c_int)                     :: ierr
    end function conf_get_string_len_c

    !> Copy a string value into caller buffer (step 2 of length-first protocol).
    function conf_get_string_c(handle, key, key_len, buf, buf_cap, written_out) &
        bind(c, name='conf_get_string_c') result(ierr)
      import :: c_int, c_char
      integer(c_int), value, intent(in)  :: handle
      character(kind=c_char), intent(in) :: key(*)
      integer(c_int), value, intent(in)  :: key_len
      character(kind=c_char), intent(out) :: buf(*)
      integer(c_int), value, intent(in)  :: buf_cap
      integer(c_int), intent(out)        :: written_out
      integer(c_int)                     :: ierr
    end function conf_get_string_c

  end interface

contains

  !> @brief Load a YAML configuration file and return an opaque handle.
  !!
  !! @param[in]  path    File path to the YAML configuration file.
  !! @param[out] handle  Opaque Config handle token (positive integer on success).
  !! @param[out] status  CONF_SUCCESS on success, otherwise an error code.
  subroutine conf_load(path, handle, status)
    character(len=*), intent(in)  :: path
    integer(c_int), intent(out)   :: handle
    integer(c_int), intent(out)   :: status

    status = conf_load_c(path, len(path, kind=c_int), handle)
  end subroutine conf_load

  !> @brief Close a Config and invalidate its handle, freeing all resources.
  !!
  !! @param[in]  handle  Opaque Config handle token to close.
  !! @param[out] status  CONF_SUCCESS on success, otherwise an error code.
  subroutine conf_close(handle, status)
    integer(c_int), intent(in)  :: handle
    integer(c_int), intent(out) :: status

    status = conf_close_c(handle)
  end subroutine conf_close

  !> @brief Query an integer value by dotted-path key.
  !!
  !! @param[in]  handle  Opaque Config handle token.
  !! @param[in]  key     Dotted-path key (e.g., "model.physics.layers").
  !! @param[out] value   The integer value on success.
  !! @param[out] status  CONF_SUCCESS on success, otherwise an error code.
  subroutine conf_get_int(handle, key, value, status)
    integer(c_int), intent(in)  :: handle
    character(len=*), intent(in) :: key
    integer(c_int), intent(out) :: value
    integer(c_int), intent(out) :: status

    status = conf_get_int_c(handle, key, len(key, kind=c_int), value)
  end subroutine conf_get_int

  !> @brief Query a real(c_double) value by dotted-path key.
  !!
  !! @param[in]  handle  Opaque Config handle token.
  !! @param[in]  key     Dotted-path key.
  !! @param[out] value   The double-precision real value on success.
  !! @param[out] status  CONF_SUCCESS on success, otherwise an error code.
  subroutine conf_get_real(handle, key, value, status)
    integer(c_int), intent(in)   :: handle
    character(len=*), intent(in) :: key
    real(c_double), intent(out)  :: value
    integer(c_int), intent(out)  :: status

    status = conf_get_double_c(handle, key, len(key, kind=c_int), value)
  end subroutine conf_get_real

  !> @brief Query a logical value by dotted-path key.
  !!
  !! The C layer returns 0/1; this wrapper converts to .false./.true.
  !!
  !! @param[in]  handle  Opaque Config handle token.
  !! @param[in]  key     Dotted-path key.
  !! @param[out] value   The logical value on success.
  !! @param[out] status  CONF_SUCCESS on success, otherwise an error code.
  subroutine conf_get_logical(handle, key, value, status)
    integer(c_int), intent(in)   :: handle
    character(len=*), intent(in) :: key
    logical, intent(out)         :: value
    integer(c_int), intent(out)  :: status
    integer(c_int) :: bool_int

    bool_int = 0_c_int
    status = conf_get_bool_c(handle, key, len(key, kind=c_int), bool_int)
    value = (bool_int == 1_c_int)
  end subroutine conf_get_logical

  !> @brief Query a string value by dotted-path key (length-first protocol).
  !!
  !! Implements the two-step length-first protocol internally:
  !!   1. Query the string length via conf_get_string_len_c.
  !!   2. Allocate the output character(len=:) to exactly that size.
  !!   3. Copy the value via conf_get_string_c.
  !!
  !! On any error, returns a zero-length string and non-SUCCESS status
  !! without leaving a partial allocation.
  !!
  !! @param[in]  handle  Opaque Config handle token.
  !! @param[in]  key     Dotted-path key.
  !! @param[out] value   Allocatable character string (allocated on success).
  !! @param[out] status  CONF_SUCCESS on success, otherwise an error code.
  subroutine conf_get_string(handle, key, value, status)
    integer(c_int), intent(in)                :: handle
    character(len=*), intent(in)              :: key
    character(len=:), allocatable, intent(out) :: value
    integer(c_int), intent(out)               :: status
    integer(c_int) :: str_len, written

    ! Step 1: get the string length
    status = conf_get_string_len_c(handle, key, len(key, kind=c_int), str_len)
    if (status /= CONF_SUCCESS) then
      value = ''
      return
    end if

    ! Step 2: allocate buffer and copy string
    allocate(character(len=str_len) :: value)
    status = conf_get_string_c(handle, key, len(key, kind=c_int), &
                               value, str_len, written)
    if (status /= CONF_SUCCESS) then
      deallocate(value)
      value = ''
    end if
  end subroutine conf_get_string

  !> @brief Check if a dotted-path key exists in the configuration.
  !!
  !! The C layer returns 0/1; this wrapper converts to .false./.true.
  !!
  !! @param[in]  handle  Opaque Config handle token.
  !! @param[in]  key     Dotted-path key.
  !! @param[out] exists  .true. if the key exists, .false. otherwise.
  !! @param[out] status  CONF_SUCCESS on success, otherwise an error code.
  subroutine conf_has_key(handle, key, exists, status)
    integer(c_int), intent(in)   :: handle
    character(len=*), intent(in) :: key
    logical, intent(out)         :: exists
    integer(c_int), intent(out)  :: status
    integer(c_int) :: exists_int

    exists_int = 0_c_int
    status = conf_has_key_c(handle, key, len(key, kind=c_int), exists_int)
    exists = (exists_int == 1_c_int)
  end subroutine conf_has_key

end module conf_mod
