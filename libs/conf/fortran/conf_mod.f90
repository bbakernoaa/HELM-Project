!> @file conf_mod.f90
!> @brief Idiomatic Fortran wrapper module for the CONF configuration parser.
!>
!> conf_mod is the public Fortran face of the CONF micro-library. It wraps the
!> extern "C" bridge implemented in src/fortran/conf_c_interop.cpp through
!> iso_c_binding interface blocks, presenting native Fortran types (default
!> integer/real/logical and deferred-length allocatable strings) to legacy
!> model code. It mirrors HALO's Fortran/iso_c_binding pattern:
!>
!>   * Every C bridge function returns an int that is a conf::Error_Code; the
!>     wrappers surface that code through an intent(out) status argument.
!>   * Opaque conf::Config handles are stored as integer(c_int) tokens and
!>     passed back to the bridge unchanged.
!>   * Input keys/paths are marshalled to the C side as a (bytes, length) pair
!>     because Fortran character data is not null-terminated; the C bridge takes
!>     a (const char*, int len) convention and never relies on a terminator.
!>   * conf_get_string performs the length-first ("two-call") protocol entirely
!>     inside the wrapper, so the caller does no manual allocation or free.
!>
!> Portability: the module restricts itself to Fortran 2008 features available
!> in gfortran 9+, Intel ifort/ifx, and NVIDIA nvfortran. It uses only assumed-
!> size (dimension(*)) interoperable dummy arrays — never assumed-shape C
!> descriptors (TS 29113) — and avoids all compiler-specific extensions.
!>
!> Requirements: 22.1, 22.2, 22.3, 22.4, 22.5, 22.6, 22.7, 22.8
module conf_mod
    use, intrinsic :: iso_c_binding
    implicit none
    private

    ! ── Public error-code parameters (one per conf::Error_Code enumerator) ───
    ! Each parameter equals the integer value of the matching C++ enumerator in
    ! include/conf/error.hpp. These values are part of the C ABI and are
    ! append-only: never renumber or reuse. (Requirement 22.2, 11.5)
    public :: CONF_SUCCESS
    public :: CONF_INVALID_ARG
    public :: CONF_FILE_NOT_FOUND
    public :: CONF_PARSE_ERROR
    public :: CONF_KEY_NOT_FOUND
    public :: CONF_TYPE_MISMATCH
    public :: CONF_BAD_HANDLE
    public :: CONF_BUFFER_TOO_SMALL
    public :: CONF_UNKNOWN

    ! ── Public wrapper procedures (idiomatic Fortran signatures) ─────────────
    public :: conf_load
    public :: conf_close
    public :: conf_get_int
    public :: conf_get_real
    public :: conf_get_logical
    public :: conf_has_key
    public :: conf_get_string

    ! ── Error_Code parameter constants ───────────────────────────────────────
    integer(c_int), parameter :: CONF_SUCCESS          = 0_c_int
    integer(c_int), parameter :: CONF_INVALID_ARG      = 1_c_int
    integer(c_int), parameter :: CONF_FILE_NOT_FOUND   = 2_c_int
    integer(c_int), parameter :: CONF_PARSE_ERROR      = 3_c_int
    integer(c_int), parameter :: CONF_KEY_NOT_FOUND    = 4_c_int
    integer(c_int), parameter :: CONF_TYPE_MISMATCH    = 5_c_int
    integer(c_int), parameter :: CONF_BAD_HANDLE       = 6_c_int
    integer(c_int), parameter :: CONF_BUFFER_TOO_SMALL = 7_c_int
    integer(c_int), parameter :: CONF_UNKNOWN          = 99_c_int

    ! ── bind(C) interface blocks for the extern "C" bridge ───────────────────
    ! Signatures mirror src/fortran/conf_c_interop.cpp exactly. Scalar inputs
    ! passed by value carry the `value` attribute; pointer outputs and string
    ! buffers are passed by reference. Strings are passed as an assumed-size
    ! character(kind=c_char) array together with a separate integer length.
    interface

        !> int conf_load_c(const char* path, int path_len, int* handle_out)
        function conf_load_c(path, path_len, handle_out) &
                bind(C, name="conf_load_c") result(rc)
            import :: c_char, c_int
            character(kind=c_char), dimension(*), intent(in) :: path
            integer(c_int), value                            :: path_len
            integer(c_int), intent(out)                      :: handle_out
            integer(c_int)                                   :: rc
        end function conf_load_c

        !> int conf_load_string_c(const char* yaml_text, int text_len, int* handle_out)
        function conf_load_string_c(yaml_text, text_len, handle_out) &
                bind(C, name="conf_load_string_c") result(rc)
            import :: c_char, c_int
            character(kind=c_char), dimension(*), intent(in) :: yaml_text
            integer(c_int), value                            :: text_len
            integer(c_int), intent(out)                      :: handle_out
            integer(c_int)                                   :: rc
        end function conf_load_string_c

        !> int conf_close_c(int handle)
        function conf_close_c(handle) &
                bind(C, name="conf_close_c") result(rc)
            import :: c_int
            integer(c_int), value :: handle
            integer(c_int)        :: rc
        end function conf_close_c

        !> int conf_has_key_c(int handle, const char* key, int key_len, int* exists_out)
        function conf_has_key_c(handle, key, key_len, exists_out) &
                bind(C, name="conf_has_key_c") result(rc)
            import :: c_char, c_int
            integer(c_int), value                            :: handle
            character(kind=c_char), dimension(*), intent(in) :: key
            integer(c_int), value                            :: key_len
            integer(c_int), intent(out)                      :: exists_out
            integer(c_int)                                   :: rc
        end function conf_has_key_c

        !> int conf_size_c(int handle, const char* key, int key_len, int* size_out)
        function conf_size_c(handle, key, key_len, size_out) &
                bind(C, name="conf_size_c") result(rc)
            import :: c_char, c_int
            integer(c_int), value                            :: handle
            character(kind=c_char), dimension(*), intent(in) :: key
            integer(c_int), value                            :: key_len
            integer(c_int), intent(out)                      :: size_out
            integer(c_int)                                   :: rc
        end function conf_size_c

        !> int conf_get_int_c(int handle, const char* key, int key_len, int* out)
        function conf_get_int_c(handle, key, key_len, out) &
                bind(C, name="conf_get_int_c") result(rc)
            import :: c_char, c_int
            integer(c_int), value                            :: handle
            character(kind=c_char), dimension(*), intent(in) :: key
            integer(c_int), value                            :: key_len
            integer(c_int), intent(out)                      :: out
            integer(c_int)                                   :: rc
        end function conf_get_int_c

        !> int conf_get_double_c(int handle, const char* key, int key_len, double* out)
        function conf_get_double_c(handle, key, key_len, out) &
                bind(C, name="conf_get_double_c") result(rc)
            import :: c_char, c_int, c_double
            integer(c_int), value                            :: handle
            character(kind=c_char), dimension(*), intent(in) :: key
            integer(c_int), value                            :: key_len
            real(c_double), intent(out)                      :: out
            integer(c_int)                                   :: rc
        end function conf_get_double_c

        !> int conf_get_bool_c(int handle, const char* key, int key_len, int* out)  // 0/1
        function conf_get_bool_c(handle, key, key_len, out) &
                bind(C, name="conf_get_bool_c") result(rc)
            import :: c_char, c_int
            integer(c_int), value                            :: handle
            character(kind=c_char), dimension(*), intent(in) :: key
            integer(c_int), value                            :: key_len
            integer(c_int), intent(out)                      :: out
            integer(c_int)                                   :: rc
        end function conf_get_bool_c

        !> int conf_get_string_len_c(int handle, const char* key, int key_len, int* str_len_out)
        function conf_get_string_len_c(handle, key, key_len, str_len_out) &
                bind(C, name="conf_get_string_len_c") result(rc)
            import :: c_char, c_int
            integer(c_int), value                            :: handle
            character(kind=c_char), dimension(*), intent(in) :: key
            integer(c_int), value                            :: key_len
            integer(c_int), intent(out)                      :: str_len_out
            integer(c_int)                                   :: rc
        end function conf_get_string_len_c

        !> int conf_get_string_c(int handle, const char* key, int key_len,
        !>                       char* buf, int buf_cap, int* written_out)
        function conf_get_string_c(handle, key, key_len, buf, buf_cap, written_out) &
                bind(C, name="conf_get_string_c") result(rc)
            import :: c_char, c_int
            integer(c_int), value                            :: handle
            character(kind=c_char), dimension(*), intent(in) :: key
            integer(c_int), value                            :: key_len
            character(kind=c_char), dimension(*), intent(out):: buf
            integer(c_int), value                            :: buf_cap
            integer(c_int), intent(out)                      :: written_out
            integer(c_int)                                   :: rc
        end function conf_get_string_c

    end interface

contains

    ! ── Lifecycle wrappers ───────────────────────────────────────────────────

    !> @brief Parse a YAML file and return an opaque handle.
    !>
    !> @param[in]  path    Path to the YAML file (trailing blanks ignored).
    !> @param[out] handle  Opaque token for the loaded Config; valid only when
    !>                     status == CONF_SUCCESS. Pass it back to other
    !>                     conf_* wrappers unchanged.
    !> @param[out] status  CONF_SUCCESS on success, else a non-zero Error_Code
    !>                     (CONF_INVALID_ARG / CONF_FILE_NOT_FOUND /
    !>                     CONF_PARSE_ERROR / CONF_UNKNOWN).
    subroutine conf_load(path, handle, status)
        character(len=*), intent(in)  :: path
        integer(c_int),   intent(out) :: handle
        integer(c_int),   intent(out) :: status

        character(kind=c_char), allocatable :: cpath(:)
        integer(c_int)                      :: clen

        ! A failed load never yields a usable handle; default it to the invalid
        ! sentinel (0) so a caller that ignores status cannot reuse a stale id.
        handle = 0_c_int
        call f_to_c_bytes(path, cpath, clen)
        status = conf_load_c(cpath, clen, handle)
    end subroutine conf_load

    !> @brief Destroy a Config and invalidate its handle (RAII frees the tree).
    !>
    !> @param[in]  handle  Opaque token previously returned by conf_load.
    !> @param[out] status  CONF_SUCCESS on success, else CONF_BAD_HANDLE.
    subroutine conf_close(handle, status)
        integer(c_int), intent(in)  :: handle
        integer(c_int), intent(out) :: status

        status = conf_close_c(handle)
    end subroutine conf_close

    ! ── Scalar getters ───────────────────────────────────────────────────────

    !> @brief Read an integer value at a dotted-path key.
    !>
    !> @param[in]  handle  Opaque token of the Config.
    !> @param[in]  key     Dotted-path key (trailing blanks ignored).
    !> @param[out] value   The integer value; written only when status ==
    !>                     CONF_SUCCESS.
    !> @param[out] status  CONF_SUCCESS, or CONF_INVALID_ARG / CONF_BAD_HANDLE /
    !>                     CONF_KEY_NOT_FOUND / CONF_TYPE_MISMATCH.
    subroutine conf_get_int(handle, key, value, status)
        integer(c_int),   intent(in)  :: handle
        character(len=*), intent(in)  :: key
        integer(c_int),   intent(out) :: value
        integer(c_int),   intent(out) :: status

        character(kind=c_char), allocatable :: ckey(:)
        integer(c_int)                      :: clen

        call f_to_c_bytes(key, ckey, clen)
        status = conf_get_int_c(handle, ckey, clen, value)
    end subroutine conf_get_int

    !> @brief Read a real (double-precision) value at a dotted-path key.
    !>
    !> Maps to the C bridge's conf_get_double_c. The returned value uses the
    !> c_double kind so it is bit-faithful to the C++ double the core produced.
    !>
    !> @param[in]  handle  Opaque token of the Config.
    !> @param[in]  key     Dotted-path key (trailing blanks ignored).
    !> @param[out] value   The double value; written only when status ==
    !>                     CONF_SUCCESS.
    !> @param[out] status  CONF_SUCCESS, or CONF_INVALID_ARG / CONF_BAD_HANDLE /
    !>                     CONF_KEY_NOT_FOUND / CONF_TYPE_MISMATCH.
    subroutine conf_get_real(handle, key, value, status)
        integer(c_int),   intent(in)  :: handle
        character(len=*), intent(in)  :: key
        real(c_double),   intent(out) :: value
        integer(c_int),   intent(out) :: status

        character(kind=c_char), allocatable :: ckey(:)
        integer(c_int)                      :: clen

        call f_to_c_bytes(key, ckey, clen)
        status = conf_get_double_c(handle, ckey, clen, value)
    end subroutine conf_get_real

    !> @brief Read a logical value at a dotted-path key.
    !>
    !> The C bridge reports 1 for true and 0 for false; this wrapper converts
    !> that into a Fortran default logical.
    !>
    !> @param[in]  handle  Opaque token of the Config.
    !> @param[in]  key     Dotted-path key (trailing blanks ignored).
    !> @param[out] value   .true. / .false.; written only when status ==
    !>                     CONF_SUCCESS.
    !> @param[out] status  CONF_SUCCESS, or CONF_INVALID_ARG / CONF_BAD_HANDLE /
    !>                     CONF_KEY_NOT_FOUND / CONF_TYPE_MISMATCH.
    subroutine conf_get_logical(handle, key, value, status)
        integer(c_int),   intent(in)  :: handle
        character(len=*), intent(in)  :: key
        logical,          intent(out) :: value
        integer(c_int),   intent(out) :: status

        character(kind=c_char), allocatable :: ckey(:)
        integer(c_int)                      :: clen
        integer(c_int)                      :: ival

        ! Default to .false. so an ignored-status caller never observes an
        ! indeterminate logical on failure.
        value = .false.
        call f_to_c_bytes(key, ckey, clen)
        status = conf_get_bool_c(handle, ckey, clen, ival)
        if (status == CONF_SUCCESS) value = (ival /= 0_c_int)
    end subroutine conf_get_logical

    ! ── Existence ─────────────────────────────────────────────────────────────

    !> @brief Report whether a dotted-path key resolves.
    !>
    !> A missing key is NOT an error: exists is set .false. and status is
    !> CONF_SUCCESS. Errors are limited to CONF_INVALID_ARG / CONF_BAD_HANDLE.
    !>
    !> @param[in]  handle  Opaque token of the Config.
    !> @param[in]  key     Dotted-path key (trailing blanks ignored).
    !> @param[out] exists  .true. if the key resolves, else .false.
    !> @param[out] status  CONF_SUCCESS, or CONF_INVALID_ARG / CONF_BAD_HANDLE.
    subroutine conf_has_key(handle, key, exists, status)
        integer(c_int),   intent(in)  :: handle
        character(len=*), intent(in)  :: key
        logical,          intent(out) :: exists
        integer(c_int),   intent(out) :: status

        character(kind=c_char), allocatable :: ckey(:)
        integer(c_int)                      :: clen
        integer(c_int)                      :: iexists

        exists = .false.
        call f_to_c_bytes(key, ckey, clen)
        status = conf_has_key_c(handle, ckey, clen, iexists)
        if (status == CONF_SUCCESS) exists = (iexists /= 0_c_int)
    end subroutine conf_has_key

    ! ── String getter (length-first protocol handled internally) ─────────────

    !> @brief Read a string value, allocating the result to its exact length.
    !>
    !> Performs the length-first protocol internally: it queries the value's
    !> byte count, allocates a receiving buffer of exactly that size, copies the
    !> bytes, and returns a deferred-length allocatable string. The caller does
    !> no manual allocation or free.
    !>
    !> On any failure (missing key, type mismatch, bad handle, invalid argument)
    !> value is returned as a zero-length string and status carries the
    !> non-Success Error_Code. No partial allocation is left behind: value is
    !> assigned exactly once, after the copy has fully succeeded.
    !>
    !> @param[in]  handle  Opaque token of the Config.
    !> @param[in]  key     Dotted-path key (trailing blanks ignored).
    !> @param[out] value   The string value, allocated to its exact byte length
    !>                     (zero-length on any failure).
    !> @param[out] status  CONF_SUCCESS, or CONF_INVALID_ARG / CONF_BAD_HANDLE /
    !>                     CONF_KEY_NOT_FOUND / CONF_TYPE_MISMATCH /
    !>                     CONF_BUFFER_TOO_SMALL.
    subroutine conf_get_string(handle, key, value, status)
        integer(c_int),                intent(in)  :: handle
        character(len=*),              intent(in)  :: key
        character(len=:), allocatable, intent(out) :: value
        integer(c_int),                intent(out) :: status

        character(kind=c_char), allocatable :: ckey(:)
        character(kind=c_char), allocatable :: cbuf(:)
        integer(c_int)                      :: clen
        integer(c_int)                      :: nbytes
        integer(c_int)                      :: written

        call f_to_c_bytes(key, ckey, clen)

        ! STEP 1: query the exact byte count of the value.
        status = conf_get_string_len_c(handle, ckey, clen, nbytes)
        if (status /= CONF_SUCCESS) then
            value = ''          ! zero-length result; nothing allocated to free
            return
        end if

        ! Zero-length value: nothing to copy, return an empty string.
        if (nbytes <= 0_c_int) then
            value  = ''
            status = CONF_SUCCESS
            return
        end if

        ! STEP 2: allocate a receiving buffer of exactly nbytes and copy.
        allocate(cbuf(nbytes))
        status = conf_get_string_c(handle, ckey, clen, cbuf, nbytes, written)
        if (status /= CONF_SUCCESS) then
            value = ''          ! discard temp buffer; leave no partial result
            deallocate(cbuf)
            return
        end if

        ! Byte-faithful copy of the c_char buffer into a default-kind string.
        value = c_bytes_to_string(cbuf, written)
        deallocate(cbuf)
    end subroutine conf_get_string

    ! ── Private marshalling helpers ───────────────────────────────────────────

    !> @brief Copy a Fortran string into a c_char byte buffer (no terminator).
    !>
    !> Trailing blanks are stripped via len_trim so fixed-length actual
    !> arguments (which the language pads with spaces) marshal to their logical
    !> content. The byte count is reported through clen for the (bytes, length)
    !> C calling convention. An all-blank/empty key yields clen == 0, which the
    !> C bridge rejects as CONF_INVALID_ARG before dereferencing the buffer.
    subroutine f_to_c_bytes(fstr, cbuf, clen)
        character(len=*),                    intent(in)  :: fstr
        character(kind=c_char), allocatable, intent(out) :: cbuf(:)
        integer(c_int),                      intent(out) :: clen

        integer :: n

        n = len_trim(fstr)
        ! Allocate at least one element so the actual argument is always a valid
        ! associated buffer; the C side never reads it when clen < 1.
        allocate(cbuf(max(n, 1)))
        if (n > 0) cbuf(1:n) = transfer(fstr(1:n), cbuf(1:n))
        clen = int(n, c_int)
    end subroutine f_to_c_bytes

    !> @brief Build a default-kind Fortran string from a c_char byte buffer.
    !>
    !> Copies exactly n bytes, preserving raw byte content (a transfer-based
    !> bit copy, valid because c_char and default character share a one-byte
    !> storage unit on every supported compiler).
    pure function c_bytes_to_string(cbuf, n) result(str)
        character(kind=c_char), intent(in) :: cbuf(:)
        integer(c_int),         intent(in) :: n
        character(len=n)                   :: str

        if (n > 0_c_int) str = transfer(cbuf(1:n), str)
    end function c_bytes_to_string

end module conf_mod
