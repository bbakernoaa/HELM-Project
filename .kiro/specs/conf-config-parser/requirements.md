# Requirements Document

## Introduction

CONF (Configuration Object & Notation Framework) is a Tier 1b elective micro-library in the HELM ecosystem. It provides a thin, stateless, RAII-managed C++20 wrapper around a private YAML backend (yaml-cpp). CONF loads complex YAML 1.2 hierarchies (nested maps, sequences, scalars) from a file path or an in-memory string and exposes strongly typed accessors for deeply nested keys via dotted-path syntax such as `model.physics.layers`. It mirrors the proven HALO architecture: a clean C++ core, an `extern "C"` / `iso_c_binding` bridge for legacy Fortran models, an opaque thread-safe Handle_Registry, an exception-to-error-code translation macro, and a target-centric CMake build that produces the `HELM::CONF` alias.

These requirements are derived from the approved design document (`design.md`, internal reference `helm_conf_spec.md`). They are organized around the design's pillars: core parsing and path resolution, typed access, error handling, the Fortran C-API bridge, the standalone build system, and the testing/correctness strategy. Each requirement is written so that the design's six correctness properties can later cite the specific acceptance criteria they validate.

Scope is framed by the design's stated Non-Goals: CONF is read-only in Phase A (no YAML serialization or write-back to disk), performs no schema validation beyond type-cast checking, and depends on neither Kokkos nor MPI. CONF includes zero HELM headers, upholding the No-Circular-Dependency law. The HELM Laws of Physics that do not apply to a serial, node-local configuration parser (Zero-Copy `mdspan`, Kokkos hardware portability) are intentionally out of scope; the RAII and No-Circular-Dependency laws apply and are captured below.

## Glossary

- **CONF**: Configuration Object & Notation Framework; the Tier 1b stateless, RAII-managed YAML configuration micro-library within the HELM ecosystem.
- **Config**: The RAII C++ class that owns a single parsed YAML document and serves as the entry point for all queries. Move-only; each instance owns an independent node tree.
- **Value**: A lightweight, non-owning typed view over a single resolved node, returned by `Config::at`. Its validity is tied to the parent Config's lifetime.
- **Node_Kind**: The enumeration describing the kind of a resolved node: `Undefined`, `Null`, `Scalar`, `Sequence`, or `Map`.
- **Error_Code**: The stable integer enumeration of CONF failure modes, shared by the C++ core (as a `Conf_Error` payload) and the C bridge (as the returned `int`). Values are part of the C ABI and are append-only.
- **Conf_Error**: The C++ exception type (derived from `std::runtime_error`) that carries one Error_Code value plus a human-readable message; thrown by the core and translated at the C boundary.
- **Dotted_Path**: A non-empty, `.`-delimited key string. Each segment indexes a map node by key or a sequence node by base-10 integer index (for example `model.physics.layers` or `grid.resolution.0`).
- **Dotted_Path_Resolver**: The internal algorithm that walks the parsed node tree one Dotted_Path segment at a time, returning a defined node or raising a Conf_Error. It is total: it never crashes on any input.
- **YAML_Backend**: The private `conf::detail` layer (the `Yaml_Tree` wrapper) that is the only place yaml-cpp is included; compiled into the library but never installed or exposed.
- **CONF_C_Bridge**: The set of `extern "C"` functions (in `conf_c_interop.cpp`) that expose a flat C API of integer error codes and opaque integer handles for Fortran consumption.
- **CONF_C_TRY**: The macro that wraps each C_Bridge function body, translating any C++ exception into an int Error_Code so that no exception crosses into Fortran.
- **Handle_Registry**: The thread-safe singleton mapping monotonically increasing positive integer tokens to `Config*` pointers, so Fortran references C++ objects without seeing raw pointers.
- **Opaque_Handle**: The `integer(c_int)` token exposed to Fortran that maps internally to a Config via the Handle_Registry. Token `0` (`CONF_HANDLE_INVALID`) is the reserved invalid sentinel.
- **conf_mod**: The Fortran module providing the public Fortran API, implemented with iso_c_binding wrappers around the CONF_C_Bridge.
- **iso_c_binding**: The Fortran 2003+ intrinsic module providing interoperability types and attributes for calling C functions from Fortran.
- **RAII**: Resource Acquisition Is Initialization; the C++ idiom binding resource lifetime to object scope. Config's destructor frees the entire backend node tree.
- **FetchContent**: The CMake module used to fetch and build a pinned yaml-cpp privately at configure/build time.
- **Spack**: The HPC package manager whose installed yaml-cpp can be discovered via `find_package` as a fallback to FetchContent.
- **yaml-cpp**: The pinned (0.8.0) third-party YAML parser used as CONF's private implementation detail, linked PRIVATE and never exposed in the public interface.
- **HELM::CONF**: The exported CMake namespaced alias for the `conf` library target.
- **BUILD_TESTING**: The CMake option (default OFF) that, when ON, builds the GTest + RapidCheck test suite.
- **BUILD_FORTRAN**: The CMake option that, when ON, enables the Fortran language and compiles the conf_mod interop layer.
- **Google Test**: The example-based unit test framework used for CONF's unit tests.
- **RapidCheck**: The property-based test framework used for CONF's property tests, matching the HALO convention.

## Requirements

### Requirement 1: YAML Loading from File and String

**User Story:** As a Fortran model developer, I want to load a YAML configuration from either a file path or an in-memory string, so that I can supply run-deck settings from disk or from a string assembled at runtime.

#### Acceptance Criteria

1. WHEN `Config::from_file` is called with a path to a readable file containing valid YAML 1.2 (parsable without a syntax error), THE Config SHALL parse the document and return a Config instance that owns the resulting node tree.
2. WHEN `Config::from_string` is called with text containing valid YAML 1.2 (parsable without a syntax error), THE Config SHALL parse the text and return a Config instance that owns the resulting node tree.
3. WHILE a Config instance exists after successful construction, THE Config SHALL answer all queries from the node tree captured at construction time, such that modifying the originating file or string afterward does not change any query result.
4. IF the path passed to `Config::from_file` does not exist or cannot be opened for any reason, including permission denial, THEN THE Config SHALL raise `Conf_Error` carrying `Error_Code::File_Not_Found` without distinguishing the underlying cause and SHALL NOT return a Config instance.
5. IF the source supplied to `Config::from_file` or `Config::from_string` is not valid YAML, THEN THE Config SHALL raise `Conf_Error` carrying `Error_Code::Parse_Error` and SHALL NOT return a Config instance.
6. WHEN the source supplied to `Config::from_file` or `Config::from_string` is empty or contains only whitespace or comments, THE Config SHALL treat it as a valid empty document whose root node has kind `Null` or `Undefined`, rather than raise `Error_Code::Parse_Error`.

### Requirement 2: Hierarchical Document Representation

**User Story:** As a domain scientist, I want CONF to represent the full YAML hierarchy of nested maps, sequences, and scalars, so that I can store and retrieve arbitrarily structured model configuration.

#### Acceptance Criteria

1. THE Config SHALL represent each node of a parsed YAML document as exactly one `Node_Kind`: `Map`, `Sequence`, `Scalar`, `Null`, or `Undefined`.
2. THE Config SHALL associate each child of a map node with the literal key string under which it appears, preserving leading, trailing, and internal whitespace and applying no trimming, case-folding, or normalization.
3. THE Config SHALL preserve the order of a sequence node's child nodes as they appear in the document and SHALL index them by contiguous base-10 integers from 0 to the child count minus 1.
4. THE Config SHALL preserve each scalar leaf value as its textual representation for on-demand conversion to int, double, bool, or string.
5. THE Config SHALL impose no artificial limit on nesting depth beyond what the parsed document itself contains.
6. THE Config SHALL distinguish a `Null` node (a key present with an explicitly null or empty value) from an `Undefined` node (a key that does not resolve) and from a `Scalar` node holding an empty string.

### Requirement 3: Dotted-Path Key Resolution

**User Story:** As a domain scientist, I want to address deeply nested configuration values with a single dotted-path key, so that I can retrieve values like `model.physics.layers` or `grid.resolution.0` without manually traversing the hierarchy.

#### Acceptance Criteria

1. WHILE walking a dotted path, IF the current node is a map, THEN THE Dotted_Path_Resolver SHALL advance to the child whose key matches the current segment byte-for-byte (case-sensitive, no trimming or normalization), or raise `Error_Code::Key_Not_Found` if no such key exists.
2. WHILE walking a dotted path, IF the current node is a sequence and the current segment consists solely of ASCII decimal digits 0–9 denoting an index in the inclusive range [0, child count minus 1], THEN THE Dotted_Path_Resolver SHALL advance to the child node at that index.
3. WHEN the Dotted_Path_Resolver consumes every segment of a path successfully, THE Dotted_Path_Resolver SHALL return the node reached by the full path, whose `Node_Kind` is one of `Null`, `Scalar`, `Sequence`, or `Map` and never `Undefined`.
4. WHEN the Dotted_Path_Resolver matches a map key, THE Dotted_Path_Resolver SHALL match the segment literally including any whitespace, with no trimming, case-folding, or normalization.
5. IF a map segment names a key that is absent, or a sequence segment is a non-digit string, an out-of-range index, or an arbitrarily large digit run exceeding the child count, or a non-final segment lands on a scalar or null node, THEN THE Dotted_Path_Resolver SHALL raise `Conf_Error` carrying `Error_Code::Key_Not_Found` without modifying the configuration tree.
6. IF the dotted path is the empty string or contains any empty segment (leading dot, trailing dot, or consecutive dots), THEN THE Dotted_Path_Resolver SHALL raise `Conf_Error` carrying `Error_Code::Invalid_Arg` before walking any node and without modifying the configuration tree.
7. THE Dotted_Path_Resolver SHALL terminate for every input string, including arbitrary length and arbitrary-magnitude digit segments, and SHALL NOT crash, abort, leak, or invoke undefined behavior on any input.

### Requirement 4: Stateless RAII Configuration Ownership

**User Story:** As an architect, I want each Config to independently own its parsed tree and release it deterministically, so that the library remains stateless and leak-free in accordance with the HELM RAII law.

#### Acceptance Criteria

1. THE Config SHALL own its parsed node tree through a single `unique_ptr` to a private implementation object and SHALL contain no raw `new`/`delete` expression in the `conf` namespace outside the private `conf::detail` backend.
2. WHEN a Config is destroyed, THE Config SHALL leave no memory associated with its parsed node tree allocated after destruction completes.
3. THE Config SHALL be move-constructible and move-assignable and SHALL NOT be copy-constructible or copy-assignable.
4. WHEN a Config is moved from, THE source Config SHALL be left in a valid empty state in which `has`, `is_map`, and `is_sequence` return false and `size` returns 0, and SHALL free nothing on destruction.
5. THE Config SHALL hold an independent node tree per instance, such that destroying or moving one Config has no effect on the ownership, validity, or query results of any other Config.
6. WHEN a Config is move-assigned into while already owning a node tree, THE Config SHALL release the previously owned tree before taking ownership of the moved-in tree.
7. WHEN a Config is move-assigned to itself, THE Config SHALL remain valid and SHALL NOT free, double-free, or corrupt its node tree.

### Requirement 5: Typed Scalar Accessors (Throwing Flavor)

**User Story:** As a domain scientist, I want throwing typed accessors for int, double, bool, and string values, so that I can read a required configuration value and fail loudly when it is missing or mistyped.

#### Acceptance Criteria

1. WHEN `get_int`, `get_double`, `get_bool`, or `get_string` is called with a dotted path that resolves to a scalar node whose textual value is a valid representation of the requested type, where every scalar node is a valid representation for `get_string`, THE Config SHALL return the value converted to the requested type.
2. IF the dotted path passed to a throwing scalar accessor names a missing map key, an out-of-range or non-integer sequence index, or a non-final segment that lands on a scalar node, THEN THE Config SHALL raise `Conf_Error` carrying `Error_Code::Key_Not_Found`.
3. IF the dotted path passed to a throwing scalar accessor resolves to a map or sequence node rather than a scalar, or to a scalar node whose textual value is not a valid representation of the requested type, THEN THE Config SHALL raise `Conf_Error` carrying `Error_Code::Type_Mismatch`.
4. IF the dotted path passed to a throwing scalar accessor is the empty string or contains a leading dot, a trailing dot, or consecutive dots, THEN THE Config SHALL raise `Conf_Error` carrying `Error_Code::Invalid_Arg`.

### Requirement 6: Typed Scalar Accessors (Non-Throwing Flavor)

**User Story:** As a domain scientist, I want non-throwing typed accessors that return an optional value, so that I can probe for configuration values without exception-handling control flow.

#### Acceptance Criteria

1. WHEN `try_int`, `try_double`, `try_bool`, or `try_string` is called with a dotted path that resolves to a scalar node whose text converts to the requested type, THE Config SHALL return an engaged optional of the requested type containing the same converted value that the corresponding throwing accessor (`get_int`, `get_double`, `get_bool`, or `get_string`) would return for that path.
2. IF the dotted path passed to a non-throwing scalar accessor does not resolve, because a map segment names an absent key, a sequence segment is a non-integer or an out-of-range index, a non-final segment lands on a scalar, or the path is the empty string or contains a leading dot, trailing dot, or consecutive dots, THEN THE Config SHALL return an empty optional.
3. IF the dotted path passed to a non-throwing scalar accessor resolves to a defined node that does not convert to the requested type, including any map or sequence node, or a scalar whose text cannot be parsed as the requested type, THEN THE Config SHALL return an empty optional.
4. THE non-throwing scalar accessors SHALL be marked `noexcept` and SHALL NOT propagate any exception to the caller for any input path string or document state.

### Requirement 7: Defaulted Accessor

**User Story:** As a domain scientist, I want a defaulted accessor that returns a supplied fallback when a value is missing or mistyped, so that optional settings collapse to sensible defaults without extra branching.

#### Acceptance Criteria

1. WHEN `get_or` is called with a dotted path that resolves to a scalar convertible to the fallback value's type (one of int, double, bool, or string), THE Config SHALL return the converted value.
2. IF the dotted path passed to `get_or` does not resolve, THEN THE Config SHALL return the supplied fallback value.
3. IF the dotted path passed to `get_or` resolves to a node that cannot convert to the fallback value's type, THEN THE Config SHALL return the supplied fallback value.
4. IF the dotted path passed to `get_or` is the empty string or contains a leading dot, trailing dot, or consecutive dots, THEN THE Config SHALL return the supplied fallback value.
5. THE `get_or` accessor SHALL be marked `noexcept` and SHALL NOT propagate any exception to the caller for any input.

### Requirement 8: List Accessors

**User Story:** As a domain scientist, I want list accessors that return a typed vector for a sequence node, so that I can read array-valued configuration such as grid resolutions or layer thicknesses in one call.

#### Acceptance Criteria

1. WHEN `get_int_list`, `get_double_list`, or `get_string_list` is called with a well-formed dotted path that resolves to a sequence whose every element converts to the requested element type, THE Config SHALL return a vector containing exactly one converted element per sequence element in document order, and SHALL return an empty vector when the resolved sequence has zero elements.
2. IF a well-formed dotted path passed to a list accessor resolves to no defined node because of a missing map key, an out-of-range sequence index, or descending past a scalar, THEN THE Config SHALL raise `Conf_Error` carrying `Error_Code::Key_Not_Found`.
3. IF the dotted path passed to a list accessor resolves to a node that is not a sequence, or to a sequence containing at least one element that cannot convert to the requested element type, THEN THE Config SHALL raise `Conf_Error` carrying `Error_Code::Type_Mismatch` and SHALL NOT return a partial or empty vector.
4. IF the dotted path passed to a list accessor is the empty string or contains a leading dot, a trailing dot, or consecutive dots, THEN THE Config SHALL raise `Conf_Error` carrying `Error_Code::Invalid_Arg`.

### Requirement 9: Structure Introspection

**User Story:** As a domain scientist, I want to query whether a key exists and what kind and size of node it is, so that I can branch on configuration shape without triggering errors.

#### Acceptance Criteria

1. WHEN `has` is called with a dotted path that resolves to a defined node (`Node_Kind` of `Null`, `Scalar`, `Sequence`, or `Map`), THE Config SHALL return true.
2. IF `has` is called with a dotted path that does not resolve, or is the empty string or contains a leading dot, trailing dot, or consecutive dots, THEN THE Config SHALL return false.
3. WHEN `is_map` is called, THE Config SHALL return true if the dotted path resolves to a map node and SHALL return false for any non-map defined node, a non-resolving path, or a malformed path.
4. WHEN `is_sequence` is called, THE Config SHALL return true if the dotted path resolves to a sequence node and SHALL return false for any non-sequence defined node, a non-resolving path, or a malformed path.
5. WHEN `size` is called with a dotted path that resolves to a map or sequence node, THE Config SHALL return the child count of that node, counting map key/value entries or sequence elements.
6. WHEN `size` is called with a dotted path that resolves to a `Scalar` or `Null` node, does not resolve, or is malformed, THE Config SHALL return 0.
7. THE introspection accessors `has`, `is_map`, `is_sequence`, and `size` SHALL be marked `noexcept` and SHALL NOT propagate any exception to the caller for any input dotted-path string or document state.

### Requirement 10: Resolved Node View (conf::Value)

**User Story:** As a domain scientist, I want a lightweight typed view of a resolved node, so that I can introspect a node's kind and convert it without re-resolving the dotted path.

#### Acceptance Criteria

1. WHEN `Config::at` is called with a dotted path that resolves to a defined node, THE Config SHALL return a Value that views that node.
2. IF the dotted path passed to `Config::at` names a missing map key, an out-of-range or non-integer sequence index, or a non-final segment that lands on a scalar, THEN THE Config SHALL raise `Conf_Error` carrying `Error_Code::Key_Not_Found`.
3. IF the dotted path passed to `Config::at` is the empty string or contains a leading dot, trailing dot, or consecutive dots, THEN THE Config SHALL raise `Conf_Error` carrying `Error_Code::Invalid_Arg`.
4. WHEN `Value::kind` is called, THE Value SHALL report the node's kind as exactly one of `Undefined`, `Null`, `Scalar`, `Sequence`, or `Map`.
5. WHEN `Value::size` is called on a map or sequence node, THE Value SHALL return that node's child count, and WHEN called on a `Scalar`, `Null`, or `Undefined` node SHALL return 0.
6. WHEN a throwing Value conversion (`as_int`, `as_double`, `as_bool`, `as_string`) is called on a scalar node whose text is a valid representation of the requested type, where every scalar is a valid representation for `as_string`, THE Value SHALL return the converted value.
7. IF a throwing Value conversion is called on a non-scalar node or a scalar whose text is not a valid representation of the requested type, THEN THE Value SHALL raise `Conf_Error` carrying `Error_Code::Type_Mismatch`.
8. WHEN a non-throwing Value conversion (`try_int`, `try_double`, `try_bool`, `try_string`) is called, THE Value SHALL return an engaged optional containing the same value the corresponding throwing conversion would return on success and an empty optional on a failed conversion, and SHALL be `noexcept`.
9. THE Value SHALL be a non-owning view that neither owns nor copies the parent Config's node tree and SHALL remain valid only while the parent Config is alive.

### Requirement 11: Error Taxonomy and Lockstep Error Model

**User Story:** As a Fortran model developer, I want a single stable error taxonomy shared between the C++ exception and the C return code, so that the codes my Fortran sees never drift from the C++ core's meaning.

#### Acceptance Criteria

1. THE Error_Code enumeration SHALL define exactly these nine values and no others: `Success` (0), `Invalid_Arg` (1), `File_Not_Found` (2), `Parse_Error` (3), `Key_Not_Found` (4), `Type_Mismatch` (5), `Bad_Handle` (6), `Buffer_Too_Small` (7), and `Unknown` (99).
2. THE Error_Code enumeration SHALL retain each named enumerator's integer value across versions, adding new failure modes only as new enumerators with previously unused values, and SHALL NOT renumber, reassign, or remove any existing enumerator.
3. THE Conf_Error exception SHALL carry exactly one Error_Code value and a non-empty diagnostic message, and SHALL expose through a `code()` accessor the identical Error_Code value supplied at construction.
4. IF a CONF operation fails, THEN THE Conf_Error raised by the C++ core SHALL carry the same Error_Code value that the corresponding CONF_C_Bridge function returns as its `int` result for that failure, and THE CONF_C_Bridge SHALL return that non-zero Error_Code rather than `Error_Code::Success`.
5. THE conf_mod module SHALL declare exactly one named integer parameter per Error_Code enumerator (nine total), each equal to the integer value of its corresponding enumerator.

### Requirement 12: Never Crash on Bad Input

**User Story:** As a Fortran model developer, I want CONF to convert every bad input into a typed error rather than crashing, so that a malformed run deck cannot take down the model process.

#### Acceptance Criteria

1. IF any CONF input is malformed, including an unreadable or nonexistent file path, malformed YAML, a missing key, a type mismatch, a malformed dotted path, an invalid or released Opaque_Handle, or a buffer smaller than the required byte count, THEN THE relevant CONF component SHALL terminate the single call in bounded time and report the corresponding Error_Code rather than crash, abort, hang, leak, or invoke undefined behavior, for input of any length or content including empty input and non-UTF-8 bytes.
2. THE non-throwing C++ accessors SHALL return, for any failing input, an empty optional (`try_*`), the supplied fallback (`get_or`), false (`has`, `is_map`, `is_sequence`), or 0 (`size`) as appropriate to the accessor, rather than propagate an exception.
3. IF a CONF_C_Bridge function is given a null required pointer, THEN it SHALL return `Error_Code::Invalid_Arg`; IF it is given an invalid or released handle, THEN it SHALL return `Error_Code::Bad_Handle` and SHALL NOT dereference any Config object; and in all failing cases it SHALL return a non-zero Error_Code rather than allow a C++ exception to escape into the caller.
4. WHEN a CONF operation has failed, THE Config, Handle_Registry, and CONF_C_Bridge SHALL remain in a valid state usable for subsequent valid calls.

### Requirement 13: Missing Key Error Handling

**User Story:** As a domain scientist, I want missing keys reported consistently across all access flavors, so that absent settings are easy to detect and default.

#### Acceptance Criteria

1. IF a dotted-path key does not resolve due to a missing map key, an out-of-range or non-integer sequence index, or descending past a scalar or null node, THEN THE throwing accessors (`get_int`, `get_double`, `get_bool`, `get_string`, the list accessors, and `Config::at`) SHALL raise `Conf_Error` carrying `Error_Code::Key_Not_Found` without modifying the configuration tree.
2. IF a dotted-path key does not resolve, THEN THE non-throwing accessors (`try_int`, `try_double`, `try_bool`, `try_string`) SHALL return an empty optional, `get_or` SHALL return its fallback, and `has` SHALL return false.
3. IF a dotted-path key does not resolve, THEN THE corresponding CONF_C_Bridge getter SHALL return `Error_Code::Key_Not_Found`.

### Requirement 14: Malformed YAML Error Handling

**User Story:** As a Fortran model developer, I want malformed YAML reported as a parse error without aborting, so that I can correct a broken config file and reload.

#### Acceptance Criteria

1. IF a file or string supplied to a Config factory contains a YAML syntax error, THEN THE Config SHALL raise `Conf_Error` carrying `Error_Code::Parse_Error`.
2. WHEN a parse failure occurs, THE Conf_Error message SHALL contain the non-empty diagnostic text reported by the YAML_Backend.
3. IF a parse failure occurs at the CONF_C_Bridge load functions, THEN THE bridge SHALL return `Error_Code::Parse_Error`, SHALL NOT register a handle for the failed load, and SHALL NOT abort the process.

### Requirement 15: Type Mismatch Error Handling

**User Story:** As a domain scientist, I want type mismatches reported distinctly from missing keys, so that I can tell a wrong-typed value apart from an absent one.

#### Acceptance Criteria

1. IF a resolved node is a map or sequence rather than a scalar, or is a scalar whose text is not a valid representation of the requested scalar type, THEN THE throwing accessors SHALL raise `Conf_Error` carrying `Error_Code::Type_Mismatch`.
2. IF a resolved node cannot convert to the requested scalar type as described in criterion 1, THEN THE non-throwing accessors SHALL return an empty optional and `get_or` SHALL return its fallback.
3. IF a resolved node cannot convert to the requested type, THEN THE corresponding CONF_C_Bridge getter SHALL return `Error_Code::Type_Mismatch`.
4. WHEN the same node both fails to resolve and would mismatch, THE accessor SHALL report `Error_Code::Key_Not_Found` for the non-resolution and `Error_Code::Type_Mismatch` only for a resolved-but-unconvertible node, keeping the two conditions distinct.

### Requirement 16: Malformed Path Error Handling

**User Story:** As a domain scientist, I want malformed dotted paths reported as an argument error, so that empty or ill-formed keys are caught before traversal.

#### Acceptance Criteria

1. IF a dotted path is the empty string or contains a leading dot, trailing dot, or consecutive dots, THEN THE throwing accessors and `Config::at` SHALL raise `Conf_Error` carrying `Error_Code::Invalid_Arg`, checked before any node traversal and in preference to `Key_Not_Found` or `Type_Mismatch`.
2. IF a dotted path is malformed as described in criterion 1, THEN THE non-throwing accessors SHALL return an empty optional, `get_or` SHALL return its fallback, and the introspection accessors SHALL return false or 0.
3. IF a dotted path is malformed as described in criterion 1, THEN the corresponding CONF_C_Bridge function SHALL return `Error_Code::Invalid_Arg`.

### Requirement 17: extern "C" Error-Code API Surface

**User Story:** As a legacy Fortran model developer, I want a flat C API of integer-returning functions over opaque handles, so that I can drive CONF from Fortran without seeing C++ types or pointers.

#### Acceptance Criteria

1. THE CONF_C_Bridge SHALL expose `extern "C"` functions covering the lifecycle operations (`conf_load_c`, `conf_load_string_c`, `conf_close_c`), existence and structure operations (`conf_has_key_c`, `conf_size_c`), and the scalar and string getters (`conf_get_int_c`, `conf_get_double_c`, `conf_get_bool_c`, `conf_get_string_len_c`, `conf_get_string_c`).
2. THE CONF_C_Bridge functions SHALL return an `int` whose value is an Error_Code, where `Error_Code::Success` (0) indicates success and any non-zero value indicates a specific failure.
3. THE CONF_C_Bridge functions SHALL accept input strings as a `(const char*, int len)` pair rather than relying on null termination, and SHALL return output values through caller-supplied pointers.
4. WHEN `conf_get_bool_c` returns successfully, THE CONF_C_Bridge SHALL write 1 for a true value and 0 for a false value through the output pointer.
5. IF a required pointer argument (`handle_out`, `key`, `path`, `yaml_text`, a numeric output pointer, `str_len_out`, or `written_out`) is null, or a `path_len`, `text_len`, or `key_len` is less than 1, THEN THE CONF_C_Bridge function SHALL return `Error_Code::Invalid_Arg`; a string-buffer capacity of 0 SHALL NOT by itself be treated as Invalid_Arg.
6. WHEN `conf_has_key_c` is called with a key that does not resolve, THE CONF_C_Bridge SHALL set the existence output to 0 and return `Error_Code::Success`, and WHEN `conf_size_c` is called with a key that resolves to a scalar/null node or does not resolve, THE CONF_C_Bridge SHALL set the size output to 0 and return `Error_Code::Success`.
7. IF any handle-taking CONF_C_Bridge function is called with token 0 or a token that is not currently registered (never issued or already released), THEN THE function SHALL return `Error_Code::Bad_Handle` and SHALL NOT dereference any Config object.
8. IF a CONF_C_Bridge function returns a non-zero Error_Code, THEN it SHALL leave all caller-supplied output arguments unchanged.

### Requirement 18: Opaque Handle Registry

**User Story:** As a legacy Fortran model developer, I want CONF objects referenced by opaque integer tokens, so that I can manage configuration lifetime from Fortran without holding C++ pointers.

#### Acceptance Criteria

1. WHEN the Handle_Registry registers a Config pointer, THE Handle_Registry SHALL return a unique integer token greater than 0.
2. THE Handle_Registry SHALL reserve token `0` (`CONF_HANDLE_INVALID`) as the invalid sentinel and SHALL never issue it as a valid token.
3. WHEN the Handle_Registry looks up a currently registered token, THE Handle_Registry SHALL return the associated Config pointer, and for any unregistered or released token SHALL return a null pointer.
4. WHEN the Handle_Registry releases a currently registered token, THE Handle_Registry SHALL remove the mapping, return the previously associated pointer, and treat that token as permanently invalid for every subsequent lookup and release.
5. THE Handle_Registry SHALL issue distinct tokens across the lifetime of a run and SHALL NOT reuse a previously issued token, including released tokens.
6. WHILE register, lookup, and release calls execute concurrently under an `MPI_THREAD_MULTIPLE`-equivalent multi-threaded environment, THE Handle_Registry SHALL guard its internal state with a mutex such that it still issues distinct tokens, returns the correct pointer or null on lookup, and does not crash, deadlock, corrupt state, or invoke undefined behavior.
7. WHEN the Handle_Registry is asked to release token 0, an unissued token, or an already-released token, THE Handle_Registry SHALL perform no mapping change and SHALL return a null pointer.

### Requirement 19: Exception-to-Error-Code Translation (CONF_C_TRY)

**User Story:** As an architect, I want every C bridge function to translate C++ exceptions into error codes, so that no exception can cross into Fortran where it would be undefined behavior.

#### Acceptance Criteria

1. WHEN a CONF_C_Bridge function body completes without throwing, THE CONF_C_TRY macro SHALL return the `int` value of `Error_Code::Success` (0).
2. WHEN a `Conf_Error` is thrown within a CONF_C_Bridge function body, THE CONF_C_TRY macro SHALL return the `int` value of that exception's carried Error_Code.
3. WHEN a `std::invalid_argument` is thrown within a CONF_C_Bridge function body, THE CONF_C_TRY macro SHALL return the `int` value of `Error_Code::Invalid_Arg`.
4. WHEN any exception other than `Conf_Error` or `std::invalid_argument` is thrown within a CONF_C_Bridge function body, including `std::bad_alloc`, any other `std::exception`, or a non-standard catch-all throw, THE CONF_C_TRY macro SHALL return the `int` value of `Error_Code::Unknown`.
5. THE CONF_C_TRY macro SHALL catch every exception type so that every `extern "C"` function always returns an `int` Error_Code and no C++ exception escapes into the Fortran caller.

### Requirement 20: Length-First String Marshalling Contract

**User Story:** As a legacy Fortran model developer, I want string values delivered into a buffer I allocate, so that no heap memory ownership crosses the language boundary and nothing leaks.

#### Acceptance Criteria

1. WHEN `conf_get_string_len_c` is called for a key that resolves to a string-convertible value, THE CONF_C_Bridge SHALL report through `str_len_out` the exact byte count the value occupies, excluding any null terminator, and this count SHALL equal the byte count `conf_get_string_c` enforces for the same handle, key, and unchanged document.
2. WHEN `conf_get_string_c` is called with a buffer whose capacity is at least the required byte count, THE CONF_C_Bridge SHALL copy exactly that many of the value's bytes into the caller-allocated buffer and report that count through `written_out`.
3. IF the buffer capacity passed to `conf_get_string_c` is smaller than the required byte count, THEN THE CONF_C_Bridge SHALL write nothing into the buffer, leave `written_out` unchanged, and return `Error_Code::Buffer_Too_Small`.
4. WHEN `conf_get_string_c` is called for a value whose required byte count is 0 and the buffer capacity is greater than or equal to 0, THE CONF_C_Bridge SHALL write nothing, report 0 through `written_out`, and return `Error_Code::Success`.
5. THE CONF_C_Bridge SHALL NOT return ownership of any heap memory to the caller; every `char*` crossing the boundary SHALL be a caller-allocated and caller-owned buffer into which CONF only copies bytes.
6. THE CONF_C_Bridge SHALL NOT null-terminate strings at the boundary; the written byte count SHALL be the authoritative logical length.
7. THE CONF_C_Bridge SHALL NOT retain any global or static string buffer between calls, and SHALL perform zero heap allocations that outlive a single bridge call.

### Requirement 21: Bad Handle Error Handling

**User Story:** As a legacy Fortran model developer, I want stale or invalid handles rejected safely, so that a released or bogus token cannot corrupt memory.

#### Acceptance Criteria

1. IF a CONF_C_Bridge function that requires a handle is passed a token that is not currently registered (never issued, already released, or the invalid sentinel 0), THEN THE CONF_C_Bridge SHALL return `Error_Code::Bad_Handle`, SHALL NOT dereference or destroy any Config object, and SHALL leave the Handle_Registry unchanged.
2. WHEN `conf_close_c` is called with a currently registered handle, THE CONF_C_Bridge SHALL release the handle, destroy the associated Config so that its node tree is freed, and return `Error_Code::Success`.
3. IF `conf_close_c` is called with a token that is not currently registered, THEN THE CONF_C_Bridge SHALL return `Error_Code::Bad_Handle` and SHALL leave the Handle_Registry unchanged.
4. WHEN a handle has been successfully closed by `conf_close_c`, THE CONF_C_Bridge SHALL return `Error_Code::Bad_Handle` for that same token on every subsequent call that requires a handle.

### Requirement 22: Fortran conf_mod Wrapper Module

**User Story:** As a legacy Fortran model developer, I want an idiomatic Fortran module wrapping the C bridge, so that I can load configs and query values using native Fortran types without managing the length-first string protocol myself.

#### Acceptance Criteria

1. THE conf_mod module SHALL declare iso_c_binding interface blocks with the `bind(c)` attribute for each CONF_C_Bridge function it wraps.
2. THE conf_mod module SHALL declare exactly nine public integer parameters, one per Error_Code value, and each parameter SHALL equal the integer value of the corresponding Error_Code enumerator.
3. WHEN a conf_mod wrapper invokes a CONF_C_Bridge function with a Fortran `character(len=*)` argument, THE conf_mod wrapper SHALL pass the argument together with its length so that the C bridge receives the correct byte count.
4. WHEN `conf_get_string` is called for a key that resolves to a string-convertible value, THE conf_mod wrapper SHALL perform the length-first protocol internally by querying the length, allocating an `allocatable` character of exactly that size, copying the value, and returning the allocated string, so that the caller performs no manual allocation or free.
5. WHEN a conf_mod wrapper invokes a CONF_C_Bridge function, THE conf_mod wrapper SHALL return the resulting Error_Code to the caller through an `intent(out)` status argument so that the caller can detect success or the specific failure.
6. IF `conf_get_string` is called for a key that does not resolve or whose value is not string-convertible, THEN THE conf_mod wrapper SHALL return a zero-length string and a non-Success status without performing a partial allocation that the caller must free.
7. THE conf_mod module SHALL store each Opaque_Handle returned by the C bridge as an `integer(c_int)` and pass it back unchanged to subsequent C bridge calls.
8. THE conf_mod module and the CONF_C_Bridge SHALL compile and link without error with Fortran 2008-compliant compilers including gfortran 9 or later, Intel ifort/ifx, and NVIDIA nvfortran.

### Requirement 23: Standalone Target-Centric CMake Build

**User Story:** As a build engineer, I want CONF to build as a standalone target-centric CMake project producing HELM::CONF, so that the library can be built and consumed independently within the HELM monorepo.

#### Acceptance Criteria

1. THE CONF build system SHALL require CMake version 3.21 or later and SHALL fail configuration with an error indicating an unsupported CMake version when an earlier CMake is used.
2. THE CONF build system SHALL require the C++20 standard as a non-optional requirement (standard required, not a best-effort preference) with compiler-specific language extensions disabled.
3. THE CONF build system SHALL define a library target named `conf` and a namespaced alias `HELM::CONF` for that target.
4. THE CONF build system SHALL expose the public `include/` directory as a PUBLIC usage requirement of the `conf` target and SHALL keep the `src/` directory (including `detail/` headers) as a PRIVATE include path that is never installed.
5. THE CONF build system SHALL install the `conf` target and export it under the `HELM::` namespace, generating `CONFConfig.cmake`, `CONFConfigVersion.cmake`, and `CONFTargets.cmake` with `SameMajorVersion` compatibility.
6. THE CONF build system SHALL install only the public headers under `include/conf/` and SHALL NOT install any `detail/` header.

### Requirement 24: Private YAML Backend Acquisition

**User Story:** As a build engineer, I want yaml-cpp fetched privately with a Spack fallback and hidden from consumers, so that the YAML dependency never leaks into the public interface or package config of HELM::CONF.

#### Acceptance Criteria

1. WHEN `CONF_USE_SYSTEM_YAMLCPP` is ON and a yaml-cpp of version 0.8 or a higher release that is `SameMajorVersion`-compatible with 0.8 is discoverable on the CMake package search path, THE CONF build system SHALL locate it via `find_package(yaml-cpp 0.8)` and link the `conf` target against the discovered target.
2. IF `CONF_USE_SYSTEM_YAMLCPP` is OFF, or it is ON but no yaml-cpp meeting the version condition in criterion 1 is found, THEN THE CONF build system SHALL fetch yaml-cpp via FetchContent pinned to the exact version 0.8.0 and build it privately.
3. WHEN building privately via FetchContent, THE CONF build system SHALL disable yaml-cpp's own tests, tools, and install steps so that yaml-cpp is neither installed nor exported.
4. THE CONF build system SHALL link yaml-cpp to the `conf` target with PRIVATE visibility so that yaml-cpp headers and link flags do not propagate to consumers of HELM::CONF.
5. THE generated `CONFConfig.cmake` SHALL NOT call `find_dependency(yaml-cpp)`, so that a downstream `find_package(CONF)` pulls in zero yaml-cpp symbols, headers, or transitive find calls.

### Requirement 25: Build Options for Testing and Fortran

**User Story:** As a build engineer, I want options to toggle the test suite and the Fortran layer, so that I can build a minimal core library or a full test-and-interop build as needed.

#### Acceptance Criteria

1. THE CONF build system SHALL provide a `BUILD_TESTING` option defaulting to OFF.
2. WHEN `BUILD_TESTING` is ON, THE CONF build system SHALL enable testing and add the C++ test suite built against Google Test and RapidCheck.
3. THE CONF build system SHALL provide a `BUILD_FORTRAN` option defaulting to ON.
4. WHEN `BUILD_FORTRAN` is ON, THE CONF build system SHALL enable the Fortran language and compile the `conf_c_interop.cpp` bridge and the `conf_mod` module into the build.
5. WHERE both `BUILD_TESTING` and `BUILD_FORTRAN` are ON, THE CONF build system SHALL add the Fortran integration test target in addition to the C++ test suite.

### Requirement 26: Tier 1 Isolation and Dependency Compliance

**User Story:** As an architect, I want CONF to depend on no other HELM library and on neither Kokkos nor MPI, so that the No-Circular-Dependency law holds and CONF stays a self-contained elective utility.

#### Acceptance Criteria

1. THE CONF library SHALL NOT include any header from another HELM component (DAGR, SPAN, AXIS, HALO, TICK, LOGS, or AMIO) in any source file, public header, or internal header.
2. THE CONF library SHALL NOT link against Kokkos, MPI, or any other HELM library target.
3. THE CONF public headers SHALL include only C++ standard library headers and CONF's own public headers, and SHALL NOT expose any yaml-cpp type.
4. THE CONF CMakeLists.txt SHALL NOT reference any `HELM::` namespaced target other than `HELM::CONF`.

### Requirement 27: Round-Trip Type Fidelity

**User Story:** As a domain scientist, I want values I read back to equal the values that were written into the YAML, so that I can trust CONF to preserve typed configuration data.

#### Acceptance Criteria

1. WHEN an integer value within the 32-bit two's-complement range is serialized to YAML, parsed by a Config, and read back with `get_int`, THE Config SHALL return a value exactly equal to the original.
2. WHEN a finite double value is serialized to YAML with at least 17 significant decimal digits, parsed by a Config, and read back with `get_double`, THE Config SHALL return a value bit-for-bit equal to the original with zero tolerance.
3. WHEN a boolean value is serialized to YAML, parsed by a Config, and read back with `get_bool`, THE Config SHALL return a value exactly equal to the original.
4. WHEN a string value of 0 to 65,535 bytes is serialized to YAML, parsed by a Config, and read back with `get_string`, THE Config SHALL return a value byte-for-byte identical to the original, preserving all internal whitespace.
5. WHEN a map of typed key/value pairs is serialized to YAML and parsed by a Config, THE Config SHALL return, for each key queried with the matching typed accessor, a value equal to the original value.

### Requirement 28: Type-Safety Across Incompatible Requests

**User Story:** As a domain scientist, I want incompatible type requests to fail cleanly rather than return garbage, so that a wrong-typed read never silently corrupts model state.

#### Acceptance Criteria

1. IF a node holding a value of one type (a map or sequence node, or a scalar whose text is not parseable as the requested numeric or boolean type, noting every scalar is valid for `get_string`) is requested through a throwing accessor (`get_int`, `get_double`, `get_bool`), THEN THE Config SHALL raise `Conf_Error` carrying `Error_Code::Type_Mismatch` and SHALL NOT return a value.
2. IF such an incompatible node is requested through a non-throwing accessor (`try_int`, `try_double`, `try_bool`), THEN THE Config SHALL return an empty optional and SHALL NOT return a fabricated value.
3. IF such an incompatible node is requested through `get_or`, THEN THE Config SHALL return the supplied fallback value.
4. WHEN a type-incompatible request has failed, THE Config SHALL remain valid for subsequent queries.

### Requirement 29: Resolver Totality Under Arbitrary Keys

**User Story:** As a Fortran model developer, I want any key string at all to produce a defined result or a clean miss, so that adversarial or accidental key strings can never crash the process.

#### Acceptance Criteria

1. WHEN a non-throwing accessor (`try_*`, `get_or`) is called with any arbitrary key string of 0 to 1,048,576 bytes, including empty, leading/trailing/consecutive dots, UTF-8, non-UTF-8 bytes, or an oversized digit run, THE Config SHALL return either an engaged optional (or value) or an empty optional (or the fallback) and SHALL NOT crash, abort, hang, leak, or invoke undefined behavior.
2. WHEN an introspection accessor (`has`, `is_map`, `is_sequence`, `size`) is called with any such arbitrary key string, THE Config SHALL return a defined boolean or count and SHALL NOT crash, abort, hang, leak, or invoke undefined behavior.
3. WHEN a throwing accessor or `Config::at` is called with any such arbitrary key string, THE Config SHALL either return a value or raise a typed `Conf_Error` and SHALL NOT crash, abort, hang, leak, or invoke undefined behavior.

### Requirement 30: No-Leak Guarantee on the C Bridge

**User Story:** As an architect, I want a sequence of bridge calls to leave no resident CONF heap allocation behind, so that long-running models that repeatedly load and query configs do not leak.

#### Acceptance Criteria

1. WHEN a sequence of CONF_C_Bridge calls beginning with a load and ending with `conf_close_c` completes, THE CONF library SHALL hold zero heap bytes allocated by CONF that outlive the sequence, as observed under a heap/leak sanitizer or allocation counter.
2. THE CONF_C_Bridge string operations SHALL allocate only caller-owned buffers and SHALL retain no CONF-owned allocation after each bridge call returns.
3. IF a load via the CONF_C_Bridge fails before a handle is registered, THEN THE CONF library SHALL hold zero heap bytes allocated by that failed load that outlive the call.

### Requirement 31: Handle Uniqueness and Non-Reuse Invariant

**User Story:** As an architect, I want every issued handle to be unique and released handles to stay invalid, so that stale-handle reuse bugs are structurally impossible.

#### Acceptance Criteria

1. WHEN any interleaving of `register_handle` and `release` operations is performed on the Handle_Registry, THE Handle_Registry SHALL ensure that every issued token is a distinct positive integer and that `0` is never issued.
2. WHEN a token has been released, THE Handle_Registry SHALL report that token as invalid (`lookup` returns null, `valid` returns false) for every subsequent lookup and SHALL never re-issue it.
3. WHEN a Config is registered, released, and another Config is subsequently registered, THE Handle_Registry SHALL issue a token distinct from the released one.

### Requirement 32: Unit Test Coverage for Mandated Behaviors

**User Story:** As a library developer, I want a Google Test suite proving loading, missing keys, malformed YAML, and type casting, so that the documented behaviors are verified by example-based tests.

#### Acceptance Criteria

1. THE test suite SHALL contain Google Test cases that load a valid file and a valid string (each asserting a Config is returned and a known query yields the expected value) and that confirm `Config::from_file` on a missing path raises `Conf_Error` whose `code()` equals `Error_Code::File_Not_Found`.
2. THE test suite SHALL contain Google Test cases that confirm malformed YAML (such as bad indentation or an unclosed bracket) raises `Conf_Error` whose `code()` equals `Error_Code::Parse_Error` without crashing or aborting.
3. THE test suite SHALL contain Google Test cases that confirm missing leaves, missing intermediate segments, out-of-range sequence indices, and descending past a scalar each raise `Conf_Error` whose `code()` equals `Error_Code::Key_Not_Found` for throwing accessors and yield a disengaged optional for non-throwing accessors.
4. THE test suite SHALL contain Google Test cases that confirm successful conversions for int, double, bool, and string return values equal to the expected fixture values and that incompatible conversions raise `Conf_Error` whose `code()` equals `Error_Code::Type_Mismatch`.
5. THE test suite SHALL contain Google Test cases that exercise the CONF_C_Bridge error paths (each returning its specific Error_Code) and the length-first string round trip, asserting `conf_get_string_len_c` reports N and `conf_get_string_c` writes N bytes and returns `Error_Code::Success`.
6. THE test suite SHALL contain a Google Test case that, given a buffer smaller than the required byte count, confirms `conf_get_string_c` writes nothing and returns `Error_Code::Buffer_Too_Small`.

### Requirement 33: Property-Based Test Coverage

**User Story:** As a library developer, I want property-based tests realizing the design's correctness properties, so that universal invariants are checked across many generated inputs.

#### Acceptance Criteria

1. THE test suite SHALL contain RapidCheck property tests realizing the round-trip property (read-back equals original), the resolver-totality property (any generated key yields an engaged or disengaged result with no undefined behavior), the no-leak C-bridge property (resident CONF heap bytes equal 0 after a load-query-close sequence), and the handle uniqueness-and-non-reuse property (issued tokens are distinct and released tokens are never re-issued).
2. WHEN a property test is executed, THE test suite SHALL run at least 100 generated iterations for that property.
3. THE property tests SHALL run single-process and SHALL require neither MPI nor `mpirun`.

### Requirement 34: Fortran Integration Test

**User Story:** As a legacy Fortran model developer, I want an end-to-end Fortran test through conf_mod, so that the full Fortran to C to yaml-cpp path including leak-free string marshalling is proven.

#### Acceptance Criteria

1. WHERE `BUILD_FORTRAN` and `BUILD_TESTING` are ON, THE test suite SHALL compile a Fortran program against `conf_mod` that loads a fixture YAML, queries an integer, a real, and a string, and asserts that each call's status equals `CONF_SUCCESS` (0) and that each returned value equals the corresponding fixture value.
2. WHEN the Fortran integration test queries a string through `conf_get_string`, THE test SHALL exercise the allocatable wrapper so that the returned string length equals the value's byte count and its contents match the fixture, without any manual allocation or free by the test.
3. WHEN the Fortran integration test finishes its queries, THE test SHALL close the handle via the conf_mod wrapper and assert a `CONF_SUCCESS` status.
