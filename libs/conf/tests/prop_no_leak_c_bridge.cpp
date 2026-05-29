// ─── Property-Based Tests: CONF C-Bridge No-Leak Invariant ───────────────────
// Feature: conf-config-parser, Property 4: No-Leak on the C Bridge
//
// Verifies that for ANY generated `load -> N queries -> close` sequence (and for
// failed loads that register no handle), the CONF library holds zero resident
// heap bytes that outlive the sequence:
//
//   ∀ ops ending in conf_close_c :
//       conf_resident_heap_after(ops) == conf_resident_heap_before(ops)
//
// ── How the invariant is observed (allocation counter) ───────────────────────
// This translation unit replaces the *global, non-over-aligned* operator
// new / operator new[] / operator delete / operator delete[] (throwing, sized,
// and nothrow forms) so that every heap (de)allocation routed through
// ::operator new is reflected in a single process-wide atomic counter of LIVE
// allocations (incremented on new, decremented on delete). Because operator
// new/delete are replaceable functions with external linkage, this single
// definition is used program-wide — including inside the statically-linked
// `conf` library and its private, statically-absorbed yaml-cpp — so CONF's own
// allocations are counted.
//
// We deliberately DO NOT override the over-aligned (std::align_val_t) forms:
// std::malloc cannot guarantee over-alignment, and the aligned new/delete pair
// is self-consistent on the default path, so leaving it alone keeps the counter
// balanced without risking misaligned storage.
//
// ── "Prime then measure" to neutralize one-time lazy statics ─────────────────
// yaml-cpp (and the C++ runtime) lazily construct process-wide statics — regex
// caches, locale facets, folded/block-scalar machinery — the first time a given
// parse code path executes. Those allocations are intentionally permanent and
// are NOT a per-sequence leak, but a naive snapshot would mis-attribute them to
// whichever sequence happened to trigger them first.
//
// To isolate genuine per-sequence leaks, each property runs the EXACT generated
// sequence TWICE: once unmeasured (to trigger any input-specific lazy statics),
// then again inside the measured snapshot window. The asserted quantity is the
// DELTA (live_after - live_before) of the SECOND run. A real per-call leak (e.g.
// a Config that conf_close_c failed to free) recurs identically on the second
// run and is still caught; only truly one-time initializations — already
// triggered by the first run — net to zero. Transient framework allocations are
// performed OUTSIDE the measured window. The same sequences are additionally
// clean under ASan/LSan when built with -fsanitize=address.
//
// The counting method is portable and deterministic.
//
// **Validates: Requirements 30.1, 30.2, 30.3**
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <new>
#include <string>
#include <utility>
#include <vector>

// ─── Global allocation counter (live = #new - #delete) ───────────────────────
namespace {

/// Process-wide count of currently-live ::operator new allocations.
/// std::atomic<long long> has a constexpr constructor, so this is
/// constant-initialized before any dynamic initialization runs — it is valid
/// from the very first allocation the program performs.
std::atomic<long long> g_live_allocations{0};

}  // namespace

// Throwing forms ──────────────────────────────────────────────────────────────
void* operator new(std::size_t size) {
    if (size == 0) size = 1;  // never request 0 bytes from malloc
    void* p = std::malloc(size);
    if (!p) throw std::bad_alloc();
    g_live_allocations.fetch_add(1, std::memory_order_relaxed);
    return p;
}

void* operator new[](std::size_t size) {
    if (size == 0) size = 1;
    void* p = std::malloc(size);
    if (!p) throw std::bad_alloc();
    g_live_allocations.fetch_add(1, std::memory_order_relaxed);
    return p;
}

// Nothrow forms ────────────────────────────────────────────────────────────────
void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
    if (size == 0) size = 1;
    void* p = std::malloc(size);
    if (p) g_live_allocations.fetch_add(1, std::memory_order_relaxed);
    return p;  // may be null; must not throw
}

void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
    if (size == 0) size = 1;
    void* p = std::malloc(size);
    if (p) g_live_allocations.fetch_add(1, std::memory_order_relaxed);
    return p;
}

// Delete forms (plain, sized, array, nothrow) ───────────────────────────────────
void operator delete(void* p) noexcept {
    if (p) {
        g_live_allocations.fetch_sub(1, std::memory_order_relaxed);
        std::free(p);
    }
}

void operator delete[](void* p) noexcept {
    if (p) {
        g_live_allocations.fetch_sub(1, std::memory_order_relaxed);
        std::free(p);
    }
}

void operator delete(void* p, std::size_t) noexcept { ::operator delete(p); }
void operator delete[](void* p, std::size_t) noexcept { ::operator delete[](p); }

void operator delete(void* p, const std::nothrow_t&) noexcept { ::operator delete(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { ::operator delete[](p); }

// ─── extern "C" CONF bridge prototypes (declared locally; no public C header) ──
extern "C" {
int conf_load_c(const char* path, int path_len, int* handle_out);
int conf_load_string_c(const char* yaml_text, int text_len, int* handle_out);
int conf_close_c(int handle);
int conf_has_key_c(int handle, const char* key, int key_len, int* exists_out);
int conf_size_c(int handle, const char* key, int key_len, int* size_out);
int conf_get_int_c(int handle, const char* key, int key_len, int* out);
int conf_get_double_c(int handle, const char* key, int key_len, double* out);
int conf_get_bool_c(int handle, const char* key, int key_len, int* out);
int conf_get_string_len_c(int handle, const char* key, int key_len, int* str_len_out);
int conf_get_string_c(int handle, const char* key, int key_len,
                      char* buf, int buf_cap, int* written_out);
}

namespace {

constexpr int kSuccess = 0;  // conf::Error_Code::Success

// A representative valid YAML document spanning every node kind the queries
// touch: scalars (int/double/bool/string), nested maps, and sequences.
const char* const kValidDoc =
    "title: HELM run deck\n"
    "count: 42\n"
    "ratio: 3.14159\n"
    "enabled: true\n"
    "name: alpha\n"
    "nested:\n"
    "  inner: 7\n"
    "  deep:\n"
    "    value: hello world\n"
    "items:\n"
    "  - 10\n"
    "  - 20\n"
    "  - 30\n"
    "words:\n"
    "  - foo\n"
    "  - bar\n";

// A fixed key menu mixing existing, missing, out-of-range, and malformed paths
// so generated query streams exercise Success, Key_Not_Found, Type_Mismatch and
// Invalid_Arg return codes — none of which may leak.
const std::vector<std::string>& key_menu() {
    static const std::vector<std::string> keys = {
        "count",             // int scalar
        "ratio",             // double scalar
        "enabled",           // bool scalar
        "name",              // string scalar
        "nested.inner",      // int scalar (deep)
        "nested.deep.value", // string scalar (deep, has a space)
        "items",             // sequence
        "items.0",           // int via sequence index
        "words.1",           // string via sequence index
        "title",             // string scalar
        "missing",           // absent leaf
        "nested.absent",     // absent intermediate child
        "items.99",          // out-of-range sequence index
        ".bad",              // malformed: leading dot
        "bad.",              // malformed: trailing dot
        "a..b",              // malformed: consecutive dots
    };
    return keys;
}

constexpr int kKindCount = 6;  // int, double, bool, has_key, size, string

/// Execute a single bridge query of the given kind against `handle`/`key`,
/// writing only into stack storage so the query itself performs no surviving
/// heap allocation in this TU. Return codes are intentionally ignored: the
/// no-leak invariant must hold for success AND every failure path.
void run_query(int handle, int kind, const std::string& key, bool small_buffer) {
    const char* k = key.data();
    const int klen = static_cast<int>(key.size());

    switch (kind) {
        case 0: {
            int out = 0;
            (void)conf_get_int_c(handle, k, klen, &out);
            break;
        }
        case 1: {
            double out = 0.0;
            (void)conf_get_double_c(handle, k, klen, &out);
            break;
        }
        case 2: {
            int out = 0;
            (void)conf_get_bool_c(handle, k, klen, &out);
            break;
        }
        case 3: {
            int exists = 0;
            (void)conf_has_key_c(handle, k, klen, &exists);
            break;
        }
        case 4: {
            int sz = 0;
            (void)conf_size_c(handle, k, klen, &sz);
            break;
        }
        default: {
            // Length-first ("two-call") string protocol, all stack-backed:
            // query the length, then copy into a fixed caller buffer. The
            // small-buffer variant forces the Buffer_Too_Small path.
            int need = 0;
            (void)conf_get_string_len_c(handle, k, klen, &need);

            char buf[256];
            const int cap = small_buffer ? 1 : static_cast<int>(sizeof(buf));
            int written = 0;
            (void)conf_get_string_c(handle, k, klen, buf, cap, &written);
            break;
        }
    }
}

/// Run a valid `load -> generated query stream -> close` sequence once.
/// Returns the load return code (kSuccess on the expected happy path). Used both
/// as the unmeasured priming run and the measured run, so a genuine per-call
/// leak recurs identically across both invocations.
int run_load_query_close(const std::vector<std::string>& keys,
                         const std::vector<std::pair<int, int>>& plan) {
    int handle = 0;
    const int load_rc = conf_load_string_c(
        kValidDoc, static_cast<int>(std::char_traits<char>::length(kValidDoc)), &handle);
    if (load_rc == kSuccess) {
        for (const auto& [kind, idx] : plan) {
            run_query(handle, kind, keys[idx], /*small_buffer=*/(idx % 3 == 0));
        }
        conf_close_c(handle);  // RAII frees the entire node tree
    }
    return load_rc;
}

/// Run a single FAILED load (no handle registered, nothing to close).
/// fail_kind 0 => malformed YAML (Parse_Error); 1 => missing file (File_Not_Found).
int run_failed_load(int fail_kind) {
    int handle = -1;
    if (fail_kind == 0) {
        const char* malformed = "root: {a: 1, : ]\n\tbad: : :";
        return conf_load_string_c(
            malformed, static_cast<int>(std::char_traits<char>::length(malformed)), &handle);
    }
    const char* missing_path = "/conf/no/such/file/here.yaml";
    return conf_load_c(
        missing_path, static_cast<int>(std::char_traits<char>::length(missing_path)), &handle);
}

}  // namespace

// ─── Property 4: a load -> N queries -> close sequence is allocation-balanced ─
// Feature: conf-config-parser, Property 4: No-Leak on the C Bridge
//
// Generates a random query stream (kind + key chosen from the menu) and, for
// most iterations, a valid load; the rest are failed loads that register no
// handle. The live-allocation counter captured immediately after the sequence
// must equal the value captured immediately before it.
//
// **Validates: Requirements 30.1, 30.2, 30.3**

RC_GTEST_PROP(NoLeakCBridgeProperty4, LoadQueriesCloseIsAllocationBalanced, ()) {
    const auto& keys = key_menu();
    const int key_count = static_cast<int>(keys.size());

    // ── Generate the full plan OUTSIDE the measured window ────────────────────
    // 1 in 5 iterations is a failed load (no handle, no close); the rest load
    // the valid document and run the generated query stream.
    const bool valid_load = (*rc::gen::inRange(0, 5)) != 0;
    const int fail_kind = *rc::gen::inRange(0, 2);  // 0: parse error, 1: missing file

    // Each query is (kind, key index). Buffer-too-small selection for string
    // queries is derived from the key index so both paths are exercised.
    const auto plan = *rc::gen::container<std::vector<std::pair<int, int>>>(
        rc::gen::pair(rc::gen::inRange(0, kKindCount),
                      rc::gen::inRange(0, key_count)));

    // ── Prime: run the exact sequence once, unmeasured, so any input-specific
    //    one-time lazy static allocation happens before the snapshot. A genuine
    //    per-call leak recurs identically on the measured run below. ──────────
    int load_rc = valid_load ? run_load_query_close(keys, plan)
                             : run_failed_load(fail_kind);

    // ── Measured window: snapshot -> identical sequence -> snapshot ───────────
    const long long before = g_live_allocations.load(std::memory_order_relaxed);
    load_rc = valid_load ? run_load_query_close(keys, plan)
                         : run_failed_load(fail_kind);
    const long long after = g_live_allocations.load(std::memory_order_relaxed);

    // ── Assertions (outside the window) ───────────────────────────────────────
    // Core invariant: the balanced sequence leaked nothing.
    RC_ASSERT(after == before);

    // Sanity: a valid document must actually load, and a failed load must report
    // a non-success code (so we know the leak check exercised the real paths).
    if (valid_load) {
        RC_ASSERT(load_rc == kSuccess);
    } else {
        RC_ASSERT(load_rc != kSuccess);
    }
}

// ─── Property 4 (focused): arbitrary / failed loads leave nothing resident ────
// Feature: conf-config-parser, Property 4: No-Leak on the C Bridge
//
// Drives conf_load_string_c with arbitrary byte strings (valid YAML, malformed
// YAML, non-UTF-8 bytes, empty). Whether the load succeeds or fails, the live
// allocation counter after the sequence must equal the value before it: a
// failed load registers and leaks nothing (Req 30.3), and a load that happens
// to succeed is balanced by the closing call (Req 30.1).
//
// **Validates: Requirements 30.1, 30.3**

RC_GTEST_PROP(NoLeakCBridgeProperty4, ArbitraryAndFailedLoadsLeaveNoResidentHeap, ()) {
    // Generate the candidate document OUTSIDE the measured window.
    const std::string text = *rc::gen::arbitrary<std::string>();

    // load-(maybe query)-close, used as both the priming and measured run.
    auto run_once = [&]() {
        int handle = -1;
        const int rc = conf_load_string_c(
            text.data(), static_cast<int>(text.size()), &handle);
        if (rc == kSuccess) {
            // Parsed as valid YAML; a couple of total-resolver queries, then close.
            int sz = 0;
            (void)conf_size_c(handle, "x", 1, &sz);
            int exists = 0;
            (void)conf_has_key_c(handle, "x", 1, &exists);
            conf_close_c(handle);
        }
        // On failure: no handle was registered, nothing to close (Req 30.3).
        return rc;
    };

    // Prime once (triggers any input-specific one-time lazy statics in yaml-cpp),
    // then measure an identical second run. A real leak recurs and is caught;
    // one-time initialization nets to zero.
    run_once();

    const long long before = g_live_allocations.load(std::memory_order_relaxed);
    run_once();
    const long long after = g_live_allocations.load(std::memory_order_relaxed);

    RC_ASSERT(after == before);
}
