// SPDX-License-Identifier: Apache-2.0
// SPAN — Property-based tests for HandleRegistry
// Feature: span-field-view, Property 8: Handle Registry Uniqueness and Isolation
//
// **Validates: Requirements 10.2, 10.3, 10.4, 10.7, 10.9, 2.5, 2.8**

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <memory>
#include <set>
#include <vector>

#include "handle_registry.hpp"

namespace {

/// Helper: register N dummy objects into the registry, returning their handles.
std::vector<int> register_n(span::detail::HandleRegistry &reg, std::size_t n) {
    std::vector<int> handles;
    handles.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        auto obj = std::make_shared<int>(static_cast<int>(i));
        int h = reg.register_view(obj);
        handles.push_back(h);
    }
    return handles;
}

/// Helper: unregister all handles from the registry.
void unregister_all(span::detail::HandleRegistry &reg, const std::vector<int> &handles) {
    for (int h : handles) {
        reg.unregister(h);
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Property (a)+(b): All handles are positive and unique; handle 0 is never assigned.
// Validates: Requirements 10.2, 10.3, 10.9
// ---------------------------------------------------------------------------
RC_GTEST_PROP(HandleRegistry, UniquenessAndPositivity, ()) {
    auto &reg = span::detail::HandleRegistry::instance();
    const auto n = *rc::gen::inRange<std::size_t>(1, 100);

    auto handles = register_n(reg, n);

    // (a) All handles are positive
    for (int h : handles) {
        RC_ASSERT(h > 0);
    }

    // (a) All handles are unique within the set of currently-active handles
    std::set<int> unique_handles(handles.begin(), handles.end());
    RC_ASSERT(unique_handles.size() == handles.size());

    // (b) Handle value 0 is never assigned
    RC_ASSERT(unique_handles.count(0) == 0);

    // Cleanup: unregister all to maintain clean state for next iteration
    unregister_all(reg, handles);
}

// ---------------------------------------------------------------------------
// Property (c): After unregister(h), lookup(h) returns nullptr.
// Validates: Requirements 10.4, 10.7, 2.5
// ---------------------------------------------------------------------------
RC_GTEST_PROP(HandleRegistry, UnregisterInvalidatesLookup, ()) {
    auto &reg = span::detail::HandleRegistry::instance();
    const auto n = *rc::gen::inRange<std::size_t>(1, 50);

    auto handles = register_n(reg, n);

    // Unregister the first half
    std::size_t half = n / 2;
    for (std::size_t i = 0; i < half; ++i) {
        bool ok = reg.unregister(handles[i]);
        RC_ASSERT(ok);
    }

    // (c) lookup on unregistered handles returns nullptr
    for (std::size_t i = 0; i < half; ++i) {
        auto result = reg.lookup(handles[i]);
        bool is_null = (result == nullptr);
        RC_ASSERT(is_null);
        RC_ASSERT(!reg.valid(handles[i]));
    }

    // Still-registered handles remain valid
    for (std::size_t i = half; i < n; ++i) {
        auto result = reg.lookup(handles[i]);
        bool is_valid = (result != nullptr);
        RC_ASSERT(is_valid);
        RC_ASSERT(reg.valid(handles[i]));
    }

    // Cleanup: unregister the remaining handles
    for (std::size_t i = half; i < n; ++i) {
        reg.unregister(handles[i]);
    }
}

// ---------------------------------------------------------------------------
// Property (d): If handle h is reused for a new registration, lookup(h)
// returns the new object, not the previously associated one.
// Validates: Requirements 10.4, 10.9, 2.8
// ---------------------------------------------------------------------------
RC_GTEST_PROP(HandleRegistry, HandleReuseIsolation, ()) {
    auto &reg = span::detail::HandleRegistry::instance();

    // Register an object
    auto obj1 = std::make_shared<int>(42);
    int h = reg.register_view(obj1);
    RC_ASSERT(h > 0);

    // Verify lookup returns obj1
    {
        auto looked_up = reg.lookup(h);
        bool is_obj1 = (looked_up.get() == static_cast<void *>(obj1.get()));
        RC_ASSERT(is_obj1);
    }

    // Unregister the handle
    reg.unregister(h);
    {
        auto looked_up = reg.lookup(h);
        bool is_null = (looked_up == nullptr);
        RC_ASSERT(is_null);
    }

    // Register a new object — may or may not get the same handle (free-list reuse)
    auto obj2 = std::make_shared<int>(99);
    int h2 = reg.register_view(obj2);

    // (d) If the handle is reused, lookup returns the NEW object
    if (h2 == h) {
        auto looked_up = reg.lookup(h2);
        bool points_to_obj2 = (looked_up.get() == static_cast<void *>(obj2.get()));
        bool not_obj1 = (looked_up.get() != static_cast<void *>(obj1.get()));
        RC_ASSERT(points_to_obj2);
        RC_ASSERT(not_obj1);
    }

    // Either way, the new handle must be valid and point to obj2
    RC_ASSERT(h2 > 0);
    {
        auto looked_up = reg.lookup(h2);
        bool points_to_obj2 = (looked_up.get() == static_cast<void *>(obj2.get()));
        RC_ASSERT(points_to_obj2);
    }

    // Cleanup
    reg.unregister(h2);
}
