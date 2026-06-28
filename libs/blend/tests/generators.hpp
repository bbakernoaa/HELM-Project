// SPDX-License-Identifier: Apache-2.0
// BLEND Test Suite — RapidCheck generators for array blending tests
//
// Provides reusable generator functions for BLEND property tests.
// All generators are in the blend::gen namespace and return rc::Gen<T> objects.
//
// Requirements: 1.7, 1.8, 1.9

#pragma once

#include <rapidcheck.h>

#include <Kokkos_Core.hpp>
#include <blend/helm_math_blend.hpp>
#include <cstddef>
#include <vector>

namespace blend::gen {

/// Generate a random array length in [1, max_len].
inline auto array_length(std::size_t max_len = 256) {
    return rc::gen::inRange<std::size_t>(1, max_len + 1);
}

/// Generate a vector of random doubles in [-1e6, 1e6].
/// Uses integer mapping since rc::gen::inRange<double> is unsupported.
inline auto field_data(std::size_t len) {
    auto elem_gen = rc::gen::map(rc::gen::inRange<int>(-1000000000, 1000000001), [](int x) { return static_cast<double>(x) / 1000.0; });
    return rc::gen::container<std::vector<double>>(len, elem_gen);
}

/// Generate an alpha value uniformly in [0.0, 1.0].
inline auto alpha_unit() {
    return rc::gen::map(rc::gen::inRange<int>(0, 10001), [](int x) { return static_cast<double>(x) / 10000.0; });
}

/// Generate an alpha value across full double range [-10.0, 10.0].
inline auto alpha_wide() {
    return rc::gen::map(rc::gen::inRange<int>(-100000, 100001), [](int x) { return static_cast<double>(x) / 10000.0; });
}

}  // namespace blend::gen
