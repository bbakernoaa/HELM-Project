#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace tick::detail {

/// Overflow-checked addition for int64_t.
/// Returns a + b, or throws std::overflow_error if the result overflows.
[[nodiscard]] constexpr auto checked_add(std::int64_t a, std::int64_t b) -> std::int64_t {
    if (std::is_constant_evaluated()) {
        // Constexpr path: limit-based pre-checks
        if (b > 0 && a > std::numeric_limits<std::int64_t>::max() - b) {
            throw std::overflow_error("Integer overflow in addition");
        }
        if (b < 0 && a < std::numeric_limits<std::int64_t>::min() - b) {
            throw std::overflow_error("Integer overflow in addition");
        }
        return a + b;
    } else {
        // Runtime path: compiler builtins
        std::int64_t result{};
        if (__builtin_add_overflow(a, b, &result)) {
            throw std::overflow_error("Integer overflow in addition");
        }
        return result;
    }
}

/// Overflow-checked subtraction for int64_t.
/// Returns a - b, or throws std::overflow_error if the result overflows.
[[nodiscard]] constexpr auto checked_sub(std::int64_t a, std::int64_t b) -> std::int64_t {
    if (std::is_constant_evaluated()) {
        // Constexpr path: limit-based pre-checks
        if (b < 0 && a > std::numeric_limits<std::int64_t>::max() + b) {
            throw std::overflow_error("Integer overflow in subtraction");
        }
        if (b > 0 && a < std::numeric_limits<std::int64_t>::min() + b) {
            throw std::overflow_error("Integer overflow in subtraction");
        }
        return a - b;
    } else {
        // Runtime path: compiler builtins
        std::int64_t result{};
        if (__builtin_sub_overflow(a, b, &result)) {
            throw std::overflow_error("Integer overflow in subtraction");
        }
        return result;
    }
}

/// Overflow-checked multiplication for int64_t.
/// Returns a * b, or throws std::overflow_error if the result overflows.
[[nodiscard]] constexpr auto checked_mul(std::int64_t a, std::int64_t b) -> std::int64_t {
    if (std::is_constant_evaluated()) {
        // Constexpr path: limit-based pre-checks
        if (a == 0 || b == 0) {
            return 0;
        }

        constexpr auto max_val = std::numeric_limits<std::int64_t>::max();
        constexpr auto min_val = std::numeric_limits<std::int64_t>::min();

        // Handle the special case of MIN * -1 (or -1 * MIN) which overflows
        if (a == -1) {
            if (b == min_val) {
                throw std::overflow_error("Integer overflow in multiplication");
            }
            return -b;
        }
        if (b == -1) {
            if (a == min_val) {
                throw std::overflow_error("Integer overflow in multiplication");
            }
            return -a;
        }

        // General case: check |a| * |b| against limits
        if (a > 0) {
            if (b > 0) {
                if (a > max_val / b) {
                    throw std::overflow_error("Integer overflow in multiplication");
                }
            } else {
                // b < 0 (b != 0 and b != -1 already handled)
                if (b < min_val / a) {
                    throw std::overflow_error("Integer overflow in multiplication");
                }
            }
        } else {
            // a < 0 (a != 0 and a != -1 already handled)
            if (b > 0) {
                if (a < min_val / b) {
                    throw std::overflow_error("Integer overflow in multiplication");
                }
            } else {
                // both negative
                if (a < max_val / b) {
                    throw std::overflow_error("Integer overflow in multiplication");
                }
            }
        }

        return a * b;
    } else {
        // Runtime path: compiler builtins
        std::int64_t result{};
        if (__builtin_mul_overflow(a, b, &result)) {
            throw std::overflow_error("Integer overflow in multiplication");
        }
        return result;
    }
}

}  // namespace tick::detail
