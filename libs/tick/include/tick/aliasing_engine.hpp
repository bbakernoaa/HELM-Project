#pragma once

#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "tick/aliased_window.hpp"
#include "tick/calendar.hpp"
#include "tick/date_time.hpp"
#include "tick/duration.hpp"
#include "tick/out_of_bounds_policy.hpp"
#include "tick/time_point.hpp"
#include "tick/time_window.hpp"

namespace tick {

// Forward declarations for if-constexpr calendar type checks
struct Gregorian_Calendar;
struct NoLeap_Calendar;

template <Calendar Sim_Cal, Calendar Ds_Cal>
class Aliasing_Engine {
   public:
    /// Construct an immutable aliasing engine.
    /// @param coverage            Half-open interval [start, end) of dataset temporal range
    /// @param snapshot_interval   Duration between dataset snapshots (must evenly divide coverage)
    /// @param policy              Out-of-bounds remapping strategy
    /// @param climatological_year Reference year for pure_climatology (ignored by other policies)
    /// @throws std::invalid_argument if snapshot_interval <= 0, doesn't divide coverage,
    ///         or climatological_year is outside coverage for pure_climatology
    constexpr Aliasing_Engine(Time_Window coverage, Duration snapshot_interval, OutOfBoundsPolicy policy, std::int32_t climatological_year = 0)
        : coverage_{coverage}, snapshot_interval_{snapshot_interval}, policy_{policy}, climatological_year_{climatological_year} {
        if (snapshot_interval.nanos() <= 0) {
            throw std::invalid_argument("Aliasing_Engine: snapshot_interval must be positive");
        }

        if (coverage.duration().nanos() % snapshot_interval.nanos() != 0) {
            throw std::invalid_argument("Aliasing_Engine: snapshot_interval must evenly divide coverage duration");
        }

        if (policy == OutOfBoundsPolicy::pure_climatology) {
            auto start_dt = Ds_Cal::to_date_time(coverage.start());
            auto end_dt = Ds_Cal::to_date_time(coverage.end() - Duration{1});

            if (climatological_year < start_dt.year || climatological_year > end_dt.year) {
                throw std::invalid_argument("Aliasing_Engine: climatological_year must be within dataset coverage year range");
            }
        }
    }

    // ─── Public interface ────────────────────────────────────────────────

    /// Resolve a simulation Time_Point to an AliasedWindow.
    [[nodiscard]] constexpr AliasedWindow resolve(Time_Point sim_time) const {
        // pure_climatology always applies year substitution regardless of bounds
        if (policy_ == OutOfBoundsPolicy::pure_climatology) {
            return resolve_climatology(sim_time);
        }

        // leap_hold: check for Feb 29 freeze BEFORE bounds check
        // (Requirement 6.1: ANY Feb 29 freezes regardless of in/out-of-bounds)
        if (policy_ == OutOfBoundsPolicy::leap_hold) {
            if constexpr (std::is_same_v<Sim_Cal, Gregorian_Calendar> && std::is_same_v<Ds_Cal, NoLeap_Calendar>) {
                Date_Time dt = Sim_Cal::to_date_time(sim_time);
                if (is_feb29(dt)) {
                    Time_Point t_left = Ds_Cal::to_time_point(Date_Time{dt.year, 2, 28, 0, 0, 0, 0});
                    Time_Point t_right = Ds_Cal::to_time_point(Date_Time{dt.year, 3, 1, 0, 0, 0, 0});
                    return AliasedWindow{Time_Window{t_left, t_right}, 0.0};
                }
            }
        }

        // For all other policies: check bounds first
        if (sim_time >= coverage_.start() && sim_time < coverage_.end()) {
            return resolve_in_bounds(sim_time);
        }

        // Out-of-bounds: dispatch by policy
        switch (policy_) {
            case OutOfBoundsPolicy::clamp_to_edge:
                return resolve_clamp(sim_time);
            case OutOfBoundsPolicy::cycle_last_year:
                return resolve_cycle(sim_time);
            case OutOfBoundsPolicy::leap_hold:
                return resolve_leap_hold(sim_time);
            default:
                // Should be unreachable due to enum class
                return resolve_clamp(sim_time);
        }
    }

    /// Calculate interpolation weight for a time within a window.
    /// Returns alpha in [0.0, 1.0) representing linear position of current within window.
    /// @throws std::invalid_argument if window has zero duration (degenerate window)
    /// @throws std::out_of_range if current is not within [window.start(), window.end())
    [[nodiscard]] static constexpr double calculate_weight(Time_Point current, Time_Window window) {
        if (window.duration().nanos() == 0) {
            throw std::invalid_argument("calculate_weight: degenerate window with zero duration");
        }

        if (current < window.start() || current >= window.end()) {
            throw std::out_of_range("calculate_weight: current time is not within the supplied window");
        }

        // All differences computed as int64_t first, single double division at the end
        const std::int64_t numerator = current.nanos() - window.start().nanos();
        const std::int64_t denominator = window.end().nanos() - window.start().nanos();

        return static_cast<double>(numerator) / static_cast<double>(denominator);
    }

    // ─── Const Accessors ─────────────────────────────────────────────────

    [[nodiscard]] constexpr Time_Window coverage() const noexcept {
        return coverage_;
    }
    [[nodiscard]] constexpr Duration snapshot_interval() const noexcept {
        return snapshot_interval_;
    }
    [[nodiscard]] constexpr OutOfBoundsPolicy policy() const noexcept {
        return policy_;
    }
    [[nodiscard]] constexpr std::int32_t climatological_year() const noexcept {
        return climatological_year_;
    }

   private:
    const Time_Window coverage_;
    const Duration snapshot_interval_;
    const OutOfBoundsPolicy policy_;
    const std::int32_t climatological_year_;

    // ─── Policy dispatch (private, const) ────────────────────────────────

    [[nodiscard]] constexpr AliasedWindow resolve_in_bounds(Time_Point sim_time) const {
        auto window = enclosing_window(sim_time, coverage_.start(), snapshot_interval_);
        auto alpha = calculate_weight(sim_time, window);
        return AliasedWindow{window, alpha};
    }

    [[nodiscard]] constexpr AliasedWindow resolve_clamp(Time_Point sim_time) const {
        if (sim_time >= coverage_.end()) {
            // Freeze at last snapshot: window is [end - interval, end), alpha = 0.0
            auto t_left = coverage_.end() - snapshot_interval_;
            return AliasedWindow{Time_Window{t_left, coverage_.end()}, 0.0};
        }

        if (sim_time < coverage_.start()) {
            // Freeze at first snapshot: window is [start, start + interval), alpha = 0.0
            auto t_right = coverage_.start() + snapshot_interval_;
            return AliasedWindow{Time_Window{coverage_.start(), t_right}, 0.0};
        }

        // In-bounds: standard interpolation
        return resolve_in_bounds(sim_time);
    }

    [[nodiscard]] constexpr AliasedWindow resolve_cycle(Time_Point sim_time) const {
        // Decompose simulation time into Date_Time components using the Simulation Calendar
        Date_Time dt = Sim_Cal::to_date_time(sim_time);

        // Determine target year based on whether sim_time is past end or before start
        std::int32_t target_year{};
        if (sim_time >= coverage_.end()) {
            // Last complete year of coverage (year of the last nanosecond in coverage)
            target_year = Ds_Cal::to_date_time(coverage_.end() - Duration{1}).year;
        } else {
            // First complete year of coverage (year of the first nanosecond in coverage)
            target_year = Ds_Cal::to_date_time(coverage_.start()).year;
        }

        // Substitute year
        dt.year = target_year;

        // Clamp day to handle cases like Feb 29 → Feb 28 in NoLeap calendar
        auto max_day = Ds_Cal::days_in_month(target_year, dt.month);
        if (dt.day > max_day) {
            dt.day = max_day;
        }

        // Convert remapped Date_Time back to a Time_Point using the Dataset Calendar
        auto remapped_tp = Ds_Cal::to_time_point(dt);

        // Resolve the remapped time point within dataset coverage
        return resolve_in_bounds(remapped_tp);
    }

    [[nodiscard]] constexpr AliasedWindow resolve_climatology(Time_Point sim_time) const {
        // 1. Decompose simulation time into calendar components using the Simulation Calendar
        Date_Time dt = Sim_Cal::to_date_time(sim_time);

        // 2. Substitute year with the climatological year, preserve all other components
        dt.year = climatological_year_;

        // 3. Clamp day to valid range for the target month in the Dataset Calendar
        auto max_day = Ds_Cal::days_in_month(climatological_year_, dt.month);
        if (dt.day > max_day) {
            dt.day = max_day;
        }

        // 4. Convert remapped Date_Time back to a Time_Point using the Dataset Calendar
        Time_Point remapped_tp = Ds_Cal::to_time_point(dt);

        // 5. Compute enclosing window anchored at the start of the climatological year
        Time_Point clim_year_start = Ds_Cal::to_time_point(Date_Time{climatological_year_, 1, 1, 0, 0, 0, 0});
        auto window = enclosing_window(remapped_tp, clim_year_start, snapshot_interval_);

        // 6. Handle December-to-January year boundary:
        //    If the window end crosses into the next year, construct the boundary window
        //    explicitly using climatological_year_ + 1 for t_right.
        Date_Time window_end_dt = Ds_Cal::to_date_time(window.end());
        if (window_end_dt.year > climatological_year_) {
            // Window spans the Dec→Jan year boundary
            Time_Point t_right = Ds_Cal::to_time_point(Date_Time{climatological_year_ + 1, 1, 1, 0, 0, 0, 0});
            Time_Point t_left = t_right - snapshot_interval_;
            window = Time_Window{t_left, t_right};
        }

        // 7. Compute interpolation weight and return AliasedWindow
        double alpha = calculate_weight(remapped_tp, window);
        return AliasedWindow{window, alpha};
    }

    [[nodiscard]] constexpr AliasedWindow resolve_leap_hold(Time_Point sim_time) const {
        Date_Time dt = Sim_Cal::to_date_time(sim_time);

        if constexpr (std::is_same_v<Sim_Cal, Gregorian_Calendar> && std::is_same_v<Ds_Cal, NoLeap_Calendar>) {
            if (is_feb29(dt)) {
                // Feb 29 in Gregorian, but NoLeap has no Feb 29.
                // Hold at Feb 28 → Mar 1 window with alpha = 0.0
                Time_Point t_left = Ds_Cal::to_time_point(Date_Time{dt.year, 2, 28, 0, 0, 0, 0});
                Time_Point t_right = Ds_Cal::to_time_point(Date_Time{dt.year, 3, 1, 0, 0, 0, 0});
                return AliasedWindow{Time_Window{t_left, t_right}, 0.0};
            }
        }

        // Non-Feb-29 or non-Gregorian→NoLeap: delegate appropriately
        if (sim_time >= coverage_.end() || sim_time < coverage_.start()) {
            return resolve_clamp(sim_time);
        }
        return resolve_in_bounds(sim_time);
    }

    // ─── Utility (private, static, constexpr) ────────────────────────────

    [[nodiscard]] static constexpr Time_Window enclosing_window(Time_Point t, Time_Point coverage_start, Duration interval) {
        auto offset_nanos = t.nanos() - coverage_start.nanos();
        auto iv = interval.nanos();
        auto floor_offset = (offset_nanos / iv) * iv;
        auto floor_start = Time_Point{coverage_start.nanos() + floor_offset};
        return Time_Window{floor_start, floor_start + interval};
    }

    [[nodiscard]] static constexpr bool is_feb29(const Date_Time &dt) noexcept {
        return dt.month == 2 && dt.day == 29;
    }
};

}  // namespace tick
