#pragma once

// TICK — Time Integration & Chronology Kernel
// Umbrella header: includes the complete TICK public API.

// Core value types
#include "tick/duration.hpp"
#include "tick/time_point.hpp"
#include "tick/date_time.hpp"

// Calendar engines
#include "tick/calendar.hpp"
#include "tick/gregorian_calendar.hpp"
#include "tick/noleap_calendar.hpp"
#include "tick/cal360_calendar.hpp"

// Alarm system
#include "tick/interval_alarm.hpp"
#include "tick/absolute_alarm.hpp"

// Synchronization
#include "tick/sync.hpp"

// Accumulation windows
#include "tick/time_window.hpp"

// Aliasing
#include "tick/out_of_bounds_policy.hpp"
#include "tick/aliased_window.hpp"
#include "tick/aliasing_engine.hpp"
