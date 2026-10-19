/*
 * env_switch.h -- how the NPUR_* runtime switches are read.
 *
 * NOT a stub of an out-of-repo dependency (unlike the rest of this directory): the switches are
 * ours, and so is this. It lives here because this is the include root `engine/` can reach.
 *
 * There is deliberately no single EnvBool(). The switches grew five different boolean parses and
 * they do NOT agree -- NPUR_POOL_STATS=2 is ON under OnUnlessZero and OFF under On -- so each one
 * keeps its own named function and the call site says which it is, including whether the switch
 * defaults on or off.
 */
#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace npur_env {

// Default OFF; only the exact value "1" turns it on. Use this for a new switch.
inline bool On(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] == '1';
}

// Default OFF; only the exact string "1". Differs from On() only on values like "10" -- two
// sites were written this way and run.sh never emits such a value, so folding them into On() is
// safe the moment someone wants to; until then the parse is preserved.
inline bool OnExact(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && std::strcmp(v, "1") == 0;
}

// Default ON; off only on the exact string "0". See OnExact() for why this is separate.
inline bool OnByDefaultExact(const char* name) {
    const char* v = std::getenv(name);
    return v == nullptr || std::strcmp(v, "0") != 0;
}

// Default OFF; set to anything that does not start with '0' turns it on.
inline bool OnUnlessZero(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != '0';
}

// Default ON; only a value starting with '0' turns it off.
inline bool OnByDefault(const char* name) {
    const char* v = std::getenv(name);
    return v == nullptr || v[0] != '0';
}

// Default ON; off on any value that parses to 0 ("0", "00", "no", ... -- atoi returns 0).
inline bool OnByDefaultNumeric(const char* name) {
    const char* v = std::getenv(name);
    return v == nullptr || std::atoi(v) != 0;
}

// Present at all, whatever the value. For the switches that are pure presence flags.
inline bool Present(const char* name) {
    return std::getenv(name) != nullptr;
}

// Unset -> fallback. Otherwise strtol, cast to uint32_t, uncapped.
inline uint32_t U32(const char* name, uint32_t fallback) {
    const char* v = std::getenv(name);
    return v == nullptr ? fallback : static_cast<uint32_t>(std::strtol(v, nullptr, 10));
}

// Unset -> 0. Otherwise strtol, cast to uint32_t (so a NEGATIVE value wraps large and trips
// `over`, which is what the hand-written versions did), then anything above `max` becomes `over`.
inline uint32_t U32Capped(const char* name, uint32_t max, uint32_t over) {
    const char* v = std::getenv(name);
    if (v == nullptr) {
        return 0;
    }
    const uint32_t parsed = static_cast<uint32_t>(std::strtol(v, nullptr, 10));
    return parsed > max ? over : parsed;
}

// Unset or out of [lo, hi] -> clamped into the range. Garbage parses as 0 and is clamped too.
inline uint32_t U32Clamped(const char* name, uint32_t fallback, uint32_t lo, uint32_t hi) {
    const char* v = std::getenv(name);
    if (v == nullptr) {
        return fallback;
    }
    const long x = std::strtol(v, nullptr, 10);
    return x < static_cast<long>(lo) ? lo : (x > static_cast<long>(hi) ? hi : static_cast<uint32_t>(x));
}

// Unset, unparsable or <= 0 -> fallback.
inline uint64_t PositiveOr(const char* name, uint64_t fallback) {
    const char* v = std::getenv(name);
    if (v == nullptr) {
        return fallback;
    }
    const long long parsed = std::strtoll(v, nullptr, 10);
    return parsed <= 0 ? fallback : static_cast<uint64_t>(parsed);
}

// Unset or outside (lo, hi] -> fallback. atof, so garbage parses as 0 and falls back.
inline double DoubleInRange(const char* name, double fallback, double lo, double hi) {
    const char* v = std::getenv(name);
    const double r = (v != nullptr) ? std::atof(v) : fallback;
    return (r > lo && r <= hi) ? r : fallback;
}

}  // namespace npur_env
