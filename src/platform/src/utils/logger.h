/*
 * Stub for NpuRetrieval `src/utils/logger.h`.
 *
 * The real logger wraps log4cplus. `engine/` code uses:
 *   - LOG_DEBUG / LOG_INFO / LOG_WARN / LOG_ERROR (with `<<` streaming)
 *   - NpuRetrieval::Logger::Instance().Init(properties)
 *   - NpuRetrieval::Logger::GetLogLevel() compared against
 *     log4cplus::DEBUG_LOG_LEVEL / INFO_LOG_LEVEL
 *
 * This stub reproduces exactly that surface on top of std::cerr, so `engine/`
 * sources compile and run without the full log4cplus dependency. Level
 * semantics follow log4cplus: a message at level L is emitted when the
 * current threshold <= L (lower threshold == more verbose).
 */
#pragma once

#include <iostream>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <string>

// ---- minimal log4cplus level shim -----------------------------------------
namespace log4cplus {
// Values kept ascending like log4cplus so `<=` comparisons behave the same.
enum LogLevelShim {
    TRACE_LOG_LEVEL = 0,
    DEBUG_LOG_LEVEL = 10000,
    INFO_LOG_LEVEL = 20000,
    WARN_LOG_LEVEL = 30000,
    ERROR_LOG_LEVEL = 40000,
    FATAL_LOG_LEVEL = 50000,
    OFF_LOG_LEVEL = 60000,
};
}  // namespace log4cplus

namespace NpuRetrieval {

class Logger {
   public:
    static Logger& Instance() {
        static Logger instance;
        return instance;
    }

    // Real signature takes a log.properties path; the stub ignores it.
    bool Init(const std::string& /*properties*/ = "") {
        return true;
    }

    // Threshold: a message at level L is emitted when GetLogLevel() <= L.
    // Initialised from env NPUR_LOG_LEVEL (TRACE/DEBUG/INFO/WARN/ERROR),
    // defaulting to DEBUG. Set e.g. NPUR_LOG_LEVEL=WARN to silence the very
    // verbose per-token / per-doc DEBUG logging in the offline builder.
    static int GetLogLevel() {
        return LevelRef();
    }
    static void SetLogLevel(int level) {
        LevelRef() = level;
    }

    static std::mutex& Mutex() {
        static std::mutex m;
        return m;
    }

   private:
    Logger() = default;
    static int& LevelRef() {
        static int level = [] {
            const char* e = std::getenv("NPUR_LOG_LEVEL");
            if (e != nullptr) {
                std::string s(e);
                if (s == "TRACE")
                    return static_cast<int>(log4cplus::TRACE_LOG_LEVEL);
                if (s == "DEBUG")
                    return static_cast<int>(log4cplus::DEBUG_LOG_LEVEL);
                if (s == "INFO")
                    return static_cast<int>(log4cplus::INFO_LOG_LEVEL);
                if (s == "WARN")
                    return static_cast<int>(log4cplus::WARN_LOG_LEVEL);
                if (s == "ERROR")
                    return static_cast<int>(log4cplus::ERROR_LOG_LEVEL);
            }
            return static_cast<int>(log4cplus::DEBUG_LOG_LEVEL);
        }();
        return level;
    }
};

}  // namespace NpuRetrieval

// ---- logging macros --------------------------------------------------------
#define NPUR_LOG_AT(levelTag, levelVal, msgExpr)                                 \
    do {                                                                         \
        if (::NpuRetrieval::Logger::GetLogLevel() <= (levelVal)) {               \
            std::ostringstream _tj_los;                                          \
            _tj_los << msgExpr;                                                  \
            std::lock_guard<std::mutex> _tj_lk(::NpuRetrieval::Logger::Mutex()); \
            std::cerr << "[" levelTag "] " << _tj_los.str() << std::endl;        \
        }                                                                        \
    } while (0)

#define LOG_TRACE(msgExpr) NPUR_LOG_AT("TRACE", ::log4cplus::TRACE_LOG_LEVEL, msgExpr)
#define LOG_DEBUG(msgExpr) NPUR_LOG_AT("DEBUG", ::log4cplus::DEBUG_LOG_LEVEL, msgExpr)
#define LOG_INFO(msgExpr) NPUR_LOG_AT("INFO", ::log4cplus::INFO_LOG_LEVEL, msgExpr)
#define LOG_WARN(msgExpr) NPUR_LOG_AT("WARN", ::log4cplus::WARN_LOG_LEVEL, msgExpr)
#define LOG_ERROR(msgExpr) NPUR_LOG_AT("ERROR", ::log4cplus::ERROR_LOG_LEVEL, msgExpr)
