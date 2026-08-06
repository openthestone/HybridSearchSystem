/*
 * Stub for NpuRetrieval `src/utils/alarm.h`.
 *
 * Used once, in data_table_repository.cpp (NPU HBM usage check):
 *   SendAlarmHcm(AlarmLevel::MAJOR, "title", msg);
 * The stub routes the alarm to the log instead of the real HCM channel.
 */
#pragma once

#include <string>

#include "src/utils/logger.h"

namespace NpuRetrieval {

enum class AlarmLevel {
    MINOR = 0,
    MAJOR = 1,
    CRITICAL = 2,
};

inline void SendAlarmHcm(AlarmLevel level, const std::string& title, const std::string& message) {
    LOG_WARN("[ALARM level=" << static_cast<int>(level) << "] " << title << " : " << message);
}

}  // namespace NpuRetrieval
