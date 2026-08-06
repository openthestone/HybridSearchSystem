/*
 * Stub for NpuRetrieval `src/utils/rapidjson_util.h`.
 *
 * Used only by build/builder/builder_impl.cpp (memory-build attachment parsing):
 *   bool GetStringFromValueObj(const rapidjson::Value& obj, std::string& out, const std::string& key);
 *   bool GetUintFromValueObj(const rapidjson::Value& obj, uint32_t& out, const std::string& key);
 *
 * Requires the (header-only) RapidJSON library to be on the include path.
 * Only needed by the builder target, not by the search harness.
 */
#pragma once

#include <cstdint>
#include <string>

#include "rapidjson/document.h"

namespace NpuRetrieval {

// Parse a JSON string into a rapidjson Document. Returns false on parse error.
inline bool ReadJsonFromString(const std::string& json, rapidjson::Document& doc) {
    doc.Parse(json.c_str(), json.size());
    return !doc.HasParseError();
}

inline bool GetStringFromValueObj(const rapidjson::Value& obj, std::string& out, const std::string& key) {
    auto it = obj.FindMember(key.c_str());
    if (it == obj.MemberEnd() || !it->value.IsString()) {
        return false;
    }
    out.assign(it->value.GetString(), it->value.GetStringLength());
    return true;
}

inline bool GetUintFromValueObj(const rapidjson::Value& obj, uint32_t& out, const std::string& key) {
    auto it = obj.FindMember(key.c_str());
    if (it == obj.MemberEnd() || !it->value.IsUint()) {
        return false;
    }
    out = it->value.GetUint();
    return true;
}

}  // namespace NpuRetrieval
