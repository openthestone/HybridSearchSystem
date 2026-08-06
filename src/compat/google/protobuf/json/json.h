/*
 * Compatibility shim for extracted NpuRetrieval sources.
 *
 * Newer protobuf releases expose JSON helpers as:
 *   google/protobuf/json/json.h
 *   google::protobuf::json::{ParseOptions, JsonStringToMessage, ...}
 *
 * The server used by the original project has protobuf available, but likely
 * exposes the older, common API:
 *   google/protobuf/util/json_util.h
 *   google::protobuf::util::{JsonParseOptions, JsonStringToMessage, ...}
 *
 * Keep Huawei sources unchanged by providing the newer include path and
 * namespace on top of the older util API.
 */
#pragma once

#include <string>

#include "google/protobuf/message.h"
#include "google/protobuf/util/json_util.h"

namespace google {
namespace protobuf {
namespace json {

using ParseOptions = ::google::protobuf::util::JsonParseOptions;
using PrintOptions = ::google::protobuf::util::JsonPrintOptions;

class Status {
   public:
    Status() = default;

    template <typename StatusLike>
    explicit Status(const StatusLike& status)
        : ok_(status.ok()), code_(ExtractCode(status)), message_(ExtractMessage(status)) {}

    bool ok() const {
        return ok_;
    }

    int code() const {
        return code_;
    }

    const std::string& message() const {
        return message_;
    }

   private:
    template <typename StatusLike>
    static auto ExtractCodeImpl(const StatusLike& status, int) -> decltype(status.code(), int()) {
        return static_cast<int>(status.code());
    }

    template <typename StatusLike>
    static auto ExtractCodeImpl(const StatusLike& status, long) -> decltype(status.error_code(), int()) {
        return static_cast<int>(status.error_code());
    }

    template <typename StatusLike>
    static int ExtractCodeImpl(const StatusLike&, ...) {
        return 0;
    }

    template <typename StatusLike>
    static int ExtractCode(const StatusLike& status) {
        return ExtractCodeImpl(status, 0);
    }

    template <typename StatusLike>
    static auto ExtractMessageImpl(const StatusLike& status, int) -> decltype(status.message(), std::string()) {
        const auto msg = status.message();
        return std::string(msg.data(), msg.size());
    }

    template <typename StatusLike>
    static auto ExtractMessageImpl(const StatusLike& status, long) -> decltype(status.error_message(), std::string()) {
        const auto msg = status.error_message();
        return std::string(msg.data(), msg.size());
    }

    template <typename StatusLike>
    static std::string ExtractMessageImpl(const StatusLike&, ...) {
        return {};
    }

    template <typename StatusLike>
    static std::string ExtractMessage(const StatusLike& status) {
        return ExtractMessageImpl(status, 0);
    }

    bool ok_ = true;
    int code_ = 0;
    std::string message_;
};

inline Status JsonStringToMessage(const std::string& input, ::google::protobuf::Message* message,
                                  const ParseOptions& options) {
    return Status(::google::protobuf::util::JsonStringToMessage(input, message, options));
}

inline Status JsonStringToMessage(const std::string& input, ::google::protobuf::Message* message) {
    return Status(::google::protobuf::util::JsonStringToMessage(input, message));
}

inline Status MessageToJsonString(const ::google::protobuf::Message& message, std::string* output) {
    return Status(::google::protobuf::util::MessageToJsonString(message, output));
}

inline Status MessageToJsonString(const ::google::protobuf::Message& message, std::string* output,
                                  const PrintOptions& options) {
    return Status(::google::protobuf::util::MessageToJsonString(message, output, options));
}

}  // namespace json
}  // namespace protobuf
}  // namespace google
