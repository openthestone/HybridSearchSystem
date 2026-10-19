/*
 * Stub for NpuRetrieval `src/common/error_code.h`.
 *
 * `engine/` code only uses ErrorCode::ResultType::{SUCCESS, FAIL} as the return
 * type of the async task lambdas fed to the Executor.
 */
#pragma once

namespace NpuRetrieval {
namespace ErrorCode {

enum class ResultType {
    SUCCESS = 0,
    FAIL = 1,
};

}  // namespace ErrorCode
}  // namespace NpuRetrieval
