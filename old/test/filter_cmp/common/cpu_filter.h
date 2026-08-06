// CPU forward-bitmap filter (sks_hw style): per-doc RPN eval over uint64 tag bitmap.
#pragma once

#include <cstdint>
#include <vector>

#include "expr.h"
#include "raw_data.h"

namespace filter_cmp {

// Returns out[d] = 1 if doc d matches expr, else 0.
// Multi-threaded over docs. Uses small fixed-size stack (depth capped at rpn.size()).
std::vector<uint8_t> CpuFilterForward(const RawForwardBitmap& data, const ExprSpec& expr, uint32_t num_threads);

}  // namespace filter_cmp
