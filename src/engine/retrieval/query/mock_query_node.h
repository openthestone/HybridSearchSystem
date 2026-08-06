#pragma once
#include "src/utils/logger.h"

namespace NpuRetrieval {
class MockQueryNode {
   public:
    MockQueryNode() = default;
    virtual ~MockQueryNode() = default;
};

using QueryNode = MockQueryNode;
}  // namespace NpuRetrieval
