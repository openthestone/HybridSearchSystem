#include "query_node.h"
#include <string>
#include "src/utils/logger.h"
namespace NpuRetrieval {
const uint32_t kIndentWidth = 4;

void QueryNode::AdjustNode(const std::unordered_set<std::string>& postingFields) {
    // recurse into each child first, then drop the ones that end up invalid
    for (auto iter = m_children.begin(); iter != m_children.end();) {
        (*iter)->AdjustNode(postingFields);
        if ((*iter)->IsNodeValid(postingFields)) {
            ++iter;
        } else {
            iter = m_children.erase(iter);
        }
    }
}

bool QueryNode::IsNodeValid(const std::unordered_set<std::string>& postingFields) const {
    LOG_TRACE("judge node validity, postingFields size:" << postingFields.size());
    return !m_children.empty();
}

void QueryNode::AddChild(std::unique_ptr<QueryNode>& child) {
    if (child == nullptr) {
        LOG_ERROR("child is nullptr.");
        return;
    }
    m_children.emplace_back(std::move(child));
}

void QueryNode::PrintTree(uint32_t indent) const {
    LOG_DEBUG("" << std::string(indent, ' ') << "└── " << GetName());
    for (const auto& child : m_children) {
        if (child != nullptr) {
            child->PrintTree(indent + kIndentWidth);  // indent for alignment
        }
    }
}
}  // namespace NpuRetrieval
