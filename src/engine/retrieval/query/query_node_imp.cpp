#include "query_node_imp.h"
#include "src/utils/logger.h"

namespace NpuRetrieval {
// AND and OR emit the same post-order layout and differ only in the op code: recurse into the
// non-term children first (each pushes a sub-result onto the filter stack), then append the direct
// term children as postings, then emit [op, postingNum, stackNum].
static bool EmitAndOrOp(const std::vector<std::unique_ptr<QueryNode>>& children, FilterOpType opType,
                        const DataTable& dataTable, std::vector<uint32_t>& postExpr,
                        std::vector<std::vector<uint8_t>*>& postingTypes,
                        std::vector<std::vector<uint16_t>*>& postingWeights,
                        std::vector<std::vector<uint8_t*>*>& postingDeviceAddrs, uint32_t& opNum) {
    uint32_t postingNum = 0;
    uint32_t stackNum = 0;
    for (const std::unique_ptr<QueryNode>& child : children) {
        if (child == nullptr) {
            LOG_WARN("child is null");
            continue;
        }
        if (child->GetNodeType() == QueryNodeType::TermNode) {
            continue;
        }
        if (child->GetPostOrderExpression(dataTable, postExpr, postingTypes, postingWeights, postingDeviceAddrs,
                                          opNum)) {
            ++stackNum;
        }
    }

    for (const std::unique_ptr<QueryNode>& child : children) {
        if (child == nullptr) {
            LOG_WARN("child is null");
            continue;
        }
        if (child->GetNodeType() != QueryNodeType::TermNode) {
            continue;
        }
        if (child->GetPostOrderExpression(dataTable, postExpr, postingTypes, postingWeights, postingDeviceAddrs,
                                          opNum)) {
            ++postingNum;
        }
    }
    postExpr.emplace_back(static_cast<uint32_t>(opType));
    postExpr.emplace_back(postingNum);
    postExpr.emplace_back(stackNum);
    opNum += 1;
    return true;
}

bool AndNode::GetPostOrderExpression(const NpuRetrieval::DataTable& dataTable, std::vector<uint32_t>& postExpr,
                                     std::vector<std::vector<uint8_t>*>& postingTypes,
                                     std::vector<std::vector<uint16_t>*>& postingWeights,
                                     std::vector<std::vector<uint8_t*>*>& postingDeviceAddrs, uint32_t& opNum) const {
    return EmitAndOrOp(m_children, FilterOpType::AND, dataTable, postExpr, postingTypes, postingWeights,
                       postingDeviceAddrs, opNum);
}

bool OrNode::GetPostOrderExpression(const NpuRetrieval::DataTable& dataTable, std::vector<uint32_t>& postExpr,
                                    std::vector<std::vector<uint8_t>*>& postingTypes,
                                    std::vector<std::vector<uint16_t>*>& postingWeights,
                                    std::vector<std::vector<uint8_t*>*>& postingDeviceAddrs, uint32_t& opNum) const {
    return EmitAndOrOp(m_children, FilterOpType::OR, dataTable, postExpr, postingTypes, postingWeights,
                       postingDeviceAddrs, opNum);
}

bool NotNode::GetPostOrderExpression(const NpuRetrieval::DataTable& dataTable, std::vector<uint32_t>& postExpr,
                                     std::vector<std::vector<uint8_t>*>& postingTypes,
                                     std::vector<std::vector<uint16_t>*>& postingWeights,
                                     std::vector<std::vector<uint8_t*>*>& postingDeviceAddrs, uint32_t& opNum) const {
    if (m_children.size() != 1) {
        LOG_ERROR("invalid child node size of not node, size is " << m_children.size());
        return false;
    }
    const std::unique_ptr<QueryNode>& child = m_children[0];
    if (child == nullptr) {
        LOG_WARN("child is null");
        return false;
    }
    uint32_t postingNum = 0;
    uint32_t stackNum = 0;
    QueryNodeType nodeType = child->GetNodeType();
    if (!child->GetPostOrderExpression(dataTable, postExpr, postingTypes, postingWeights, postingDeviceAddrs, opNum)) {
        LOG_ERROR("get subnode post order epression failed");
        return false;
    }
    if (nodeType == QueryNodeType::TermNode) {
        postingNum = 1;
    } else {
        stackNum = 1;
    }

    postExpr.emplace_back(static_cast<uint32_t>(FilterOpType::NOT));
    postExpr.emplace_back(postingNum);
    postExpr.emplace_back(stackNum);
    opNum += 1;
    return true;
}

TermNode::TermNode(const std::string& fieldName, uint64_t token) {
    m_fieldName = fieldName;
    m_token = token;
}

void TermNode::AdjustNode(const std::unordered_set<std::string>& postingFields) {
    LOG_TRACE("TermNode no need AdjustNode, postingFields size:" << postingFields.size());
    return;
}

bool TermNode::IsNodeValid(const std::unordered_set<std::string>& postingFields) const {
    if (postingFields.find(m_fieldName) == postingFields.end()) {
        LOG_WARN("field[" << m_fieldName << "] not exist in index, will be ignored.");
        return false;
    }
    return true;
}

bool TermNode::GetPostOrderExpression(const NpuRetrieval::DataTable& dataTable, std::vector<uint32_t>& postExpr,
                                      std::vector<std::vector<uint8_t>*>& postingTypes,
                                      std::vector<std::vector<uint16_t>*>& postingWeights,
                                      std::vector<std::vector<uint8_t*>*>& postingDeviceAddrs, uint32_t& opNum) const {
    LOG_TRACE("current opnum:" << opNum << " postExpr size:" << postExpr.size());
    return AppendPostings(dataTable, postingTypes, postingWeights, postingDeviceAddrs);
}

bool TermNode::AppendPostings(const DataTable& dataTable, std::vector<std::vector<uint8_t>*>& postingTypes,
                              std::vector<std::vector<uint16_t>*>& postingWeights,
                              std::vector<std::vector<uint8_t*>*>& postingDeviceAddrs) const {
    PostingFieldData* fieldData = nullptr;
    if (!dataTable.GetPostingFieldData(m_fieldName, fieldData)) {
        LOG_ERROR("get posting field data failed, field name:" << m_fieldName);
        return false;
    }
    // AdjustNode has already removed invalid target fields, so fieldData should not be null here.
    if (fieldData == nullptr) {
        LOG_ERROR("fieldData is null ");
        return false;
    }
    bool tokenExist = true;
    uint8_t* postingAddr = fieldData->GetData(m_token, tokenExist);
    if (postingAddr == nullptr) {
        LOG_DEBUG("m_token not found");
        return false;
    }
    std::vector<uint8_t>* postingType = fieldData->GetPostingTypes(m_token);
    if (postingType == nullptr) {
        LOG_DEBUG("postingType is nullptr");
        return false;
    }
    postingTypes.emplace_back(postingType);
    // Null keeps postingWeights the same length as postingTypes, which PrepareExpr checks and emit
    // indexes in lockstep.
    postingWeights.emplace_back(PostingWeightsWanted() ? fieldData->GetPostingWeights(m_token) : nullptr);

    std::vector<uint8_t*>* postingDeviceAddr = fieldData->GetPostingDeviceAddrs(m_token);
    if (postingDeviceAddr == nullptr) {
        LOG_DEBUG("postingDeviceAddr is nullptr");
        return false;
    }
    postingDeviceAddrs.emplace_back(postingDeviceAddr);
    return true;
}

TermsNode::TermsNode(const std::string& fieldName, std::unique_ptr<std::unordered_set<uint64_t>> token) {
    m_fieldName = fieldName;
    m_token = std::move(token);
}

void TermsNode::AdjustNode(const std::unordered_set<std::string>& postingFields) {
    LOG_TRACE("TermsNode no need AdjustNode, postingFields size:" << postingFields.size());
    return;
}

bool TermsNode::IsNodeValid(const std::unordered_set<std::string>& postingFields) const {
    if (postingFields.find(m_fieldName) == postingFields.end()) {
        LOG_WARN("field[" << m_fieldName << "] not exist in index, will be ignored.");
        return false;
    }
    return true;
}

bool TermsNode::GetPostOrderExpression(const NpuRetrieval::DataTable& dataTable, std::vector<uint32_t>& postExpr,
                                       std::vector<std::vector<uint8_t>*>& postingTypes,
                                       std::vector<std::vector<uint16_t>*>& postingWeights,
                                       std::vector<std::vector<uint8_t*>*>& postingDeviceAddrs, uint32_t& opNum) const {
    uint32_t postingNum = 0;
    uint32_t stackNum = 0;
    size_t postingBase = postingDeviceAddrs.size();
    if (!AppendPostings(dataTable, postingTypes, postingWeights, postingDeviceAddrs)) {
        LOG_ERROR("append posting failed.");
        return false;
    }
    postingNum = postingDeviceAddrs.size() - postingBase;
    if (postingNum == 0) {
        LOG_ERROR("field:" << m_fieldName << " has no valid token.");
        return false;
    }
    postExpr.emplace_back(static_cast<uint32_t>(FilterOpType::OR));
    postExpr.emplace_back(postingNum);
    postExpr.emplace_back(stackNum);
    opNum += 1;
    return true;
}

bool TermsNode::AppendPostings(const DataTable& dataTable, std::vector<std::vector<uint8_t>*>& postingTypes,
                               std::vector<std::vector<uint16_t>*>& postingWeights,
                               std::vector<std::vector<uint8_t*>*>& postingDeviceAddrs) const {
    PostingFieldData* fieldData = nullptr;
    if (!dataTable.GetPostingFieldData(m_fieldName, fieldData)) {
        LOG_ERROR("get posting field data failed, field name:" << m_fieldName);
        return false;
    }
    if (fieldData == nullptr || m_token == nullptr) {
        LOG_ERROR("fieldData is null ");
        return false;
    }
    bool addedPosting = false;
    for (const uint64_t& token : *m_token) {
        bool tokenExist = true;
        uint8_t* postingAddr = fieldData->GetData(token, tokenExist);
        if (postingAddr == nullptr) {
            LOG_DEBUG("m_token not found");
            continue;
        }
        // once a terms node has added one posting, don't add the all-zero posting again
        if ((!tokenExist) && addedPosting) {
            continue;
        }
        std::vector<uint8_t>* postingType = fieldData->GetPostingTypes(token);
        if (postingType == nullptr) {
            LOG_DEBUG("postingType is nullptr");
            continue;
        }
        postingTypes.emplace_back(postingType);
        postingWeights.emplace_back(PostingWeightsWanted() ? fieldData->GetPostingWeights(token) : nullptr);
        std::vector<uint8_t*>* postingDeviceAddr = fieldData->GetPostingDeviceAddrs(token);
        if (postingDeviceAddr == nullptr) {
            LOG_DEBUG("postingDeviceAddr is nullptr");
            return false;
        }
        postingDeviceAddrs.emplace_back(postingDeviceAddr);
        addedPosting = true;
    }
    return true;
}

bool OrTermsNode::GetPostOrderExpression(const NpuRetrieval::DataTable& dataTable, std::vector<uint32_t>& postExpr,
                                         std::vector<std::vector<uint8_t>*>& postingTypes,
                                         std::vector<std::vector<uint16_t>*>& postingWeights,
                                         std::vector<std::vector<uint8_t*>*>& postingDeviceAddrs,
                                         uint32_t& opNum) const {
    uint32_t postingNum = 0;
    uint32_t stackNum = 0;
    size_t postingBase = postingDeviceAddrs.size();
    if (!AppendPostings(dataTable, postingTypes, postingWeights, postingDeviceAddrs)) {
        LOG_ERROR("append posting failed.");
        return false;
    }
    postingNum = postingDeviceAddrs.size() - postingBase;
    if (postingNum == 0) {
        LOG_ERROR("has no valid token.");
        return false;
    }
    postExpr.emplace_back(static_cast<uint32_t>(FilterOpType::OR));
    postExpr.emplace_back(postingNum);
    postExpr.emplace_back(stackNum);
    opNum += 1;
    return true;
}

bool OrTermsNode::AppendPostings(const DataTable& dataTable, std::vector<std::vector<uint8_t>*>& postingTypes,
                                 std::vector<std::vector<uint16_t>*>& postingWeights,
                                 std::vector<std::vector<uint8_t*>*>& postingDeviceAddrs) const {
    for (const std::unique_ptr<QueryNode>& child : m_children) {
        if (child == nullptr) {
            LOG_WARN("child is null");
            continue;
        }
        QueryLeafNode* leaf = dynamic_cast<QueryLeafNode*>(child.get());
        if (leaf == nullptr) {
            LOG_WARN("leaf is null");
            continue;
        }
        QueryNodeType nodeType = leaf->GetNodeType();
        if (nodeType != QueryNodeType::TermNode && nodeType != QueryNodeType::TermsNode) {
            LOG_WARN("or_terms only support term/terms, skiped, unsupport type: " << static_cast<uint32_t>(nodeType));
            continue;
        }
        if (!leaf->AppendPostings(dataTable, postingTypes, postingWeights, postingDeviceAddrs)) {
            LOG_WARN("Append Postings failed");
            continue;
        }
    }
    return true;
}

bool ConjunctionNode::GetPostOrderExpression(const NpuRetrieval::DataTable& dataTable, std::vector<uint32_t>& postExpr,
                                             std::vector<std::vector<uint8_t>*>& postingTypes,
                                             std::vector<std::vector<uint16_t>*>& postingWeights,
                                             std::vector<std::vector<uint8_t*>*>& postingDeviceAddrs,
                                             uint32_t& opNum) const {
    std::vector<uint32_t> groupSizes;
    groupSizes.reserve(m_children.size());
    for (const std::unique_ptr<QueryNode>& child : m_children) {
        if (child == nullptr) {
            LOG_WARN("child is null");
            continue;
        }
        QueryLeafNode* leaf = dynamic_cast<QueryLeafNode*>(child.get());
        if (leaf == nullptr) {
            LOG_WARN("leaf is null");
            continue;
        }

        QueryNodeType nodeType = leaf->GetNodeType();
        if (nodeType != QueryNodeType::TermNode && nodeType != QueryNodeType::TermsNode &&
            nodeType != QueryNodeType::OrTermsNode) {
            LOG_WARN("conjunction only support term/terms/or_terms, skiped, unsupport type: "
                     << static_cast<uint32_t>(nodeType));
            continue;
        }
        size_t postingBase = postingDeviceAddrs.size();
        if (!leaf->AppendPostings(dataTable, postingTypes, postingWeights, postingDeviceAddrs)) {
            LOG_WARN("Append Postings failed");
            continue;
        }
        uint32_t postingNum = postingDeviceAddrs.size() - postingBase;
        if (postingNum > 0) {
            groupSizes.emplace_back(postingNum);
        }
    }
    if (groupSizes.empty()) {
        LOG_ERROR("Conjunction has no valid token.");
        return false;
    }

    postExpr.emplace_back(static_cast<uint32_t>(FilterOpType::CONJUNCTION));
    postExpr.emplace_back(groupSizes.size());
    for (const uint32_t num : groupSizes) {
        postExpr.emplace_back(num);
    }
    opNum += 1;
    return true;
}
}  // namespace NpuRetrieval
