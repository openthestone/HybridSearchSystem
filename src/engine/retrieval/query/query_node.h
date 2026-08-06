#pragma once

#include <iostream>
#include <shared_mutex>
#include <vector>
#include "src/full_recall/index/data_table.h"

namespace NpuRetrieval {
enum class QueryNodeType {
    AndNode = 1,
    OrNode,
    NotNode,
    TermNode,
    TermsNode,
    OrTermsNode,
    ConjunctionNode,
};
enum FilterOpType : uint32_t {
    CONJUNCTION = 0,
    AND = 1,
    OR = 2,
    NOT = 3,
};
class QueryNode {
   public:
    QueryNode() = default;

    virtual ~QueryNode() = default;
    // re-adjust the query tree, pruning invalid nodes
    virtual void AdjustNode(const std::unordered_set<std::string>& postingFields);
    // whether this node is valid
    virtual bool IsNodeValid(const std::unordered_set<std::string>& postingFields) const;

    virtual bool GetPostOrderExpression(const DataTable& dataTable, std::vector<uint32_t>& postExpr,
                                        std::vector<std::vector<uint8_t>*>& postingTypes,
                                        std::vector<std::vector<uint8_t*>*>& postingDeviceAddrs,
                                        uint32_t& opNum) const = 0;

    void AddChild(std::unique_ptr<QueryNode>& child);

    virtual QueryNodeType GetNodeType() const = 0;

    virtual std::string GetName() const = 0;

    uint32_t GetChildrenNum() const {
        return m_children.size();
    }

    void PrintTree(uint32_t indent = 0) const;

    mutable std::shared_mutex m_adjustNodeMutex;  // read-write lock
    volatile bool m_isAdjustNodeDone = false;

   protected:
    std::vector<std::unique_ptr<QueryNode>> m_children;
};

class QueryLeafNode : public QueryNode {
   public:
    virtual bool AppendPostings(const DataTable& dataTable, std::vector<std::vector<uint8_t>*>& postingTypes,
                                std::vector<std::vector<uint8_t*>*>& postingDeviceAddrs) const = 0;
};
}  // namespace NpuRetrieval
