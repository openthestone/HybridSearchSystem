#pragma once
#include "query_node.h"
#include <unordered_set>

namespace NpuRetrieval {

const std::string AND_NODE = "and";
const std::string OR_NODE = "or";
const std::string NOT_NODE = "not";
const std::string TERM_NODE = "term";
const std::string TERMS_NODE = "terms";
const std::string OR_TERMS_NODE = "or_terms";
const std::string CONJUNCTION_NODE = "conjunction";

class AndNode : public QueryNode {
   public:
    bool GetPostOrderExpression(const DataTable& dataTable, std::vector<uint32_t>& postExpr,
                                std::vector<std::vector<uint8_t>*>& postingTypes,
                                std::vector<std::vector<uint8_t*>*>& postingDeviceAddrs,
                                uint32_t& opNum) const override;
    QueryNodeType GetNodeType() const override {
        return QueryNodeType::AndNode;
    };

    std::string GetName() const override {
        return AND_NODE;
    }
};

class OrNode : public QueryNode {
   public:
    bool GetPostOrderExpression(const DataTable& dataTable, std::vector<uint32_t>& postExpr,
                                std::vector<std::vector<uint8_t>*>& postingTypes,
                                std::vector<std::vector<uint8_t*>*>& postingDeviceAddrs,
                                uint32_t& opNum) const override;
    QueryNodeType GetNodeType() const override {
        return QueryNodeType::OrNode;
    };

    std::string GetName() const override {
        return OR_NODE;
    }
};

class NotNode : public QueryNode {
   public:
    bool GetPostOrderExpression(const DataTable& dataTable, std::vector<uint32_t>& postExpr,
                                std::vector<std::vector<uint8_t>*>& postingTypes,
                                std::vector<std::vector<uint8_t*>*>& postingDeviceAddrs,
                                uint32_t& opNum) const override;
    QueryNodeType GetNodeType() const override {
        return QueryNodeType::NotNode;
    };

    std::string GetName() const override {
        return NOT_NODE;
    }
};

class TermNode : public QueryLeafNode {
   public:
    TermNode(const std::string& fieldName, uint64_t token);

    void AdjustNode(const std::unordered_set<std::string>& postingFields) override;
    bool IsNodeValid(const std::unordered_set<std::string>& postingFields) const override;
    bool GetPostOrderExpression(const DataTable& dataTable, std::vector<uint32_t>& postExpr,
                                std::vector<std::vector<uint8_t>*>& postingTypes,
                                std::vector<std::vector<uint8_t*>*>& postingDeviceAddrs,
                                uint32_t& opNum) const override;
    QueryNodeType GetNodeType() const override {
        return QueryNodeType::TermNode;
    };
    bool AppendPostings(const DataTable& dataTable, std::vector<std::vector<uint8_t>*>& postingTypes,
                        std::vector<std::vector<uint8_t*>*>& postingDeviceAddrs) const override;

    std::string GetName() const override {
        return TERM_NODE + ":" + m_fieldName + ":" + std::to_string(m_token);
    }

   private:
    std::string m_fieldName;
    uint64_t m_token;
};

class TermsNode : public QueryLeafNode {
   public:
    TermsNode(const std::string& fieldName, std::unique_ptr<std::unordered_set<uint64_t>> token);

    void AdjustNode(const std::unordered_set<std::string>& postingFields) override;
    bool IsNodeValid(const std::unordered_set<std::string>& postingFields) const override;
    bool GetPostOrderExpression(const DataTable& dataTable, std::vector<uint32_t>& postExpr,
                                std::vector<std::vector<uint8_t>*>& postingTypes,
                                std::vector<std::vector<uint8_t*>*>& postingDeviceAddrs,
                                uint32_t& opNum) const override;
    QueryNodeType GetNodeType() const override {
        return QueryNodeType::TermsNode;
    };

    bool AppendPostings(const DataTable& dataTable, std::vector<std::vector<uint8_t>*>& postingTypes,
                        std::vector<std::vector<uint8_t*>*>& postingDeviceAddrs) const override;

    std::string GetName() const override {
        std::string token;
        if (m_token != nullptr) {
            for (auto& one : *m_token) {
                token = token + std::to_string(one) + ",";
            }
        }
        return TERMS_NODE + ":" + m_fieldName + ":" + token;
    }

   private:
    std::string m_fieldName;
    std::unique_ptr<std::unordered_set<uint64_t>> m_token;
};

class OrTermsNode : public QueryLeafNode {
   public:
    bool GetPostOrderExpression(const DataTable& dataTable, std::vector<uint32_t>& postExpr,
                                std::vector<std::vector<uint8_t>*>& postingTypes,
                                std::vector<std::vector<uint8_t*>*>& postingDeviceAddrs,
                                uint32_t& opNum) const override;
    QueryNodeType GetNodeType() const override {
        return QueryNodeType::OrTermsNode;
    };
    bool AppendPostings(const DataTable& dataTable, std::vector<std::vector<uint8_t>*>& postingTypes,
                        std::vector<std::vector<uint8_t*>*>& postingDeviceAddrs) const override;

    std::string GetName() const override {
        return OR_TERMS_NODE;
    }
};

class ConjunctionNode : public QueryNode {
   public:
    ConjunctionNode() = default;

    ~ConjunctionNode() override = default;
    bool GetPostOrderExpression(const DataTable& dataTable, std::vector<uint32_t>& postExpr,
                                std::vector<std::vector<uint8_t>*>& postingTypes,
                                std::vector<std::vector<uint8_t*>*>& postingDeviceAddrs,
                                uint32_t& opNum) const override;
    QueryNodeType GetNodeType() const override {
        return QueryNodeType::ConjunctionNode;
    };

    std::string GetName() const override {
        return CONJUNCTION_NODE;
    }
};
}  // namespace NpuRetrieval
