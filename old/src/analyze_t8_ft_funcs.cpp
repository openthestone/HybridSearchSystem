// =========================================================
// Expression analysis (FilterTmpNode tree)
// =========================================================
ExprStats AnalyzeExpression(const FilterExp& fexp) {
    ExprStats stats;
    if (fexp.ft_empty || fexp.ft_nodes.empty())
        return stats;

    struct Result {
        int depth;
        int or_terms;
    };
    const auto& nodes = fexp.ft_nodes;

    std::function<Result(uint32_t)> visit = [&](uint32_t node_id) -> Result {
        const FilterTmpNode& n = nodes[node_id];
        if (n.type == FilterTmpNode::LEAF) {
            ++stats.num_predicates;
            if (n.inverted)
                ++stats.num_not;
            return {1, 0};
        }
        // GROUP
        bool is_and = (n.op == FilterTmpNode::AND_GROUP);
        if (is_and)
            ++stats.num_and;
        else
            ++stats.num_or;

        int max_depth = 0;
        int total_or_terms = 0;
        for (uint32_t child_id : n.children) {
            Result cr = visit(child_id);
            max_depth = std::max(max_depth, cr.depth);
            if (!is_and) {
                if (cr.or_terms == 0)
                    ++total_or_terms;
                else
                    total_or_terms += cr.or_terms;
            }
        }

        if (!is_and && total_or_terms > stats.max_or_width)
            stats.max_or_width = total_or_terms;

        return {max_depth + 1, is_and ? 0 : total_or_terms};
    };

    Result root_r = visit(fexp.ft_root);
    stats.expr_depth = root_r.depth;
    stats.total_or_terms = root_r.or_terms;
    return stats;
}

// =========================================================
// Normalized expression string
// =========================================================
std::string GenerateNormalizedExpr(const FilterExp& fexp) {
    if (fexp.ft_empty)
        return "PASS_ALL";
    if (fexp.ft_nodes.empty())
        return "ALL_FALSE";

    const auto& nodes = fexp.ft_nodes;

    struct Result {
        std::string str;
        int precedence;  // 0=operand, 1=OR, 2=AND
    };

    std::function<Result(uint32_t)> visit = [&](uint32_t node_id) -> Result {
        const FilterTmpNode& n = nodes[node_id];
        if (n.type == FilterTmpNode::LEAF) {
            if (n.inverted)
                return {"!T" + std::to_string(n.tag_id), 0};
            return {"T" + std::to_string(n.tag_id), 0};
        }

        bool is_and = (n.op == FilterTmpNode::AND_GROUP);
        const char* op_str = is_and ? "&" : "|";
        int prec = is_and ? 2 : 1;

        std::string combined;
        auto wrap = [&](const Result& child) -> std::string {
            if (child.precedence > 0 && child.precedence < prec)
                return "(" + child.str + ")";
            return child.str;
        };

        for (size_t i = 0; i < n.children.size(); ++i) {
            Result cr = visit(n.children[i]);
            if (i > 0)
                combined += op_str;
            combined += wrap(cr);
        }

        return {combined, prec};
    };

    Result r = visit(fexp.ft_root);
    return r.str;
}

// =========================================================
// Bitmap volume (pure counting)
// =========================================================
BitmapVolumeStats ComputeBitmapVolume(const FilterExp& fexp, const Bucket& bucket) {
    BitmapVolumeStats stats;
    if (fexp.ft_empty || fexp.ft_nodes.empty())
        return stats;

    uint32_t stride = bucket.get_stride();
    if (stride == 0)
        return stats;

    uint32_t cache_line_bytes = cpu_cache_line_size > 0 ? static_cast<uint32_t>(cpu_cache_line_size) : 64;
    uint32_t block_u64 = std::max<uint32_t>(1, (cache_line_bytes / sizeof(uint64_t)) * 4);
    uint32_t num_blocks = (stride + block_u64 - 1) / block_u64;

    const auto& nodes = fexp.ft_nodes;
    uint64_t not_ops = 0, and_ops = 0, or_ops = 0;
    for (const auto& n : nodes) {
        if (n.type == FilterTmpNode::LEAF && n.inverted)
            not_ops++;
        else if (n.type == FilterTmpNode::GROUP) {
            if (n.op == FilterTmpNode::AND_GROUP)
                and_ops++;
            else
                or_ops++;
        }
    }

    stats.not_ops = not_ops * num_blocks;
    stats.and_ops = and_ops * num_blocks;
    stats.or_ops = or_ops * num_blocks;

    uint32_t block_len = std::min(block_u64, stride);
    stats.words_read = (not_ops + 2 * (and_ops + or_ops)) * block_len * num_blocks;
    stats.words_written = (not_ops + and_ops + or_ops) * block_len * num_blocks;
    return stats;
}
