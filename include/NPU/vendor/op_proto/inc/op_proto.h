#ifndef OP_PROTO_H_
#define OP_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {

REG_OP(MaskFilter)
    .INPUT(scores, ge::TensorType::ALL())
    .INPUT(masks, ge::TensorType::ALL())
    .INPUT(output_buf, ge::TensorType::ALL())
    .INPUT(index_buf, ge::TensorType::ALL())
    .INPUT(count_buf, ge::TensorType::ALL())
    .INPUT(task_buf, ge::TensorType::ALL())
    .OUTPUT(result, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(MaskFilter);

IMPLEMT_COMMON_INFERFUNC(MaskFilterInferShape) {
    auto desc = op.GetOutputDescByName("result");
    desc.SetShape(ge::Shape({1}));
    desc.SetDataType(DT_FLOAT);
    (void)op.UpdateOutputDesc("result", desc);
    return GRAPH_SUCCESS;
}

COMMON_INFER_FUNC_REG(MaskFilter, MaskFilterInferShape);

} // namespace ge

#endif // OP_PROTO_H_
