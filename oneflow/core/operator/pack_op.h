#ifndef ONEFLOW_CORE_OPERATOR_PACK_OP_H_
#define ONEFLOW_CORE_OPERATOR_PACK_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

class PackOp final : public Operator {
 public:
  OF_DISALLOW_COPY(PackOp);
  PackOp() = default;
  ~PackOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override { return op_conf().pack_conf(); }
  LogicalNode* NewProperLogicalNode() { return new PackForwardLogicalNode; }
  void InferOutBlobTimeShape(std::function<const Shape*(const std::string&)> GetTimeShape4BnInOp,
                             const ParallelContext* parallel_ctx, Shape* time_shape) const override;

  bool NeedInBlobWhenBackward() const override { return true; }
  bool NeedOutBlobWhenBackward() const override { return false; }
  int32_t GetPackNum(int64_t parallel_num) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_PACK_OP_H_
