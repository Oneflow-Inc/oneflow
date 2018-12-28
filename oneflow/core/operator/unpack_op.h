#ifndef ONEFLOW_CORE_OPERATOR_UNPACK_OP_H_
#define ONEFLOW_CORE_OPERATOR_UNPACK_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

class UnpackOp final : public Operator {
 public:
  OF_DISALLOW_COPY(UnpackOp);
  UnpackOp() = default;
  ~UnpackOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override { return op_conf().unpack_conf(); }
  LogicalNode* NewProperLogicalNode() { return new UnpackForwardLogicalNode; }

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  void InferOutBlobTimeShape(std::function<const Shape*(const std::string&)> GetTimeShape4BnInOp,
                             const ParallelContext* parallel_ctx, Shape* time_shape) const override;
  bool NeedInBlobWhenBackward() const override { return true; }
  bool NeedOutBlobWhenBackward() const override { return false; }
  int32_t GetUnpackNum(int64_t parallel_num) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_UNPACK_OP_H_
