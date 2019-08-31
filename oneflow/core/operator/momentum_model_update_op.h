#ifndef ONEFLOW_CORE_OPERATOR_MOMENTUM_MODEL_UPDATE_OP_H_
#define ONEFLOW_CORE_OPERATOR_MOMENTUM_MODEL_UPDATE_OP_H_

#include "oneflow/core/operator/normal_model_update_op.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

class MomentumModelUpdateOp final : public NormalModelUpdtOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MomentumModelUpdateOp);
  MomentumModelUpdateOp() = default;
  ~MomentumModelUpdateOp() = default;

  const PbMessage& GetCustomizedConf() const override;
  LogicalNode* NewProperLogicalNode() const override { return new OptimizerLogicalNode; }

 private:
  void MdUpdtVirtualInitFromOpConf() override;
  Maybe<void> MdUpdtVirtualInferBlobDescs(
      std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override;
  const HashSet<std::string> AlwaysBroadcastParallelBns() const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_MOMENTUM_MODEL_UPDATE_OP_H_
