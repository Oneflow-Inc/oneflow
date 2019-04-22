#ifndef ONEFLOW_CORE_OPERATOR_NAIVE_MODEL_UPDATE_OP_H_
#define ONEFLOW_CORE_OPERATOR_NAIVE_MODEL_UPDATE_OP_H_

#include "oneflow/core/operator/normal_model_update_op.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

class NaiveModelUpdateOp final : public NormalModelUpdtOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NaiveModelUpdateOp);
  NaiveModelUpdateOp() = default;
  ~NaiveModelUpdateOp() = default;
  const PbMessage& GetCustomizedConf() const override;

  LogicalNode* NewProperLogicalNode() const override { return new OptimizerLogicalNode; }

  const HashSet<std::string> AlwaysBroadcastParallelBns() const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_NAIVE_MODEL_UPDATE_OP_H_
