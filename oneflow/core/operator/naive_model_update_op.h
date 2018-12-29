#ifndef ONEFLOW_CORE_OPERATOR_NAIVE_MODEL_UPDATE_OP_H_
#define ONEFLOW_CORE_OPERATOR_NAIVE_MODEL_UPDATE_OP_H_

#include "oneflow/core/operator/normal_model_update_op.h"

namespace oneflow {

class NaiveModelUpdateOp final : public NormalModelUpdtOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NaiveModelUpdateOp);
  NaiveModelUpdateOp() = default;
  ~NaiveModelUpdateOp() = default;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_NAIVE_MODEL_UPDATE_OP_H_
