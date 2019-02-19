#ifndef ONEFLOW_CORE_OPERATOR_RMSPROP_MODEL_UPDATE_OP_H_
#define ONEFLOW_CORE_OPERATOR_RMSPROP_MODEL_UPDATE_OP_H_

#include "oneflow/core/operator/normal_model_update_op.h"

namespace oneflow {

class RMSPropModelUpdateOp final : public NormalModelUpdtOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RMSPropModelUpdateOp);
  RMSPropModelUpdateOp() = default;
  ~RMSPropModelUpdateOp() = default;

 private:
  void MdUpdtVirtualInitFromOpConf() override;
  void MdUpdtVirtualInferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext* parallel_ctx) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_RMSPROP_MODEL_UPDATE_OP_H_
