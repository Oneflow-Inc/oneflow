#ifndef ONEFLOW_CORE_OPERATOR_RMSPROP_MODEL_UPDATE_OP_H_
#define ONEFLOW_CORE_OPERATOR_RMSPROP_MODEL_UPDATE_OP_H_

#include "oneflow/core/operator/model_update_op.h"

namespace oneflow {

class RMSPropModelUpdateOp final : public ModelUpdtOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RMSPropModelUpdateOp);
  RMSPropModelUpdateOp() = default;
  ~RMSPropModelUpdateOp() = default;

  const PbMessage& GetSpecialConf() const override;

 protected:
  void VirtualInitFromOpConf() override;
  void VirtualInferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override;

 private:
  std::string ibn2lbn(const std::string& input_bn) const override {
    return kPackedBlobName;
  }
  std::string obn2lbn(const std::string& output_bn) const override {
    return kPackedBlobName;
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_RMSPROP_MODEL_UPDATE_OP_H_
