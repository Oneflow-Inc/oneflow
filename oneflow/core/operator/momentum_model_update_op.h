#ifndef ONEFLOW_CORE_OPERATOR_MOMENTUM_MODEL_UPDATE_OP_H_
#define ONEFLOW_CORE_OPERATOR_MOMENTUM_MODEL_UPDATE_OP_H_

#include "oneflow/core/operator/model_update_op.h"

namespace oneflow {

class MomentumModelUpdateOp final : public ModelUpdtOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MomentumModelUpdateOp);
  MomentumModelUpdateOp() = default;
  ~MomentumModelUpdateOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetSpecialConf() const override;
  void InferBlobDesc4FwBlobs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      ParallelPolicy policy, int64_t parallel_id,
      int64_t parallel_num) override;

 private:
  std::string ibn2lbn(const std::string& input_bn) const override {
    return kPackedBlobName;
  }
  std::string obn2lbn(const std::string& output_bn) const override {
    return kPackedBlobName;
  }
  std::string mtbn2lbn(const std::string& output_bn) const override {
    return kPackedBlobName;
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_MOMENTUM_MODEL_UPDATE_OP_H_
