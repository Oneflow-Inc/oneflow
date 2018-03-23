#ifndef ONEFLOW_CORE_OPERATOR_NORMALIZATION_MODEL_UPDATE_OP_H_
#define ONEFLOW_CORE_OPERATOR_NORMALIZATION_MODEL_UPDATE_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class NormalizationModelUpdtOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalizationModelUpdtOp);
  NormalizationModelUpdtOp() = default;
  ~NormalizationModelUpdtOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

  virtual void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) {}

 private:
  std::string ibn2lbn(const std::string& input_bn) const override {
    return kPackedBlobName;
  }
  std::string obn2lbn(const std::string& output_bn) const override {
    return kPackedBlobName;
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_NORMALIZATION_MODEL_UPDATE_OP_H_
