#ifndef ONEFLOW_CORE_OPERATOR_ACCUMULATE_OP_H_
#define ONEFLOW_CORE_OPERATOR_ACCUMULATE_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class AccumulateOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AccumulateOp);
  AccumulateOp() = default;
  ~AccumulateOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetSpecialConf() const override;

  void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override {}

 private:
  std::string ibn2lbn(const std::string& input_bn) const override {
    return kPackedBlobName;
  }
  std::string obn2lbn(const std::string& output_bn) const override {
    return kPackedBlobName;
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_ACCUMULATE_OP_H_
