#ifndef ONEFLOW_CORE_OPERATOR_CLONE_OP_H_
#define ONEFLOW_CORE_OPERATOR_CLONE_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class CloneOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CloneOp);
  CloneOp() = default;
  ~CloneOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetSpecialConf() const override;
  void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const;

 private:
  std::string ibn2lbn(const std::string& input_bn) const override {
    return GetStringFromSpecialConf("lbn");
  }
  std::string obn2lbn(const std::string& output_bn) const override {
    return op_name() + "/" + output_bn;
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CLONE_OP_H_
