#ifndef ONEFLOW_CORE_OPERATOR_BASIC_RNN_OP_H_
#define ONEFLOW_CORE_OPERATOR_BASIC_RNN_OP_H_

#include "oneflow/core/operator/recurrent_op.h"

namespace oneflow {

class BasicRnnOp final : public RecurrentOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BasicRnnOp);
  BasicRnnOp() = default;
  ~BasicRnnOp() = default;
  const PbMessage& GetSpecialConf() const override;

 private:
  void VirtualInitFromOpConf() override;
  void VirtualInferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_BASIC_RNN_OP_H_
