#ifndef ONEFLOW_CORE_OPERATOR_BASIC_RNN_OP_H_
#define ONEFLOW_CORE_OPERATOR_BASIC_RNN_OP_H_

#include "oneflow/core/operator/recurrent_op.h"

namespace oneflow {

class BasicRnnOp final : public RecurrentOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BasicRnnOp);
  BasicRnnOp() = default;
  ~BasicRnnOp() = default;
  const PbMessage& GetCustomizedConf() const override;

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { UNIMPLEMENTED(); }
  void InferOutputBlobModelSplitAxis(
      std::function<int32_t*(const std::string&)> ModelSplitAxis4BnInOp,
      std::function<int32_t(const std::string&)> ShapeNumAxes4BnInOp,
      const ParallelContext* parallel_context) const override {
    UNIMPLEMENTED();
  }

  void VirtualInitFromOpConf();
  void VirtualInferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_BASIC_RNN_OP_H_
