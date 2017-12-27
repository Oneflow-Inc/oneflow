#ifndef ONEFLOW_CORE_OPERATOR_RECURRENT_OP_H_
#define ONEFLOW_CORE_OPERATOR_RECURRENT_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class RecurrentOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RecurrentOp);
  RecurrentOp() = default;
  ~RecurrentOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetSpecialConf() const override;
  bool IsRecurrentOp() const { return true; }

  void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override;
  int32_t ModelSplitAxis() const override { return 1; }
  int32_t MaxModelSplitNum() const override {
    return op_conf().recurrent_conf().hidden_size();
  }

 private:
  std::string ibn2lbn(const std::string& input_bn) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_RECURRENT_OP_H_
