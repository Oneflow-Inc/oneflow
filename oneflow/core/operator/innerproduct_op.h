#ifndef ONEFLOW_CORE_OPERATOR_INNERPRODUCT_OP_H_
#define ONEFLOW_CORE_OPERATOR_INNERPRODUCT_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class InnerProductOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(InnerProductOp);
  InnerProductOp() = default;
  ~InnerProductOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetSpecialConf() const override;
  void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override;
  int32_t ModelSplitAxis() const override { return 1; }
  int32_t MaxModelSplitNum() const override {
    return op_conf().innerproduct_conf().out_num();
  }

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_INNERPRODUCT_OP_H_
