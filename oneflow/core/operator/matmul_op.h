#ifndef ONEFLOW_CORE_OPERATOR_MATMUL_OP_H_
#define ONEFLOW_CORE_OPERATOR_MATMUL_OP_H_
#include "oneflow/core/operator/operator.h"
namespace oneflow {

class MatmulOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MatmulOp);
  MatmulOp() = default;
  ~MatmulOp() = default;
  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  int32_t ModelSplitAxis() const override { return 1; }
  int32_t MaxModelSplitNum() const override { return op_conf().matmul_conf().units(); }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_MATMUL_OP_H_
