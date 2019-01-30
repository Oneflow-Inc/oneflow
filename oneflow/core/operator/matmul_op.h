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
  bool NeedOutBlobWhenBackward() const override { return false; }
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  void InferBwBufBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                           const ParallelContext*) const override;

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override;
  void GetOpParallelSignatures(std::vector<OpParallelSignature>*) const override;
  void InferOutputBlobModelSplitAxis(
      std::function<int32_t*(const std::string&)> ModelSplitAxis4BnInOp,
      std::function<int32_t(const std::string&)> ShapeNumAxes4BnInOp,
      const ParallelContext* parallel_context) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_MATMUL_OP_H_
