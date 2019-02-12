#ifndef ONEFLOW_CORE_OPERATOR_NCCL_REDUCE_SCATTER_OP_H_
#define ONEFLOW_CORE_OPERATOR_NCCL_REDUCE_SCATTER_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class NcclReduceScatterOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclReduceScatterOp);
  NcclReduceScatterOp() = default;
  ~NcclReduceScatterOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { UNIMPLEMENTED(); }
  void InferOutputBlobSbpInferHint(
      std::function<SbpInferHint*(const std::string&)> SbpInferHint4BnInOp,
      const ParallelContext* parallel_context) const override {
    UNIMPLEMENTED();
  }
  LogicalBlobId ibn2lbi(const std::string& input_bn) const override { return GenPackedLbi(); }
  LogicalBlobId obn2lbi(const std::string& output_bn) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_NCCL_REDUCE_SCATTER_OP_H_
