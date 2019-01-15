#ifndef ONEFLOW_CORE_OPERATOR_REDUCE_SPLIT_OP_H_
#define ONEFLOW_CORE_OPERATOR_REDUCE_SPLIT_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ReduceSplitOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceSplitOp);
  ReduceSplitOp() = default;
  ~ReduceSplitOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override {}

 private:
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext*, KernelConf*) const override;
  void InferOutBlobModelSplitAxis(std::function<int64_t*(const std::string&)> ModelSplitAxis4BnInOp,
                                  std::function<int64_t(const std::string&)> ShapeNumAxes4BnInOp,
                                  const ParallelContext* parallel_context) const override {
    UNIMPLEMENTED();
  }
  LogicalBlobId ibn2lbi(const std::string& input_bn) const override { return GenPackedLbi(); }
  LogicalBlobId obn2lbi(const std::string& output_bn) const override { return GenPackedLbi(); }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_REDUCE_SPLIT_OP_H_
