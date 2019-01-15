#ifndef ONEFLOW_CORE_OPERATOR_REDUCE_GATHER_OP_H_
#define ONEFLOW_CORE_OPERATOR_REDUCE_GATHER_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ReduceGatherOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceGatherOp);
  ReduceGatherOp() = default;
  ~ReduceGatherOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;

 private:
  void InferOutBlobModelSplitAxis(std::function<int64_t*(const std::string&)> ModelSplitAxis4BnInOp,
                                  std::function<int64_t(const std::string&)> ShapeNumAxes4BnInOp,
                                  const ParallelContext* parallel_context) const override {
    UNIMPLEMENTED();
  }

  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext*, KernelConf*) const override;
  LogicalBlobId ibn2lbi(const std::string& input_bn) const override { return GenPackedLbi(); }
  LogicalBlobId obn2lbi(const std::string& output_bn) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_REDUCE_GATHER_OP_H_
