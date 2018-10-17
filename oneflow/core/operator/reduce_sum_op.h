#ifndef ONEFLOW_CORE_OPERATOR_REDUCE_SUM_OP_H_
#define ONEFLOW_CORE_OPERATOR_REDUCE_SUM_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ReduceSumOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceSumOp);
  ReduceSumOp() = default;
  ~ReduceSumOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;

 private:
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override;
  int32_t GetCorrectAxis(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp) const;
  LogicalBlobId ibn2lbi(const std::string& input_bn) const override {
    const ReduceSumOpConf& conf = op_conf().reduce_sum_conf();
    if (conf.has_in_sys()) {
      CHECK_EQ(conf.axis(), 0);
      return conf.in_sys();
    } else if (conf.has_in()) {
      CHECK_GE(conf.axis(), 1);
      return GenLogicalBlobId(conf.in());
    } else {
      UNIMPLEMENTED();
    }
    return LogicalBlobId();
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_REDUCE_SUM_OP_H_
