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
  bool NeedInBlobWhenBackward() const override { return false; }
  bool NeedOutBlobWhenBackward() const override { return false; }
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return false; }
  LogicalBlobId ibn2lbi(const std::string& input_bn) const override {
    const ReduceSumOpConf& conf = op_conf().reduce_sum_conf();
    if (conf.has_in_sys()) {
      CHECK_EQ(conf.axis_size(), 1);
      CHECK_EQ(conf.axis().Get(0), 0);
      return conf.in_sys();
    } else if (conf.has_in()) {
      return GenLogicalBlobId(conf.in());
    } else {
      UNIMPLEMENTED();
    }
    return LogicalBlobId();
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_REDUCE_SUM_OP_H_
