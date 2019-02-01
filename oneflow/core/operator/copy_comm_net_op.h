#ifndef ONEFLOW_CORE_OPERATOR_COPY_COMM_NET_OP_H_
#define ONEFLOW_CORE_OPERATOR_COPY_COMM_NET_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class CopyCommNetOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyCommNetOp);
  CopyCommNetOp() = default;
  ~CopyCommNetOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { UNIMPLEMENTED(); }
  void InferOutputBlobLbpdHint(std::function<LbpdHint*(const std::string&)> LbpdHint4BnInOp,
                               std::function<int32_t(const std::string&)> ShapeNumAxes4BnInOp,
                               const ParallelContext* parallel_context) const override {
    UNIMPLEMENTED();
  }

  LogicalBlobId ibn2lbi(const std::string& input_bn) const override;
  LogicalBlobId obn2lbi(const std::string& output_bn) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_COPY_COMM_NET_OP_H_
