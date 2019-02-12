#ifndef ONEFLOW_CORE_OPERATOR_ACCUMULATE_OP_H_
#define ONEFLOW_CORE_OPERATOR_ACCUMULATE_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class AccumulateOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AccumulateOp);
  AccumulateOp() = default;
  ~AccumulateOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override {}
  void InferOutputBlobTimeShape(std::function<const Shape*(const std::string&)> GetTimeShape4BnInOp,
                                const ParallelContext* parallel_ctx,
                                Shape* time_shape) const override {
    TODO();
  }

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return false; }
  void InferOutputBlobSbpInferHint(
      std::function<SbpInferHint*(const std::string&)> SbpInferHint4BnInOp,
      std::function<int32_t(const std::string&)> ShapeNumAxes4BnInOp,
      const ParallelContext* parallel_context) const override {
    NaiveInferOutputBlobSbpInferHint(SbpInferHint4BnInOp, ShapeNumAxes4BnInOp, parallel_context);
  }

  LogicalBlobId ibn2lbi(const std::string& input_bn) const override { return GenPackedLbi(); }
  LogicalBlobId obn2lbi(const std::string& output_bn) const override { return GenPackedLbi(); }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_ACCUMULATE_OP_H_
