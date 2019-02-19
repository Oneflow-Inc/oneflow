#ifndef ONEFLOW_CORE_OPERATOR_CAST_INITED_MODEL_OP_H_
#define ONEFLOW_CORE_OPERATOR_CAST_INITED_MODEL_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class CastInitedModelOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CastInitedModelOp);
  CastInitedModelOp() = default;
  ~CastInitedModelOp() = default;

  void InitFromOpConf() override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  const PbMessage& GetCustomizedConf() const override { return op_conf().cast_inited_model_conf(); }
  bool NeedInBlobWhenBackward() const override { return false; }
  bool NeedOutBlobWhenBackward() const override { return false; }

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override {
    return UNIMPLEMENTED();
  }
  LogicalBlobId ibn2lbi(const std::string& input_bn) const override { return GenPackedLbi(); }
  LogicalBlobId obn2lbi(const std::string& output_bn) const override { return GenPackedLbi(); }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CAST_INITED_MODEL_OP_H_
