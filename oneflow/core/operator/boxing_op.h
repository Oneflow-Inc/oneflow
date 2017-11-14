#ifndef ONEFLOW_CORE_OPERATOR_BOXING_OP_H_
#define ONEFLOW_CORE_OPERATOR_BOXING_OP_H_

#include "oneflow/core/operator/operator_manager.h"

namespace oneflow {

class BoxingOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingOp);
  BoxingOp() = default;
  ~BoxingOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetSpecialConf() const override;

  void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx);

 private:
  std::string ibn2lbn(const std::string& input_bn) const override;
  std::string obn2lbn(const std::string& output_bn) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_BOXING_OP_H_
