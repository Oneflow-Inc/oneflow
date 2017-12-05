#ifndef ONEFLOW_CORE_OPERATOR_BOXING_OP_H_
#define ONEFLOW_CORE_OPERATOR_BOXING_OP_H_

#include "oneflow/core/operator/operator.h"

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
  void GenBoxingInfo(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const std::vector<std::string>& bns, int32_t axis,
      bool is_concat_or_split, BoxingInfo* boxing_info) const;
  void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx,
      KernelConf* kernel_conf) const override;

  std::string ibn2lbn(const std::string& input_bn) const override;
  std::string obn2lbn(const std::string& output_bn) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_BOXING_OP_H_
