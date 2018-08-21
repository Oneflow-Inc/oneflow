#ifndef ONEFLOW_CORE_OPERATOR_PB_BOXING_OP_H_
#define ONEFLOW_CORE_OPERATOR_PB_BOXING_OP_H_

#include "oneflow/core/operator/boxing_op.h"

namespace oneflow {

class PbBoxingOp final : public BoxingOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PbBoxingOp);
  PbBoxingOp() = default;
  ~PbBoxingOp() = default;

  void InitFromOpConf() override;
  const BoxingOpConf& boxing_conf() const override;
  const PbRpf<std::string>& InputBns() const override { return pb_input_bns(); }
  const PbRpf<std::string>& OutputBns() const override { return pb_output_bns(); }

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;

 protected:
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_PB_BOXING_OP_H_
