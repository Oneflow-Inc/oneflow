#ifndef ONEFLOW_CORE_OPERATOR_POD_BOXING_OP_H_
#define ONEFLOW_CORE_OPERATOR_POD_BOXING_OP_H_

#include "oneflow/core/operator/boxing_op.h"

namespace oneflow {

class PodBoxingOp final : public BoxingOp {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PodBoxingOp);
  PodBoxingOp() = default;
  ~PodBoxingOp() = default;

  void InitFromOpConf() override;
  const BoxingOpConf& boxing_conf() const override;
  const PbRpf<std::string>& InputBns() const override { return input_bns(); }
  const PbRpf<std::string>& OutputBns() const override { return output_bns(); }

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;

 protected:
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override;

 private:
  void InferDataTmpBlobDesc(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            std::vector<int64_t>* data_temp_shape_vec,
                            std::vector<int64_t>* instance_inner_shape_vec) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_POD_BOXING_OP_H_
