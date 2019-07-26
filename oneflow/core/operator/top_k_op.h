#ifndef ONEFLOW_CORE_OPERATOR_TOP_K_OP_H_
#define ONEFLOW_CORE_OPERATOR_TOP_K_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

struct TopKOpCtx : public OpContext {
#ifdef WITH_CUDA
  TopKOpCtx(int32_t instance_size, int32_t k, int32_t temp_storage_bytes)
      : instance_size_(instance_size), k_(k), temp_storage_bytes_(temp_storage_bytes) {}
  int32_t instance_size_;
  int32_t k_;
  int32_t temp_storage_bytes_;
#endif
};

class TopKOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TopKOp);
  TopKOp() = default;
  ~TopKOp() override = default;

  void InitFromOpConf() override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext*, int64_t record_piece_size,
                      std::function<void(OpContext*)> EnrollOpCtx) const override;
  const PbMessage& GetCustomizedConf() const override { return op_conf().top_k_conf(); }

 private:
  virtual void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*,
      KernelConf*, const OpContext*) const override;
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return false; }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_TOP_K_OP_H_
