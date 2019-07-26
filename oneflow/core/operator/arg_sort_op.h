#ifndef ONEFLOW_CORE_OPERATOR_ARG_SORT_OP_H_
#define ONEFLOW_CORE_OPERATOR_ARG_SORT_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

struct ArgSortOpCtx : public OpContext {
#ifdef WITH_CUDA
  ArgSortOpCtx(int32_t temp_storage_bytes) : temp_storage_bytes_(temp_storage_bytes) {}
  int32_t temp_storage_bytes_;
#endif
};

class ArgSortOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ArgSortOp);
  ArgSortOp() = default;
  ~ArgSortOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext*, int64_t record_piece_size,
                      std::function<void(OpContext*)> EnrollOpCtx) const override;

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return false; }
  virtual void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*,
      KernelConf*, const OpContext*) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_ARG_SORT_OP_H_
