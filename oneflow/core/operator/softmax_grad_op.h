#ifndef ONEFLOW_CORE_OPERATOR_SOFTMAX_GRAD_OP_H_
#define ONEFLOW_CORE_OPERATOR_SOFTMAX_GRAD_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

struct SoftmaxGradOpCtx : public OpContext {
  int32_t axis;
  int32_t dims;
  int64_t transpose_rows;
  int64_t transpose_cols;
  bool need_transpose;
};

class SoftmaxGradOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SoftmaxGradOp);
  SoftmaxGradOp() = default;
  ~SoftmaxGradOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx, int64_t record_piece_size,
                             std::function<void(OpContext*)> EnrollOpCtx) const override;

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }

  void GetSbpSignatures(SbpSignatureList* sbp_sig_list) const override;

  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext*, KernelConf*, const OpContext*) const override;
  SoftmaxGradOpCtx* NewSoftmaxGradOpCtx(const Shape& in_shape) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_SOFTMAX_GRAD_OP_H_
