#ifndef ONEFLOW_CORE_OPERATOR_SOFTMAX_OP_H_
#define ONEFLOW_CORE_OPERATOR_SOFTMAX_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

struct SoftmaxOpCtx : public OpContext {
  int32_t axis;
  int32_t dims;
  int64_t transpose_rows;
  int64_t transpose_cols;
  bool need_transpose;
};

class SoftmaxOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SoftmaxOp);
  SoftmaxOp() = default;
  ~SoftmaxOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx, int64_t record_piece_size,
                      std::function<void(OpContext*)> EnrollOpCtx) const override;
  void InferBwBufBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                           const ParallelContext*, const OpContext*) const override;

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return false; }
  void InferOutputBlobSbpInferHint(
      std::function<SbpInferHint*(const std::string&)> SbpInferHint4BnInOp,
      std::function<int32_t(const std::string&)> ShapeNumAxes4BnInOp,
      const ParallelContext* parallel_context) const override {
    CHECK_EQ(parallel_context->policy(), kDataParallel);
    NaiveInferOutputBlobSbpInferHint(SbpInferHint4BnInOp, ShapeNumAxes4BnInOp, parallel_context);
  }

  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext*, KernelConf*, const OpContext*) const override;
  SoftmaxOpCtx* NewSoftmaxOpCtx(const Shape& in_shape) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_SOFTMAX_OP_H_
