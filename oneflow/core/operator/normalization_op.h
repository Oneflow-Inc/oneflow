#ifndef ONEFLOW_CORE_OPERATOR_NORMALIZATION_OP_H_
#define ONEFLOW_CORE_OPERATOR_NORMALIZATION_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/device/cudnn_util.h"

namespace oneflow {

struct NormalizationOpCtx : public OpContext {
  int32_t axis;
  int32_t dims;
  int64_t transpose_rows;
  int64_t transpose_cols;
  bool need_transpose;
};

class NormalizationOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalizationOp);
  NormalizationOp() = default;
  ~NormalizationOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  bool NeedOutBlobWhenBackward() const override { return false; }

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx, int64_t record_piece_size,
                      std::function<void(OpContext*)> EnrollOpCtx) const override;

 private:
  void InferOutBlobModelSplitAxis(std::function<int32_t*(const std::string&)> ModelSplitAxis4BnInOp,
                                  std::function<int32_t(const std::string&)> ShapeNumAxes4BnInOp,
                                  const ParallelContext* parallel_context) const override {
    CHECK_EQ(parallel_context->policy(), kDataParallel);
    NaiveInferOutBlobModelSplitAxis(ModelSplitAxis4BnInOp, ShapeNumAxes4BnInOp, parallel_context);
  }

  void InferParamBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                           const NormalizationOpConf&, int64_t norm_part_num, DataType in_data_type,
                           bool use_cudnn) const;
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext*, KernelConf*, const OpContext*) const override;
#ifdef WITH_CUDA
  void InferBlobDescsForCudnn(
      std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp) const;
  void VirtualGenKernelConfForCudnn(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*,
      KernelConf*) const;
#endif
  void VirtualFixParallelDesc(ParallelDesc* pr_desc) const override;
  NormalizationOpCtx* NewNormalizationOpCtx(const Shape& in_shape) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_NORMALIZATION_OP_H_
