#ifndef ONEFLOW_CORE_OPERATOR_IMAGE_SEGMENTATION_MASK_OP_H_
#define ONEFLOW_CORE_OPERATOR_IMAGE_SEGMENTATION_MASK_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ImageSegmentationMaskOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ImageSegmentationMaskOp);
  ImageSegmentationMaskOp() = default;
  ~ImageSegmentationMaskOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext*) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_IMAGE_SEGMENTATION_MASK_OP_H_
