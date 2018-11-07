#ifndef ONEFLOE_CORE_KERNEL_GENERATE_ANCHORS_KERNEL_H_
#define ONEFLOE_CORE_KERNEL_GENERATE_ANCHORS_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<typename T>
class GenerateAnchorsKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GenerateAnchorsKernel);
  GenerateAnchorsKernel() = default;
  ~GenerateAnchorsKernel() = default;

  static const size_t kBoxElemSize = 4;

 private:
  void InitConstBufBlobs(DeviceCtx* ctx,
                         std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  // void GenerateBoxes(const PriorBoxOpConf& conf, const int32_t img_num, const int32_t height,
  //                    const int32_t width, Blob* boxes_blob) const;
  // T Clip(const T value, const bool use_clip) const;
  std::vector<T> GenerateBaseAnchors() const;
  void ShiftAnchors(const std::vector<T>& base_anchors_vec, Blob* anchors_blob) const;
};

}  // namespace oneflow

#endif  // ONEFLOE_CORE_KERNEL_GENERATE_ANCHORS_KERNEL_H_