#ifndef ONEFLOW_CORE_KERNEL_CONVOLUTION_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_CONVOLUTION_KERNEL_H_

#include "oneflow/core/kernel/kernel_manager.h"

namespace oneflow {

template<DeviceType device_type, typename FloatingPointType>
class ConvolutionKernelUtil final {
 public:
  static void Im2Col(const KernelCtx& ctx, const FloatingPointType* data_im,
                     const int channels, const int height, const int width,
                     const int kernel_h, const int kernel_w, const int pad_h,
                     const int pad_w, const int stride_h, const int stride_w,
                     const int dilation_h, const int dilation_w,
                     FloatingPointType* data_col);
  static void Col2Im(const KernelCtx& ctx, const FloatingPointType* data_col,
                     const int channels, const int height, const int width,
                     const int kernel_h, const int kernel_w, const int pad_h,
                     const int pad_w, const int stride_h, const int stride_w,
                     const int dilation_h, const int dilation_w,
                     FloatingPointType* data_im);
};

template<DeviceType device_type, typename FloatingPointType>
class ConvolutionKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvolutionKernel);
  ConvolutionKernel() = default;
  ~ConvolutionKernel() = default;

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;
  void Backward(const KernelCtx&,
                std::function<Blob*(const std::string&)>) const override;

 private:
  void InitModelBlobsWithRandomSeed(
      const KernelCtx&, std::mt19937 random_seed_gen,
      std::function<Blob*(const std::string&)>) const override;
  void InitModelBlobsWithSnapshot(
      const KernelCtx& ctx, int32_t part_id, int32_t part_num,
      const Snapshot* snapshot,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void InitModelTmpBlobs(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  void ComputeWeightDiff(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const;
  void ComputeInputDiff(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const;
  void ComputeBiasDiff(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_CONVOLUTION_KERNEL_H_
