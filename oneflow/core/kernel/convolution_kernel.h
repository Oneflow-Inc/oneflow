#ifndef ONEFLOW_CORE_KERNEL_CONVOLUTION_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_CONVOLUTION_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/device/cudnn_support.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ConvolutionKernelUtil final {
 public:
  static void Im2Col(const KernelCtx& ctx, const T* data_im, const int channels,
                     const int height, const int width, const int kernel_h,
                     const int kernel_w, const int pad_h, const int pad_w,
                     const int stride_h, const int stride_w,
                     const int dilation_h, const int dilation_w, T* data_col);
  static void Col2Im(const KernelCtx& ctx, const T* data_col,
                     const int channels, const int height, const int width,
                     const int kernel_h, const int kernel_w, const int pad_h,
                     const int pad_w, const int stride_h, const int stride_w,
                     const int dilation_h, const int dilation_w, T* data_im);
};

template<DeviceType device_type, typename T>
class ConvolutionKernel : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvolutionKernel);
  ConvolutionKernel() = default;
  ~ConvolutionKernel() = default;

 protected:
  void ForwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
  void InitModelBlobsWithRandomSeed(
      const KernelCtx&, std::mt19937 random_seed_gen,
      std::function<Blob*(const std::string&)>) const override;
  void InitModelBlobsWithDir(
      const KernelCtx& ctx, int32_t part_id, int32_t part_num,
      const std::string& model_load_dir,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void InitModelTmpBlobs(
      const KernelCtx& ctx, const ParallelContext* parallel_ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

 private:
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

#ifdef WITH_CUDNN
template<typename T>
class CudnnConvolutionKernel final
    : public ConvolutionKernel<DeviceType::kGPU, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudnnConvolutionKernel);
  CudnnConvolutionKernel() = default;
  ~CudnnConvolutionKernel() = default;

 private:
  void ForwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;

  CudnnConvolutionDesc cudnn_conv_desc_;
};
#endif  // WITH_CUDNN

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_CONVOLUTION_KERNEL_H_
