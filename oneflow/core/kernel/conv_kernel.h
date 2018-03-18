#ifndef ONEFLOW_CORE_KERNEL_CONV_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_CONV_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/operator/conv_op.h"
#include "oneflow/core/device/cudnn_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ConvKernelIf : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvKernelIf);
  ConvKernelIf() = default;
  virtual ~ConvKernelIf() = default;

 protected:
  void BackwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void InitPureModelTmpBlobs(
      DeviceCtx*,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void InitModelBlobsWithRandomSeed(
      DeviceCtx*, std::mt19937* random_seed_gen,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void InitModelBlobsWithDir(
      DeviceCtx*, int32_t part_id, int32_t part_num,
      const std::string& model_load_dir,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  virtual void WeightBackward(
      DeviceCtx*, std::function<Blob*(const std::string&)>) const = 0;
  virtual void BiasBackward(DeviceCtx*,
                            std::function<Blob*(const std::string&)>) const = 0;

  const PbMessage& GetCustomizedOpConf() const override;
  const ConvKernelConf& GetConvKernelConf() const;
  const int32_t KernelDim() const;
};

template<DeviceType device_type, typename T>
class ConvKernel;

template<typename T>
class ConvKernel<DeviceType::kCPU, T> final
    : public ConvKernelIf<DeviceType::kCPU, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvKernel);
  ConvKernel() = default;
  ~ConvKernel() = default;

 private:
  void ForwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void WeightBackward(DeviceCtx*,
                      std::function<Blob*(const std::string&)>) const override;
  void BiasBackward(DeviceCtx*,
                    std::function<Blob*(const std::string&)>) const override;
};

template<typename T>
class ConvKernel<DeviceType::kGPU, T> final
    : public ConvKernelIf<DeviceType::kGPU, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvKernel);
  ConvKernel() = default;
  ~ConvKernel() = default;

 private:
  void VirtualKernelInit(const ParallelContext*) override;
  void ForwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void WeightBackward(DeviceCtx*,
                      std::function<Blob*(const std::string&)>) const override;
  void BiasBackward(DeviceCtx*,
                    std::function<Blob*(const std::string&)>) const override;

  std::unique_ptr<CudnnTensorDesc> in_desc_;
  std::unique_ptr<CudnnTensorDesc> out_desc_;
  std::unique_ptr<CudnnFilterDesc> filter_desc_;
  std::unique_ptr<CudnnConvDesc> conv_desc_;
  std::unique_ptr<CudnnTensorDesc> bias_desc_;
};

template<typename T>
class ConvKernelUtil final {
 public:
  static void Im2Col(DeviceCtx* device_ctx, const T* in_dptr,
                     const Shape& in_shape, const Shape& weight_shape,
                     const Shape& out_shape, const std::string& data_format,
                     const int32_t* strides, const int32_t* dilation_rate,
                     const int32_t* padding_before, T* col_buf);

  static void Col2Im(DeviceCtx* device_ctx, const T* col_buf,
                     const Shape& in_shape, const Shape& weight_shape,
                     const Shape& out_shape, const std::string& data_format,
                     const int32_t* strides, const int32_t* dilation_rate,
                     const int32_t* padding_before, T* in_diff_ptr);

  static void NCDHWIm2Col(DeviceCtx* device_ctx, const T* in_dptr,
                          const Shape& in_shape, const Shape& weight_shape,
                          const Shape& out_shape, const int32_t* strides,
                          const int32_t* dilation_rate,
                          const int32_t* padding_before, T* col_buf);

  static void NDHWCIm2Col(DeviceCtx* device_ctx, const T* in_dptr,
                          const Shape& in_shape, const Shape& weight_shape,
                          const Shape& out_shape, const int32_t* strides,
                          const int32_t* dilation_rate,
                          const int32_t* padding_before, T* col_buf);

  static void NCDHWCol2Im(DeviceCtx* device_ctx, const T* col_buf,
                          const Shape& in_shape, const Shape& weight_shape,
                          const Shape& out_shape, const int32_t* strides,
                          const int32_t* dilation_rate,
                          const int32_t* padding_before, T* in_diff_ptr);

  static void NDHWCCol2Im(DeviceCtx* device_ctx, const T* col_buf,
                          const Shape& in_shape, const Shape& weight_shape,
                          const Shape& out_shape, const int32_t* strides,
                          const int32_t* dilation_rate,
                          const int32_t* padding_before, T* in_diff_ptr);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_CONV_KERNEL_H_
