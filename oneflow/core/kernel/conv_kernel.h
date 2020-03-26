#ifndef ONEFLOW_CORE_KERNEL_CONV_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_CONV_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/device/cudnn_util.h"
#include "oneflow/core/device/cudnn_conv_util.h"

namespace oneflow {

namespace {

template<typename T>
const T* GetImgDptr(const Blob* blob, int64_t idx) {
  return blob->dptr<T>() + blob->shape().Count(1) * idx;
}

template<typename T>
T* GetImgMutDptr(Blob* blob, int64_t idx) {
  return const_cast<T*>(GetImgDptr<T>(blob, idx));
}

}  // namespace

template<DeviceType device_type, typename T>
class ConvKernelIf : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvKernelIf);
  ConvKernelIf() = default;
  virtual ~ConvKernelIf() = default;

 protected:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void InitConstBufBlobs(DeviceCtx*,
                         std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  virtual void DoForwardDataContent(DeviceCtx*, const Blob* in_blob, const Blob* weight_blob,
                                    Blob* out_blob,
                                    std::function<Blob*(const std::string&)> BnInOp2Blob) const = 0;
  virtual void WeightBackward(DeviceCtx*, const Blob* out_diff_blob, const Blob* in_blob,
                              Blob* weight_diff_blob, Blob* in_diff_blob,
                              std::function<Blob*(const std::string&)> BnInOp2Blob) const = 0;
  virtual void BiasBackward(DeviceCtx*, const Blob* out_diff_blob, Blob* bias_diff_blob,
                            std::function<Blob*(const std::string&)> BnInOp2Blob) const = 0;

  const PbMessage& GetCustomizedOpConf() const override;
  const ConvKernelConf& GetConvKernelConf() const;
  const int32_t OpKernelDim() const;
};

template<typename T>
using Im2ColFunc = void (*)(const int dim_num, DeviceCtx* device_ctx, const T* in_dptr,
                            const ShapeView& in_shape, const ShapeView& weight_shape,
                            const ShapeView& out_shape, const int32_t* strides,
                            const int32_t* dilation_rate, const int32_t* padding_before,
                            T* col_buf);

template<typename T>
using Col2ImFunc = void (*)(const int dim_num, DeviceCtx* device_ctx, const T* col_buf,
                            const ShapeView& in_shape, const ShapeView& weight_shape,
                            const ShapeView& out_shape, const int32_t* strides,
                            const int32_t* dilation_rate, const int32_t* padding_before,
                            T* in_diff_ptr);

template<typename T>
using GemmFunc = void (*)(DeviceCtx* ctx, enum CBLAS_TRANSPOSE, enum CBLAS_TRANSPOSE, const int m,
                          const int n, const int k, const T alpha, const T* a, const T* b,
                          const T beta, T* c);

template<DeviceType device_type, typename T>
class ConvKernelImplByIm2Col : public ConvKernelIf<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvKernelImplByIm2Col);
  ConvKernelImplByIm2Col() = default;
  ~ConvKernelImplByIm2Col() = default;

 protected:
  void VirtualKernelInit() override;
  void DoForwardDataContent(DeviceCtx*, const Blob* in_blob, const Blob* weight_blob,
                            Blob* out_blob,
                            std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void WeightBackward(DeviceCtx*, const Blob* out_diff_blob, const Blob* in_blob,
                      Blob* weight_diff_blob, Blob* in_diff_blob,
                      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void BiasBackward(DeviceCtx*, const Blob* out_diff_blob, Blob* bias_diff_blob,
                    std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

 private:
  Im2ColFunc<T> im2col_func_;
  Col2ImFunc<T> col2im_func_;
  GemmFunc<T> forward_func_;
  enum CBLAS_TRANSPOSE is_out_diff_need_trans_;
  size_t dhw_offset_;
  const int32_t* strides_;
  const int32_t* dilation_rate_;
  const int32_t* padding_before_;
  ShapeView in_shape_;
  ShapeView out_shape_;
  ShapeView weight_shape_;
};

template<DeviceType device_type, typename T>
class ConvKernel;

template<typename T>
class ConvKernel<DeviceType::kCPU, T> final : public ConvKernelImplByIm2Col<DeviceType::kCPU, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvKernel);
  ConvKernel() = default;
  ~ConvKernel() = default;

 private:
};

template<typename T>
class ConvKernel<DeviceType::kGPU, T> final : public ConvKernelImplByIm2Col<DeviceType::kGPU, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvKernel);
  ConvKernel() = default;
  ~ConvKernel() = default;

 private:
  void VirtualKernelInit() override;
  void KernelInitWithCudnn();

  void DoForwardDataContent(DeviceCtx*, const Blob* in_blob, const Blob* weight_blob,
                            Blob* out_blob,
                            std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void DoForwardDataContentWithCudnn(DeviceCtx*, const Blob* in_blob, const Blob* weight_blob,
                                     Blob* out_blob,
                                     std::function<Blob*(const std::string&)> BnInOp2Blob) const;

  void WeightBackward(DeviceCtx*, const Blob* out_diff_blob, const Blob* in_blob,
                      Blob* weight_diff_blob, Blob* in_diff_blob,
                      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void WeightBackwardWithCudnn(DeviceCtx*, const Blob* out_diff_blob, const Blob* in_blob,
                               Blob* weight_diff_blob, Blob* in_diff_blob,
                               std::function<Blob*(const std::string&)> BnInOp2Blob) const;

  void BiasBackward(DeviceCtx*, const Blob* out_diff_blob, Blob* bias_diff_blob,
                    std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void BiasBackwardWithCudnn(DeviceCtx*, const Blob* out_diff_blob, Blob* bias_diff_blob,
                             std::function<Blob*(const std::string&)> BnInOp2Blob) const;

  std::unique_ptr<CudnnTensorDesc> in_desc_;
  std::unique_ptr<CudnnTensorDesc> out_desc_;
  std::unique_ptr<CudnnFilterDesc> filter_desc_;
  std::unique_ptr<CudnnConvDesc> conv_desc_;
  std::unique_ptr<CudnnTensorDesc> bias_desc_;
};

template<>
class ConvKernel<DeviceType::kGPU, float16> final : public ConvKernelIf<DeviceType::kGPU, float16> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvKernel);
  ConvKernel() = default;
  ~ConvKernel() = default;

 private:
  void VirtualKernelInit() override;
  void KernelInitWithCudnn();

  void DoForwardDataContent(DeviceCtx*, const Blob* in_blob, const Blob* weight_blob,
                            Blob* out_blob,
                            std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void WeightBackward(DeviceCtx*, const Blob* out_diff_blob, const Blob* in_blob,
                      Blob* weight_diff_blob, Blob* in_diff_blob,
                      std::function<Blob*(const std::string&)> BnInOp2Blob) const override {}
  void BiasBackward(DeviceCtx*, const Blob* out_diff_blob, Blob* bias_diff_blob,
                    std::function<Blob*(const std::string&)> BnInOp2Blob) const override {}

  std::unique_ptr<CudnnTensorDesc> in_desc_;
  std::unique_ptr<CudnnTensorDesc> out_desc_;
  std::unique_ptr<CudnnFilterDesc> filter_desc_;
  std::unique_ptr<CudnnConvDesc> conv_desc_;
  std::unique_ptr<CudnnTensorDesc> bias_desc_;
};

// ConvKernel<kCPU, float16> is not used
template<>
class ConvKernel<DeviceType::kCPU, float16> final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvKernel);
  ConvKernel() = default;
  ~ConvKernel() = default;
};

template<typename T>
class ColBufWriter {
 public:
  ColBufWriter(const T* src_ptr, T* dst_ptr, int64_t c_size, int64_t id_size, int64_t ih_size,
               int64_t iw_size, int64_t od_size, int64_t oh_size, int64_t ow_size);
  virtual ~ColBufWriter() = default;
  virtual void DHWCWrite(int64_t c, int64_t id, int64_t ih, int64_t iw) = 0;
  virtual void CDHWWrite(int64_t c, int64_t id, int64_t ih, int64_t iw) = 0;
  virtual void InvalidDFunc() = 0;
  virtual void InvalidHFunc() = 0;
  virtual void InvalidWFunc() = 0;
  virtual void NextImCSize() = 0;

 protected:
  const T* src_ptr_;
  T* dst_ptr_;
  int64_t c_size_;
  int64_t id_size_;
  int64_t ih_size_;
  int64_t iw_size_;
  int64_t od_size_;
  int64_t oh_size_;
  int64_t ow_size_;
};

template<typename T>
class Im2ColWriter final : public ColBufWriter<T> {
 public:
  Im2ColWriter(const T* src_ptr, T* dst_ptr, int64_t c_size, int64_t id_size, int64_t ih_size,
               int64_t iw_size, int64_t od_size, int64_t oh_size, int64_t ow_size);
  ~Im2ColWriter() = default;
  void DHWCWrite(int64_t c, int64_t id, int64_t ih, int64_t iw) override;
  void CDHWWrite(int64_t c, int64_t id, int64_t ih, int64_t iw) override;
  void InvalidDFunc() override;
  void InvalidHFunc() override;
  void InvalidWFunc() override;
  void NextImCSize() override;
};

template<typename T>
class Col2ImWriter final : public ColBufWriter<T> {
 public:
  Col2ImWriter(const T* src_ptr, T* dst_ptr, int64_t c_size, int64_t id_size, int64_t ih_size,
               int64_t iw_size, int64_t od_size, int64_t oh_size, int64_t ow_size);
  ~Col2ImWriter() = default;
  void DHWCWrite(int64_t c, int64_t id, int64_t ih, int64_t iw) override;
  void CDHWWrite(int64_t c, int64_t id, int64_t ih, int64_t iw) override;
  void InvalidDFunc() override;
  void InvalidHFunc() override;
  void InvalidWFunc() override;
  void NextImCSize() override;
};

template<typename T>
using DHWValidFunc = void (ColBufWriter<T>::*)(int64_t c, int64_t kd, int64_t kh, int64_t kw);

template<typename T>
class ColBufUtil final {
 public:
  ColBufUtil(const ShapeView& in_shape, const ShapeView& out_shape, int32_t dhw_offset,
             const int32_t* strides, const int32_t* dilation_rate, const int32_t* padding_before);
  void operator()(ColBufWriter<T>* col_buf_writer, int64_t c, int64_t kd, int64_t kh, int64_t kw);

 private:
  int64_t id_num_;
  int64_t ih_num_;
  int64_t iw_num_;
  int64_t od_num_;
  int64_t oh_num_;
  int64_t ow_num_;
  const int32_t* strides_;
  const int32_t* dilation_rate_;
  const int32_t* padding_before_;
  DHWValidFunc<T> dhw_valid_func_;
};

template<DeviceType device_type, typename T>
class ConvKernelUtil;

template<typename T>
struct ConvKernelUtil<DeviceType::kCPU, T> final {
 public:
  static void NCDHWIm2Col(const int dim_num, DeviceCtx* device_ctx, const T* in_dptr,
                          const ShapeView& in_shape, const ShapeView& weight_shape,
                          const ShapeView& out_shape, const int32_t* strides,
                          const int32_t* dilation_rate, const int32_t* padding_before, T* col_buf);

  static void NDHWCIm2Col(const int dim_num, DeviceCtx* device_ctx, const T* in_dptr,
                          const ShapeView& in_shape, const ShapeView& weight_shape,
                          const ShapeView& out_shape, const int32_t* strides,
                          const int32_t* dilation_rate, const int32_t* padding_before, T* col_buf);

  static void NCDHWCol2Im(const int dim_num, DeviceCtx* device_ctx, const T* col_buf,
                          const ShapeView& in_shape, const ShapeView& weight_shape,
                          const ShapeView& out_shape, const int32_t* strides,
                          const int32_t* dilation_rate, const int32_t* padding_before,
                          T* in_diff_ptr);

  static void NDHWCCol2Im(const int dim_num, DeviceCtx* device_ctx, const T* col_buf,
                          const ShapeView& in_shape, const ShapeView& weight_shape,
                          const ShapeView& out_shape, const int32_t* strides,
                          const int32_t* dilation_rate, const int32_t* padding_before,
                          T* in_diff_ptr);

 private:
  static void DoNCDWHFunc(const ShapeView& weight_shape, ColBufUtil<T>& conv_util,
                          ColBufWriter<T>* col_buf_writer);

  static void DoNDWHCFunc(const ShapeView& weight_shape, ColBufUtil<T>& conv_util,
                          ColBufWriter<T>* col_buf_writer);
};

template<typename T>
struct ConvKernelUtil<DeviceType::kGPU, T> final {
 public:
  static void NCDHWIm2Col(const int dim_num, DeviceCtx* device_ctx, const T* in_dptr,
                          const ShapeView& in_shape, const ShapeView& weight_shape,
                          const ShapeView& out_shape, const int32_t* strides,
                          const int32_t* dilation_rate, const int32_t* padding_before, T* col_buf);

  static void NDHWCIm2Col(const int dim_num, DeviceCtx* device_ctx, const T* in_dptr,
                          const ShapeView& in_shape, const ShapeView& weight_shape,
                          const ShapeView& out_shape, const int32_t* strides,
                          const int32_t* dilation_rate, const int32_t* padding_before, T* col_buf);

  static void NCDHWCol2Im(const int dim_num, DeviceCtx* device_ctx, const T* col_buf,
                          const ShapeView& in_shape, const ShapeView& weight_shape,
                          const ShapeView& out_shape, const int32_t* strides,
                          const int32_t* dilation_rate, const int32_t* padding_before,
                          T* in_diff_ptr);

  static void NDHWCCol2Im(const int dim_num, DeviceCtx* device_ctx, const T* col_buf,
                          const ShapeView& in_shape, const ShapeView& weight_shape,
                          const ShapeView& out_shape, const int32_t* strides,
                          const int32_t* dilation_rate, const int32_t* padding_before,
                          T* in_diff_ptr);

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_CONV_KERNEL_H_
