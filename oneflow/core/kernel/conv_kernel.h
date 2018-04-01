#ifndef ONEFLOW_CORE_KERNEL_CONV_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_CONV_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/operator/conv_op.h"
#include "oneflow/core/device/cudnn_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ConvKernelIf : public KernelIf<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvKernelIf);
  ConvKernelIf() = default;
  virtual ~ConvKernelIf() = default;

 protected:
  void ForwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
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

  virtual void DoForwardDataContent(
      DeviceCtx*, const Blob* in_blob, const Blob* weight_blob, Blob* out_blob,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const = 0;
  virtual void WeightBackward(
      DeviceCtx*, const Blob* out_diff_blob, const Blob* in_blob,
      Blob* weight_diff_blob, Blob* in_diff_blob,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const = 0;
  virtual void BiasBackward(
      DeviceCtx*, const Blob* out_diff_blob, Blob* bias_diff_blob,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const = 0;

  const PbMessage& GetCustomizedOpConf() const override;
  const ConvKernelConf& GetConvKernelConf() const;
  const int32_t OpKernelDim() const;
};

template<typename T>
using Im2ColFunc = void (*)(DeviceCtx* device_ctx, const T* in_dptr,
                            const Shape& in_shape, const Shape& weight_shape,
                            const Shape& out_shape, const int32_t* strides,
                            const int32_t* dilation_rate,
                            const int32_t* padding_before, T* col_buf);

template<typename T>
using Col2ImFunc = void (*)(DeviceCtx* device_ctx, const T* col_buf,
                            const Shape& in_shape, const Shape& weight_shape,
                            const Shape& out_shape, const int32_t* strides,
                            const int32_t* dilation_rate,
                            const int32_t* padding_before, T* in_diff_ptr);

template<typename T>
using GemmFunc = void (*)(DeviceCtx* ctx, enum CBLAS_TRANSPOSE,
                          enum CBLAS_TRANSPOSE, const int m, const int n,
                          const int k, const T alpha, const T* a, const T* b,
                          const T beta, T* c);

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
  void VirtualKernelInit(const ParallelContext*) override;
  void DoForwardDataContent(
      DeviceCtx*, const Blob* in_blob, const Blob* weight_blob, Blob* out_blob,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void WeightBackward(
      DeviceCtx*, const Blob* out_diff_blob, const Blob* in_blob,
      Blob* weight_diff_blob, Blob* in_diff_blob,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void BiasBackward(
      DeviceCtx*, const Blob* out_diff_blob, Blob* bias_diff_blob,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  Im2ColFunc<T> im2col_func_;
  Col2ImFunc<T> col2im_func_;
  GemmFunc<T> forward_func_;
  enum CBLAS_TRANSPOSE is_out_diff_need_trans_;
  size_t dhw_offset_;
  const int32_t* strides_;
  const int32_t* dilation_rate_;
  const int32_t* padding_before_;
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
  void DoForwardDataContent(
      DeviceCtx*, const Blob* in_blob, const Blob* weight_blob, Blob* out_blob,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void WeightBackward(
      DeviceCtx*, const Blob* out_diff_blob, const Blob* in_blob,
      Blob* weight_diff_blob, Blob* in_diff_blob,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void BiasBackward(
      DeviceCtx*, const Blob* out_diff_blob, Blob* bias_diff_blob,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  std::unique_ptr<CudnnTensorDesc> in_desc_;
  std::unique_ptr<CudnnTensorDesc> out_desc_;
  std::unique_ptr<CudnnFilterDesc> filter_desc_;
  std::unique_ptr<CudnnConvDesc> conv_desc_;
  std::unique_ptr<CudnnTensorDesc> bias_desc_;
};

template<typename T>
class ColBufWriter final {
 public:
  ColBufWriter(const T* src_ptr, T* dst_ptr, int64_t c_size, int64_t id_size,
               int64_t ih_size, int64_t iw_size);
  void Im2ColDHWCWrite(int64_t c, int64_t id, int64_t ih, int64_t iw);
  void Im2ColCDHWWrite(int64_t c, int64_t id, int64_t ih, int64_t iw);
  void WriteZero() { *(dst_ptr_++) = 0; }
  void CleanIdSize();
  void CleanIhSize();
  void CleanIwSize();
  void SkipIdSize() { src_ptr_ += id_size_; }
  void SkipIhSize() { src_ptr_ += ih_size_; }
  void SkipIwSize() { src_ptr_ += iw_size_; }
  void Col2ImDHWCWrite(int64_t c, int64_t id, int64_t ih, int64_t iw);
  void Col2ImCDHWWrite(int64_t c, int64_t id, int64_t ih, int64_t iw);
  void NextCSize() { src_ptr_ += c_size_; }

 private:
  const T* src_ptr_;
  T* dst_ptr_;
  int64_t c_size_;
  int64_t id_size_;
  int64_t ih_size_;
  int64_t iw_size_;
};

template<typename T>
using DHWInvalidFunc = void (ColBufWriter<T>::*)();

template<typename T>
using DHWValidFunc = void (ColBufWriter<T>::*)(int64_t c, int64_t kd,
                                               int64_t kh, int64_t kw);

template<typename T>
class ColBufUtil final {
 public:
  ColBufUtil(const Shape& in_shape, const Shape& out_shape, int32_t dhw_offset,
             bool is_im2col, const int32_t* strides,
             const int32_t* dilation_rate, const int32_t* padding_before);
  void operator()(ColBufWriter<T>& col_buf_writer, int64_t c, int64_t kd,
                  int64_t kh, int64_t kw);

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
  DHWInvalidFunc<T> d_invalid_func_;
  DHWInvalidFunc<T> h_invalid_func_;
  DHWInvalidFunc<T> w_invalid_func_;
  DHWValidFunc<T> dhw_valid_func_;
};

template<typename T>
struct ConvKernelUtil final {
 public:
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

 private:
  static void DoNCDWHFunc(const Shape& weight_shape, ColBufUtil<T>& conv_util,
                          ColBufWriter<T>& col_buf_writer);

  static void DoNDWHCFunc(const Shape& weight_shape, ColBufUtil<T>& conv_util,
                          ColBufWriter<T>& col_buf_writer);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_CONV_KERNEL_H_
