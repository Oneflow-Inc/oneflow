#include "oneflow/core/kernel/layer_norm_kernel.h"

namespace oneflow {

namespace {

class LayerNormCudnnBnCtx final {
 public:
  LayerNormCudnnBnCtx(const DenseShapeView& data_shape, const DenseShapeView& param_shape,
                      DataType data_type) {
    const int64_t cudnn_c = param_shape.elem_cnt();
    CHECK_EQ(data_shape.elem_cnt() % cudnn_c, 0);
    const int64_t cudnn_w = data_shape.elem_cnt() / cudnn_c;
    CHECK_LT(cudnn_c, GetMaxVal<int32_t>());
    CHECK_LT(cudnn_w, GetMaxVal<int32_t>());
    data_tensor_desc_.reset(new CudnnTensorDesc(CUDNN_TENSOR_NCHW, data_type, 1,
                                                static_cast<int32_t>(cudnn_c), 1,
                                                static_cast<int32_t>(cudnn_w)));
    DataType param_dtype = data_type == DataType::kFloat16 ? DataType::kFloat : data_type;
    param_tensor_desc_.reset(new CudnnTensorDesc(CUDNN_TENSOR_NCHW, param_dtype, 1,
                                                 static_cast<int32_t>(cudnn_c), 1, 1));
#if (CUDNN_VERSION >= 7000)
    mode_ = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
#else
    mode_ = CUDNN_BATCHNORM_SPATIAL;
#endif
  }
  ~LayerNormCudnnBnCtx() = default;

  const cudnnTensorDescriptor_t& data_tensor_desc() const { return data_tensor_desc_->Get(); }
  const cudnnTensorDescriptor_t& param_tensor_desc() const { return param_tensor_desc_->Get(); }
  cudnnBatchNormMode_t mode() const { return mode_; };

 private:
  std::unique_ptr<CudnnTensorDesc> data_tensor_desc_;
  std::unique_ptr<CudnnTensorDesc> param_tensor_desc_;
  cudnnBatchNormMode_t mode_;
};

}  // namespace

template<typename T>
struct LayerNormKernelUtil<DeviceType::kGPU, T> {
  static void NormalizeForward(const DeviceCtx* ctx, const Blob* in, const Blob* scale,
                               const Blob* bias, double epsilon, Blob* out, Blob* mean,
                               Blob* inv_variance);
  static void NormalizeBackward(const DeviceCtx* ctx, const Blob* in, const Blob* scale,
                                const Blob* mean, const Blob* inv_variance, const Blob* out_diff,
                                double epsilon, Blob* in_diff, Blob* scale_diff, Blob* bias_diff);
};

template<typename T>
void LayerNormKernelUtil<DeviceType::kGPU, T>::NormalizeForward(const DeviceCtx* ctx,
                                                                const Blob* in, const Blob* scale,
                                                                const Blob* bias, double epsilon,
                                                                Blob* out, Blob* mean,
                                                                Blob* inv_variance) {
  CHECK_GE(epsilon, CUDNN_BN_MIN_EPSILON);
  LayerNormCudnnBnCtx bn_ctx(in->shape(), mean->shape(), in->data_type());
  CudaCheck(cudnnBatchNormalizationForwardTraining(
      ctx->cudnn_handle(), bn_ctx.mode(), CudnnSPOnePtr<T>(), CudnnSPZeroPtr<T>(),
      bn_ctx.data_tensor_desc(), in->dptr<T>(), bn_ctx.data_tensor_desc(), out->mut_dptr<T>(),
      bn_ctx.param_tensor_desc(), scale->dptr(), bias->dptr(), 1.0, nullptr, nullptr, epsilon,
      mean->mut_dptr(), inv_variance->mut_dptr()));
}

template<typename T>
void LayerNormKernelUtil<DeviceType::kGPU, T>::NormalizeBackward(
    const DeviceCtx* ctx, const Blob* in, const Blob* scale, const Blob* mean,
    const Blob* inv_variance, const Blob* out_diff, double epsilon, Blob* in_diff, Blob* scale_diff,
    Blob* bias_diff) {
  CHECK_GE(epsilon, CUDNN_BN_MIN_EPSILON);
  LayerNormCudnnBnCtx bn_ctx(in->shape(), scale_diff->shape(), in->data_type());
  CudaCheck(cudnnBatchNormalizationBackward(
      ctx->cudnn_handle(), bn_ctx.mode(), CudnnSPOnePtr<T>(), CudnnSPZeroPtr<T>(),
      CudnnSPOnePtr<T>(), CudnnSPZeroPtr<T>(), bn_ctx.data_tensor_desc(), in->dptr<T>(),
      bn_ctx.data_tensor_desc(), out_diff->dptr<T>(), bn_ctx.data_tensor_desc(),
      in_diff->mut_dptr<T>(), bn_ctx.param_tensor_desc(), scale->dptr(), scale_diff->mut_dptr(),
      bias_diff->mut_dptr(), epsilon, mean ? mean->dptr() : nullptr,
      inv_variance ? inv_variance->dptr() : nullptr));
}

#define INSTANTIATE_LAYER_NORM_KERNEL_UTIL_GPU(type_cpp, type_proto) \
  template struct LayerNormKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_LAYER_NORM_KERNEL_UTIL_GPU,
                     FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ)
#undef INSTANTIATE_LAYER_NORM_KERNEL_UTIL_GPU

}  // namespace oneflow
