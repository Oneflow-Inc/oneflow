#include "oneflow/core/device/cudnn_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

namespace {

class LayerNormCudnnBnCtx final {
 public:
  LayerNormCudnnBnCtx(const ShapeView& data_shape, const ShapeView& param_shape,
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
class LayerNormGpuKernel final : public user_op::OpKernel {
 public:
  LayerNormGpuKernel() = default;
  ~LayerNormGpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* cudnn_bn_scale_ones =
        ctx->Tensor4ArgNameAndIndex("cudnn_bn_scale_ones", 0);
    const user_op::Tensor* cudnn_bn_bias_zeros =
        ctx->Tensor4ArgNameAndIndex("cudnn_bn_bias_zeros", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
    user_op::Tensor* inv_variance = ctx->Tensor4ArgNameAndIndex("inv_variance", 0);
    const bool& scale = ctx->GetAttr<bool>("scale");
    const bool& center = ctx->GetAttr<bool>("center");
    user_op::Tensor* normalized = scale ? ctx->Tensor4ArgNameAndIndex("normalized", 0) : y;
    const T& epsilon = ctx->GetAttr<T>("epsilon");
    CHECK_GE(epsilon, CUDNN_BN_MIN_EPSILON);
    LayerNormCudnnBnCtx bn_ctx(x->shape(), mean->shape(), x->data_type());
    CudaCheck(cudnnBatchNormalizationForwardTraining(
        ctx->device_ctx()->cudnn_handle(), bn_ctx.mode(), CudnnSPOnePtr<T>(), CudnnSPZeroPtr<T>(),
        bn_ctx.data_tensor_desc(), x->dptr<T>(), bn_ctx.data_tensor_desc(), y->mut_dptr<T>(),
        bn_ctx.param_tensor_desc(), cudnn_bn_scale_ones->dptr(), cudnn_bn_bias_zeros->dptr(), 1.0,
        nullptr, nullptr, epsilon, mean->mut_dptr(), inv_variance->mut_dptr()));
    if (scale) {
      const user_op::Tensor* gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);
      const int64_t m = gamma->shape().elem_cnt();
      CHECK_EQ(y->shape().elem_cnt() % m, 0);
      const int64_t n = y->shape().elem_cnt() / m;
      NdarrayUtil<DeviceType::kGPU, T>::BroadcastMul(
          ctx->device_ctx(), XpuVarNdarray<T>({n, m}, y->mut_dptr<T>()),
          XpuVarNdarray<const T>({n, m}, normalized->dptr<T>()),
          XpuVarNdarray<const T>({1, m}, gamma->dptr<T>()));
    }
    if (center) {
      const user_op::Tensor* beta = ctx->Tensor4ArgNameAndIndex("beta", 0);
      const int64_t m = beta->shape().elem_cnt();
      CHECK_EQ(y->shape().elem_cnt() % m, 0);
      const int64_t n = y->shape().elem_cnt() / m;
      NdarrayUtil<DeviceType::kGPU, T>::BroadcastAdd(
          ctx->device_ctx(), XpuVarNdarray<T>({n, m}, y->mut_dptr<T>()),
          XpuVarNdarray<const T>({n, m}, y->dptr<T>()),
          XpuVarNdarray<const T>({1, m}, beta->dptr<T>()));
    }
  };
};

#define REGISTER_LAYER_NORM_GPU_KERNEL(dtype)                                       \
  REGISTER_USER_KERNEL("layer_norm")                                                \
      .SetCreateFn<LayerNormGpuKernel<dtype>>()                                     \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                  \
        const user_op::TensorDesc* x_desc = ctx.TensorDesc4ArgNameAndIndex("x", 0); \
        return ctx.device_type() == DeviceType::kGPU                                \
               && x_desc->data_type() == GetDataType<dtype>::value;                 \
      });

REGISTER_LAYER_NORM_GPU_KERNEL(float)
REGISTER_LAYER_NORM_GPU_KERNEL(double)

template<typename T>
class LayerNormGradGpuKernel final : public user_op::OpKernel {
 public:
  LayerNormGradGpuKernel() = default;
  ~LayerNormGradGpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override{
      // Add your code...
  };
};

#define REGISTER_LAYER_NORM_GRAD_GPU_KERNEL(dtype)                                    \
  REGISTER_USER_KERNEL("layer_norm_grad")                                             \
      .SetCreateFn<LayerNormGradGpuKernel<dtype>>()                                   \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                    \
        const user_op::TensorDesc* dy_desc = ctx.TensorDesc4ArgNameAndIndex("dy", 0); \
        return ctx.device_type() == DeviceType::kGPU                                  \
               && dy_desc->data_type() == GetDataType<dtype>::value;                  \
      });

REGISTER_LAYER_NORM_GRAD_GPU_KERNEL(float)
REGISTER_LAYER_NORM_GRAD_GPU_KERNEL(double)

template<typename T>
class LayerNormParamGradGpuKernel final : public user_op::OpKernel {
 public:
  LayerNormParamGradGpuKernel() = default;
  ~LayerNormParamGradGpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override{
      // Add your code...
  };
};

#define REGISTER_LAYER_NORM_PARAM_GRAD_GPU_KERNEL(dtype)                              \
  REGISTER_USER_KERNEL("layer_norm_param_grad")                                       \
      .SetCreateFn<LayerNormParamGradGpuKernel<dtype>>()                              \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                    \
        const user_op::TensorDesc* dy_desc = ctx.TensorDesc4ArgNameAndIndex("dy", 0); \
        return ctx.device_type() == DeviceType::kGPU                                  \
               && dy_desc->data_type() == GetDataType<dtype>::value;                  \
      });

REGISTER_LAYER_NORM_PARAM_GRAD_GPU_KERNEL(float)
REGISTER_LAYER_NORM_PARAM_GRAD_GPU_KERNEL(double)

}  // namespace oneflow
