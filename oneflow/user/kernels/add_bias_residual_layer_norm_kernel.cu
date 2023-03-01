#include "oneflow/core/device/cudnn_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/cuda/atomic.cuh"
#include <cub/cub.cuh>
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/ep/include/primitive/fill.h"
#include "oneflow/core/ep/include/primitive/matmul.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/cuda/layer_norm.cuh"
#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif  // CUDA_VERSION >= 11000

namespace oneflow {

namespace {

template<typename T, bool do_scale, bool do_center, bool do_bias, size_t nb_residual>
void AddBiasResidualLayerNormForwardGpu(ep::Stream* stream, const int64_t num_instances,
                        const int64_t norm_size, const double epsilon, const T* x_ptr,
                        const T* gamma_ptr, const T* beta_ptr, const T* pre_bias_ptr, 
                        const T* pre_residual_1_ptr, const T* pre_residual_2_ptr, 
                        const double pre_alpha_1, const double pre_alpha_2, T* y_ptr,
                        user_op::Tensor* mean, user_op::Tensor* inv_variance){
    using ComputeType = typename cuda::layer_norm::DefaultComputeType<T>::type;   
    // TODO: 
}

template<typename T>
void DispatchAddBiasResidualLayerNormForwardGpu(ep::Stream* stream, const int64_t num_instances,
                        const int64_t norm_size, const double epsilon, const T* x_ptr,
                        const T* gamma_ptr, const T* beta_ptr, const T* pre_bias_ptr, 
                        const T* pre_residual_1_ptr, const T* pre_residual_2_ptr, 
                        const double pre_alpha_1, const double pre_alpha_2, T* y_ptr,
                        user_op::Tensor* mean, user_op::Tensor* inv_variance){

#define DISPATCH_BY_GAMMA_BETA_BIAS(has_gamma, has_beta, has_pre_bias) \
    if(pre_residual_1_ptr != nullptr && pre_residual_2_ptr != nullptr){ \
        AddBiasResidualLayerNormForwardGpu<has_gamma, has_beta, has_pre_bias, 2>(stream, num_instances, norm_size, \
            epsilon, x_ptr, gamma_ptr, beta_ptr, pre_bias_ptr, pre_residual_1_ptr, \
            pre_residual_2_ptr, pre_alpha_1, pre_alpha_2, y_ptr, mean, inv_variance); \
    } else if (pre_residual_1_ptr == nullptr && pre_residual_2_ptr != nullptr){ \
        AddBiasResidualLayerNormForwardGpu<has_gamma, has_beta, has_pre_bias, 1>(stream, num_instances, norm_size, \
            epsilon, x_ptr, gamma_ptr, beta_ptr, pre_bias_ptr, pre_residual_1_ptr, \
            pre_residual_2_ptr, pre_alpha_1, pre_alpha_2, y_ptr, mean, inv_variance); \
    } else if (pre_residual_1_ptr != nullptr && pre_residual_2_ptr == nullptr){ \
        AddBiasResidualLayerNormForwardGpu<has_gamma, has_beta, has_pre_bias, 1>(stream, num_instances, norm_size, \
            epsilon, x_ptr, gamma_ptr, beta_ptr, pre_bias_ptr, pre_residual_1_ptr, \
            pre_residual_2_ptr, pre_alpha_1, pre_alpha_2, y_ptr, mean, inv_variance); \
    } else { \
        AddBiasResidualLayerNormForwardGpu<has_gamma, has_beta, has_pre_bias, 0>(stream, num_instances, norm_size, \
            epsilon, x_ptr, gamma_ptr, beta_ptr, pre_bias_ptr, pre_residual_1_ptr, \
            pre_residual_2_ptr, pre_alpha_1, pre_alpha_2, y_ptr, mean, inv_variance); \
    }

    if(gamma_ptr != nullptr && beta_ptr != nullptr){
        if(pre_bias_ptr != nullptr){ DISPATCH_BY_GAMMA_BETA_BIAS(true, true, true); } 
        else { DISPATCH_BY_GAMMA_BETA_BIAS(true, true, false); }
    } else if (gamma_ptr != nullptr && beta_ptr == nullptr) {
        if(pre_bias_ptr != nullptr){ DISPATCH_BY_GAMMA_BETA_BIAS(false, true, true); } 
        else { DISPATCH_BY_GAMMA_BETA_BIAS(false, true, false); }
    } else if (gamma_ptr == nullptr && beta_ptr != nullptr) {
        if(pre_bias_ptr != nullptr){ DISPATCH_BY_GAMMA_BETA_BIAS(true, false, true); } 
        else { DISPATCH_BY_GAMMA_BETA_BIAS(true, false, false); }
    } else {
        if(pre_bias_ptr != nullptr){ DISPATCH_BY_GAMMA_BETA_BIAS(false, false, true); } 
        else { DISPATCH_BY_GAMMA_BETA_BIAS(false, false, false); }
    }

#undef DISPATCH_BY_GAMMA_BETA_BIAS
}

template<typename T>
class AddBiasResidualLayerNormGpuKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  AddBiasResidualLayerNormGpuKernel() = default;
  ~AddBiasResidualLayerNormGpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override {
    // obtain x and check its shape
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const ShapeView &x_shape = x->shape_view();
    CHECK_GT(x_shape.NumAxes(), 1)
      << "number of axes of \'x\' should have be greater than 1, yet get " << x_shape.NumAxes();

#define GET_GAMMA_BETA_BIAS_AND_SHAPE_CHECK(tensor) \
    if (ctx->has_input(#tensor, 0)) { \
        const user_op::Tensor* tensor = ctx->Tensor4ArgNameAndIndex(#tensor, 0); \
        tensor##_shape = tensor->shape_view(); \
        tensor##_ptr = tensor->dptr<T>(); \
        CHECK_EQ(tensor##_shape.NumAxes(), 1) \
            << "number of axes of \'" << #tensor << "\' should have be greater than 1, yet get " \
            << tensor##_shape.NumAxes(); \
        CHECK_EQ(tensor##_shape.At(0), x_shape.At(x_shape.NumAxes() - 1)) \
            << "dimension 1 of \'" << #tensor << "\'(" << tensor##_shape.At(0) \
            << ") is not consistant with the last dimension of \'x\'(" \
            << x_shape.At(x_shape.NumAxes() - 1) << ")"; \
    }

    // obtain gamma and check its shape
    const T* gamma_ptr = nullptr;
    ShapeView gamma_shape;
    GET_GAMMA_BETA_BIAS_AND_SHAPE_CHECK(gamma);
    
    // obtain beta and check its shape
    const T* beta_ptr = nullptr;
    ShapeView beta_shape;
    GET_GAMMA_BETA_BIAS_AND_SHAPE_CHECK(beta);

    // obtain pre_bias and check its shape
    const T* pre_bias_ptr = nullptr;
    ShapeView pre_bias_shape;
    GET_GAMMA_BETA_BIAS_AND_SHAPE_CHECK(pre_bias);

#undef GET_GAMMA_BETA_BIAS_AND_SHAPE_CHECK

    // obtain residual and check their shape
    const T* pre_residual_1_ptr = nullptr;
    const T* pre_residual_2_ptr = nullptr;
    ShapeView pre_residual_1_shape;
    ShapeView pre_residual_2_shape;
    if (ctx->has_input("pre_residual_1", 0)) {
        const user_op::Tensor* pre_residual_1 = ctx->Tensor4ArgNameAndIndex("pre_residual_1", 0);
        pre_residual_1_shape = pre_residual_1->shape_view();
        pre_residual_1_ptr = pre_residual_1->dptr<T>();
        CHECK_EQ(pre_residual_1_shape, x_shape);
    }
    if (ctx->has_input("pre_residual_2", 0)) {
        CHECK(ctx->has_input("pre_residual_1", 0))
            << "must provide pre_residual_1 while pre_residual_2 is provided";
        const user_op::Tensor* pre_residual_2 = ctx->Tensor4ArgNameAndIndex("pre_residual_2", 0);
        pre_residual_2_shape = pre_residual_2->shape_view();
        pre_residual_2_ptr = pre_residual_2->dptr<T>();
        CHECK_EQ(pre_residual_2_shape, x_shape);
    }

    // obtain epsilon and check its value
    const double epsilon = ctx->Attr<double>("epsilon");
    CHECK_GE(epsilon, CUDNN_BN_MIN_EPSILON);

    // obtain output tensors
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
    user_op::Tensor* inv_variance = ctx->Tensor4ArgNameAndIndex("inv_variance", 0);
    const ShapeView &y_shape = y->shape_view();
    const ShapeView &mean_shape = mean->shape_view();
    const ShapeView &inv_variance_shape = inv_variance->shape_view();

  }
};

} // namespace

#define REGISTER_ADD_BIAS_RESIDUAL_LAYER_NORM_CUDA_KERNEL(dtype) \
    REGISTER_USER_KERNEL("add_bias_residual_layer_norm") \
        .SetCreateFn<AddBiasResidualLayerNormGpuKernel<dtype>>() \
        .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_ADD_BIAS_RESIDUAL_LAYER_NORM_CUDA_KERNEL(float)
REGISTER_ADD_BIAS_RESIDUAL_LAYER_NORM_CUDA_KERNEL(double)
REGISTER_ADD_BIAS_RESIDUAL_LAYER_NORM_CUDA_KERNEL(half)
#if CUDA_VERSION >= 11000
REGISTER_ADD_BIAS_RESIDUAL_LAYER_NORM_CUDA_KERNEL(nv_bfloat16)
#endif

} // namespace oneflow