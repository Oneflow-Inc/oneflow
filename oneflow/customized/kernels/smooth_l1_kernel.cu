#include <cstdint>
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void SmoothL1Forward(const int64_t elem_cnt, const T* x, const T* label,
                                const float beta, const float scale, T* out) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const T abs_diff = std::abs(x[i] - label[i]);
    if (abs_diff < beta) {
      out[i] = 0.5 * abs_diff * abs_diff / beta;
    } else {
      out[i] = abs_diff - 0.5 * beta;
    }
    out[i] *= scale;
  }
}

template<typename T>
__global__ void SmoothL1Backward(const int64_t elem_cnt, const T* dy, const T* x, const T* label,
                                 const float beta, const float scale, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const T diff = x[i] - label[i];
    const T abs_diff = std::abs(diff);
    if (abs_diff < beta) {
      dx[i] = diff / beta;
    } else {
      dx[i] = (diff > GetZeroVal<T>()) - (diff < GetZeroVal<T>());
    }
    dx[i] *= scale * dy[i];
  }
}

}  // namespace

template<typename T>
class SmoothL1GPUKernel final : public user_op::OpKernel {
 public:
  SmoothL1GPUKernel(user_op::KernelInitContext* ctx) : user_op::OpKernel(ctx) {}

  SmoothL1GPUKernel() = default;
  ~SmoothL1GPUKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const float beta = ctx->GetAttr<float>("beta");
    const float scale = ctx->GetAttr<float>("scale");
    const user_op::Tensor* x_blob = ctx->Tensor4ArgNameAndIndex("x", 0);
    const T* x = x_blob->dptr<T>();
    const int64_t elem_cnt = x_blob->shape().elem_cnt();
    const T* label = ctx->Tensor4ArgNameAndIndex("label", 0)->dptr<T>();
    T* y = ctx->Tensor4ArgNameAndIndex("y", 0)->mut_dptr<T>();
    SmoothL1Forward<T><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                         ctx->device_ctx()->cuda_stream()>>>(elem_cnt, x, label, beta, scale, y);
  };
};

#define REGISTER_SMOOTH_L1_GPU_KERNEL(dtype)                                                 \
  REGISTER_USER_KERNEL("smooth_l1")                                                          \
      .SetCreateFn(                                                                          \
          [](user_op::KernelInitContext* ctx) { return new SmoothL1GPUKernel<dtype>(ctx); }) \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                           \
        const user_op::TensorDesc* y_desc = ctx.TensorDesc4ArgNameAndIndex("y", 0);          \
        return ctx.device_type() == DeviceType::kGPU                                         \
               && y_desc->data_type() == GetDataType<dtype>::value;                          \
      });

REGISTER_SMOOTH_L1_GPU_KERNEL(float)
REGISTER_SMOOTH_L1_GPU_KERNEL(double)

template<typename T>
class SmoothL1GradGpuKernel final : public user_op::OpKernel {
 public:
  SmoothL1GradGpuKernel(user_op::KernelInitContext* ctx) : user_op::OpKernel(ctx) {}

  SmoothL1GradGpuKernel() = default;
  ~SmoothL1GradGpuKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const float beta = ctx->GetAttr<float>("beta");
    const float scale = ctx->GetAttr<float>("scale");
    const user_op::Tensor* x_blob = ctx->Tensor4ArgNameAndIndex("x", 0);
    const T* x = x_blob->dptr<T>();
    const int64_t elem_cnt = x_blob->shape().elem_cnt();
    const T* dy = ctx->Tensor4ArgNameAndIndex("dy", 0)->dptr<T>();
    const T* label = ctx->Tensor4ArgNameAndIndex("label", 0)->dptr<T>();
    T* dx = ctx->Tensor4ArgNameAndIndex("dx", 0)->mut_dptr<T>();
    SmoothL1Backward<T>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
           ctx->device_ctx()->cuda_stream()>>>(elem_cnt, dy, x, label, beta, scale, dx);
  };
};

#define REGISTER_SMOOTH_L1_GRAD_GPU_KERNEL(dtype)                                                \
  REGISTER_USER_KERNEL("smooth_l1")                                                              \
      .SetCreateFn(                                                                              \
          [](user_op::KernelInitContext* ctx) { return new SmoothL1GradGpuKernel<dtype>(ctx); }) \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                               \
        const user_op::TensorDesc* dx_desc = ctx.TensorDesc4ArgNameAndIndex("dx", 0);            \
        return ctx.device_type() == DeviceType::kGPU                                             \
               && dx_desc->data_type() == GetDataType<dtype>::value;                             \
      });

REGISTER_SMOOTH_L1_GRAD_GPU_KERNEL(float)
REGISTER_SMOOTH_L1_GRAD_GPU_KERNEL(double)

}  // namespace oneflow
