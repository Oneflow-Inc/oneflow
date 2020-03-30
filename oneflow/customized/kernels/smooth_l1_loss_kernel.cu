#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void SmoothL1Forward(const int64_t elem_cnt, const T* prediction, const T* label,
                                const T beta, T* loss) {
  const T half_beta = static_cast<T>(0.5) * beta;
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const T abs_diff = std::abs(prediction[i] - label[i]);
    if (abs_diff < beta) {
      loss[i] = static_cast<T>(0.5) * abs_diff * abs_diff / beta;
    } else {
      loss[i] = abs_diff - half_beta;
    }
  }
}

template<typename T>
__global__ void SmoothL1Backward(const int64_t elem_cnt, const T* loss_grad, const T* prediction,
                                 const T* label, const T beta, T* prediction_grad) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) {
    const T diff = prediction[i] - label[i];
    const T abs_diff = std::abs(diff);
    if (abs_diff < beta) {
      prediction_grad[i] = diff / beta;
    } else {
      prediction_grad[i] = (diff > GetZeroVal<T>()) - (diff < GetZeroVal<T>());
    }
    prediction_grad[i] = prediction_grad[i] * loss_grad[i];
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
    const user_op::Tensor* prediction_blob = ctx->Tensor4ArgNameAndIndex("prediction", 0);
    const T* prediction = prediction_blob->dptr<T>();
    const int64_t elem_cnt = prediction_blob->shape().elem_cnt();
    const T* label = ctx->Tensor4ArgNameAndIndex("label", 0)->dptr<T>();
    T* loss = ctx->Tensor4ArgNameAndIndex("loss", 0)->mut_dptr<T>();
    SmoothL1Forward<T>
        <<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
           ctx->device_ctx()->cuda_stream()>>>(elem_cnt, prediction, label, beta, loss);
  };
};

#define REGISTER_SMOOTH_L1_GPU_KERNEL(dtype)                                                 \
  REGISTER_USER_KERNEL("smooth_l1_loss")                                                     \
      .SetCreateFn(                                                                          \
          [](user_op::KernelInitContext* ctx) { return new SmoothL1GPUKernel<dtype>(ctx); }) \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                           \
        const user_op::TensorDesc* loss_desc = ctx.TensorDesc4ArgNameAndIndex("loss", 0);    \
        return ctx.device_type() == DeviceType::kGPU                                         \
               && loss_desc->data_type() == GetDataType<dtype>::value;                       \
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
    const user_op::Tensor* prediction_blob = ctx->Tensor4ArgNameAndIndex("prediction", 0);
    const T* prediction = prediction_blob->dptr<T>();
    const int64_t elem_cnt = prediction_blob->shape().elem_cnt();
    const T* loss_grad = ctx->Tensor4ArgNameAndIndex("loss_grad", 0)->dptr<T>();
    const T* label = ctx->Tensor4ArgNameAndIndex("label", 0)->dptr<T>();
    T* prediction_grad = ctx->Tensor4ArgNameAndIndex("prediction_grad", 0)->mut_dptr<T>();
    SmoothL1Backward<T><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                          ctx->device_ctx()->cuda_stream()>>>(elem_cnt, loss_grad, prediction,
                                                              label, beta, prediction_grad);
  };
};

#define REGISTER_SMOOTH_L1_GRAD_GPU_KERNEL(dtype)                                                \
  REGISTER_USER_KERNEL("smooth_l1_loss_grad")                                                    \
      .SetCreateFn(                                                                              \
          [](user_op::KernelInitContext* ctx) { return new SmoothL1GradGpuKernel<dtype>(ctx); }) \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                               \
        const user_op::TensorDesc* prediction_grad_desc =                                        \
            ctx.TensorDesc4ArgNameAndIndex("prediction_grad", 0);                                \
        return ctx.device_type() == DeviceType::kGPU                                             \
               && prediction_grad_desc->data_type() == GetDataType<dtype>::value;                \
      });

REGISTER_SMOOTH_L1_GRAD_GPU_KERNEL(float)
REGISTER_SMOOTH_L1_GRAD_GPU_KERNEL(double)

}  // namespace oneflow
