#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T>
class SoftmaxKernel final : public user_op::OpKernel {
 public:
  SoftmaxKernel() = default;
  ~SoftmaxKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("out", 0);
    NewKernelUtil<device_type>::Relu(ctx->device_ctx(), x->shape().elem_cnt(), x->dptr<T>(),
                                     y->mut_dptr<T>());
  };
};

//template<typename T>
size_t InferTmpSize(user_op::InferContext* ctx, const std::string& bn) {
  const Shape* in_shape = ctx->Shape4ArgNameAndIndex(bn, 0);
  const int32_t instance_size = in_shape->dim_vec().back();
  const int32_t instance_num = in_shape->elem_cnt() / instance_size;

  /* softmax_num */
  int32_t softmax_num_bytes = 10;
  //    GetCudaAlignedSize(instance_num * sizeof(cub::KeyValuePair<int32_t, dtype>));

  /* buf */
  size_t buf_bytes = 10;

  return softmax_num_bytes + buf_bytes;
}

#define REGISTER_SOFTMAX_KERNEL(device, dtype)                                                  \
  REGISTER_USER_KERNEL("softmax")                                                               \
      .SetCreateFn<SoftmaxKernel<device, dtype>>()                                              \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                              \
        const user_op::TensorDesc* y_desc = ctx.TensorDesc4ArgNameAndIndex("out", 0);           \
        return ctx.device_type() == device && y_desc->data_type() == GetDataType<dtype>::value; \
      })                                                                                        \
      .SetInferTmpSizeFn(InferTmpSize, "in");

REGISTER_SOFTMAX_KERNEL(DeviceType::kCPU, float)
REGISTER_SOFTMAX_KERNEL(DeviceType::kCPU, double)
REGISTER_SOFTMAX_KERNEL(DeviceType::kGPU, float)
REGISTER_SOFTMAX_KERNEL(DeviceType::kGPU, double)
REGISTER_SOFTMAX_KERNEL(DeviceType::kGPU, float16)

template<DeviceType device_type, typename T>
class SoftmaxGradKernel final : public user_op::OpKernel {
 public:
  SoftmaxGradKernel() = default;
  ~SoftmaxGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* y_blob = ctx->Tensor4ArgNameAndIndex("y", 0);
    const user_op::Tensor* dy_blob = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx_blob = ctx->Tensor4ArgNameAndIndex("dx", 0);
    NewKernelUtil<device_type>::ReluBackward(ctx->device_ctx(), y_blob->shape().elem_cnt(),
                                             y_blob->dptr<T>(), y_blob->dptr<T>(),
                                             dy_blob->dptr<T>(), dx_blob->mut_dptr<T>());
  };
};

#define REGISTER_SOFTMAX_GRAD_KERNEL(device, dtype)                                              \
  REGISTER_USER_KERNEL("softmax_grad")                                                           \
      .SetCreateFn<SoftmaxGradKernel<device, dtype>>()                                           \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                               \
        const user_op::TensorDesc* dx_desc = ctx.TensorDesc4ArgNameAndIndex("dx", 0);            \
        return ctx.device_type() == device && dx_desc->data_type() == GetDataType<dtype>::value; \
      })                                                                                         \
      .SetInferTmpSizeFn(InferTmpSize, "dx");

REGISTER_SOFTMAX_GRAD_KERNEL(DeviceType::kCPU, float)
REGISTER_SOFTMAX_GRAD_KERNEL(DeviceType::kCPU, double)
REGISTER_SOFTMAX_GRAD_KERNEL(DeviceType::kGPU, float)
REGISTER_SOFTMAX_GRAD_KERNEL(DeviceType::kGPU, double)
REGISTER_SOFTMAX_GRAD_KERNEL(DeviceType::kGPU, float16)

}  // namespace

}  // namespace oneflow
