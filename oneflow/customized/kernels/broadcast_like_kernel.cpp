#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"

namespace oneflow {

namespace user_op {

template<DeviceType device_type, typename T>
class BroadcastLikeKernel final : public OpKernel {
 public:
  BroadcastLikeKernel() = default;
  ~BroadcastLikeKernel() = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* in_tenor = ctx->Tensor4ArgNameAndIndex("x", 0);
    const Tensor* like_tensor = ctx->Tensor4ArgNameAndIndex("like", 0);
    Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
    const int64_t num_axes = out_tensor->shape().NumAxes();
    const auto& axis = ctx->GetAttr<std::vector<int32_t>>("axis");
    const Shape& reduced_shape =
        CreateReducedShapeOrOnesShape(like_tensor->shape(), {axis.begin(), axis.end()});
    NdarrayUtil<device_type, T>::BroadcastTo(
        ctx->device_ctx(), XpuVarNdarray<T>(out_tensor->shape(), out_tensor->mut_dptr<T>()),
        XpuVarNdarray<const T>(reduced_shape, in_tenor->dptr<T>()));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device, typename T>
bool IsMatchedPred(const KernelRegContext& ctx) {
  const TensorDesc* output_tensor_desc = ctx.TensorDesc4ArgNameAndIndex("out_tensor", 0);
  if (ctx.device_type() == device && output_tensor_desc->data_type() == GetDataType<T>::value) {
    return true;
  }
  return false;
}

#define REGISTER_BROADCAST_LIKE_XPU_KERNEL(device, dtype) \
  REGISTER_USER_KERNEL("broadcast_like")                  \
      .SetCreateFn<BroadcastLikeKernel<device, dtype>>()  \
      .SetIsMatchedPred(IsMatchedPred<device, dtype>);

#define REGISTER_BROADCAST_LIKE_KERNEL(dtype)                 \
  REGISTER_BROADCAST_LIKE_XPU_KERNEL(DeviceType::kCPU, dtype) \
  REGISTER_BROADCAST_LIKE_XPU_KERNEL(DeviceType::kGPU, dtype)

REGISTER_BROADCAST_LIKE_KERNEL(float)
REGISTER_BROADCAST_LIKE_KERNEL(double)
REGISTER_BROADCAST_LIKE_KERNEL(int32_t)
REGISTER_BROADCAST_LIKE_KERNEL(int64_t)

}  // namespace user_op
}  // namespace oneflow