#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace user_op {

template<DeviceType device_type, typename T>
class TransposeKernel final : public OpKernel {
 public:
  TransposeKernel() = default;
  ~TransposeKernel() = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* tensor_in = ctx->Tensor4ArgNameAndIndex("input", 0);
    Tensor* tensor_out = ctx->Tensor4ArgNameAndIndex("output", 0);
    const auto& perm = ctx->Attr<std::vector<int32_t>>("perm");
    NewKernelUtil<device_type>::Transpose(ctx->device_ctx(), tensor_in->shape().NumAxes(),
                                          tensor_in->shape(), tensor_out->shape(),
                                          StdVec2PbRf(perm), tensor_in->shape().elem_cnt(),
                                          tensor_in->dptr<T>(), tensor_out->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_TRANSPOSE_KERNEL(device, dtype)                                         \
  REGISTER_USER_KERNEL("transpose")                                                      \
      .SetCreateFn<TransposeKernel<device, dtype>>()                                     \
      .SetIsMatchedPred([](const KernelRegContext& ctx) {                                \
        const TensorDesc* tensor_in_desc = ctx.TensorDesc4ArgNameAndIndex("input", 0);   \
        const TensorDesc* tensor_out_desc = ctx.TensorDesc4ArgNameAndIndex("output", 0); \
        if (ctx.device_type() == device                                                  \
            && tensor_out_desc->data_type() == GetDataType<dtype>::value                 \
            && tensor_in_desc->data_type() == GetDataType<dtype>::value) {               \
          return true;                                                                   \
        }                                                                                \
        return false;                                                                    \
      });

REGISTER_TRANSPOSE_KERNEL(DeviceType::kCPU, int8_t)
REGISTER_TRANSPOSE_KERNEL(DeviceType::kCPU, int32_t)
REGISTER_TRANSPOSE_KERNEL(DeviceType::kCPU, int64_t)
REGISTER_TRANSPOSE_KERNEL(DeviceType::kCPU, float)
REGISTER_TRANSPOSE_KERNEL(DeviceType::kCPU, double)

REGISTER_TRANSPOSE_KERNEL(DeviceType::kGPU, int8_t)
REGISTER_TRANSPOSE_KERNEL(DeviceType::kGPU, int32_t)
REGISTER_TRANSPOSE_KERNEL(DeviceType::kGPU, int64_t)
REGISTER_TRANSPOSE_KERNEL(DeviceType::kGPU, float)
REGISTER_TRANSPOSE_KERNEL(DeviceType::kGPU, double)
REGISTER_TRANSPOSE_KERNEL(DeviceType::kGPU, float16)
}  // namespace user_op
}  // namespace oneflow
