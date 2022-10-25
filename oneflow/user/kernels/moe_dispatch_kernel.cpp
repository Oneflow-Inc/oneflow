#include "oneflow/core/framework/framework.h"

namespace oneflow {

template <typename T>
class CpuMOEDispatchKernel final : public user_op::OpKernel {
 public:
  CpuMOEDispatchKernel() = default;
  ~CpuMOEDispatchKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* locations = ctx->Tensor4ArgNameAndIndex("locations", 0);
    const user_op::Tensor* indices = ctx->Tensor4ArgNameAndIndex("indices", 0);

    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    const T* in_ptr = in->dptr<T>();
    const int32_t* locations_ptr = locations->dptr<int32_t>();
    const int32_t* indices_ptr = indices->dptr<int32_t>();

    T* out_ptr = out->mut_dptr<T>();

    const int32_t samples = in->shape_view().At(0);
    const int32_t hidden_size = in->shape_view().At(1);
    const int32_t capacity = ctx->Attr<int32_t>("capacity");

    for (int i = 0; i < samples; ++i) {
      if (locations_ptr[i] < capacity && indices_ptr[i] >= 0) {
        for (int j = 0; j < hidden_size; ++j) {
          out_ptr[(indices_ptr[i] * capacity + locations_ptr[i]) * hidden_size + j] = \
              in_ptr[i * hidden_size + j];
        }
      }
    }  // end for
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_MOE_DISPATCH_KERNEL(dtype)                       \
  REGISTER_USER_KERNEL("moe_dispatch")                                \
      .SetCreateFn<CpuMOEDispatchKernel<dtype>>()                     \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU) \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_CPU_MOE_DISPATCH_KERNEL(float)

template <typename T>
class CpuMOECombineKernel final : public user_op::OpKernel {
 public:
  CpuMOECombineKernel() = default;
  ~CpuMOECombineKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* gates = ctx->Tensor4ArgNameAndIndex("gates", 0);
    const user_op::Tensor* locations = ctx->Tensor4ArgNameAndIndex("locations", 0);
    const user_op::Tensor* indices = ctx->Tensor4ArgNameAndIndex("indices", 0);

    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    const T* in_ptr = in->dptr<T>();
    const T* gates = gates->dptr<T>();
    const int32_t* locations_ptr = locations->dptr<int32_t>();
    const int32_t* indices_ptr = indices->dptr<int32_t>();

    T* out_ptr = out->mut_dptr<T>();

    const int32_t capacity = in->shape_view().At(1);
    const int32_t hidden_size = in->shape_view().At(2);
    const int32_t samples = gates->shape_view().At(0);

    for (int i = 0; i < samples; ++i) {
      if (locations_ptr[i] < capacity && indices_ptr[i] >= 0) {
        for (int j = 0; j < hidden_size; ++j) {
          out_ptr[i * hidden_size + j] = \
              gates[i] * in_ptr[(indices_ptr[i] * capacity + locations_ptr[i]) * hidden_size + j];
        }
      } else {
        for (int j = 0; j < hidden_size; ++j) {
          out_ptr[i * hidden_size + j] = static_cast<T>(0);
        }
      }
    } // end for
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_MOE_COMBINE_KERNEL(dtype)                        \
  REGISTER_USER_KERNEL("moe_combine")                                 \
      .SetCreateFn<CpuMOECombineKernel<dtype>>()                      \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU) \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_CPU_MOE_COMBINE_KERNEL(float)

}  // namespace oneflow
