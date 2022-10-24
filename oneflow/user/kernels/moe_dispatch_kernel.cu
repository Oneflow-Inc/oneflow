#include "oneflow/core/common/data_type.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace {

template<typename T>
__global__ __launch_bounds__(1024) void MOEDispatch(T* out, const T* in,
                                                    const int* __restrict__ indices,
                                                    const int* __restrict__ locations, int samples,
                                                    int hidden_size, int capacity) {
  // grid_size, blockIdx.x = 512
  // block_size, threadIdx.x = 1024
  for (int i = blockIdx.x; i < samples; i += gridDim.x) {
    if (locations[i] < capacity && indices[i] >= 0) {
#pragma unroll
      for (int j = threadIdx.x; j < hidden_size; j += 1024) {
        out[(indices[i] * capacity + locations[i]) * hidden_size + j] = in[i * hidden_size + j];
      }
    }
  }
}

}  // namespace

template <typename T>
class GpuMOEDispatchKernel final : public user_op::OpKernel {
 public:
  GpuMOEDispatchKernel() = default;
  ~GpuMOEDispatchKernel() = default;

 private:
  using user_op::OpKernel::Compute;
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

    int grid_size = 512;
    int block_size = 1024;

    MOEDispatch<<<grid_size, block_size, 0, ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
        out_ptr, in_ptr, indices_ptr, locations_ptr, samples, hidden_size, capacity);

  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_MOE_DISPATCH_KERNEL(dtype)                       \
  REGISTER_USER_KERNEL("moe_dispatch")                                \
      .SetCreateFn<GpuMOEDispatchKernel<dtype>>()                     \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_GPU_MOE_DISPATCH_KERNEL(float)

}  // namespace oneflow
