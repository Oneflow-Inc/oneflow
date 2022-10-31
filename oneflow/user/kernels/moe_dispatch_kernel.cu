#include "oneflow/core/common/data_type.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

namespace {

template<typename T>
__global__ __launch_bounds__(1024) void MOEDispatch(T* out, const T* in,
                                                    const T* __restrict__ gates,
                                                    const int* __restrict__ indices,
                                                    const int* __restrict__ locations, int samples,
                                                    int hidden_size, int capacity) {
  // grid_size, blockIdx.x = 512
  // block_size, threadIdx.x = 1024
  for (int i = blockIdx.x; i < samples; i += gridDim.x) {
    if (locations[i] < capacity && indices[i] >= 0) {
      T gate = (gates == nullptr ? static_cast<T>(1.0) : gates[i]);
#pragma unroll
      for (int j = threadIdx.x; j < hidden_size; j += 1024) {
        out[(indices[i] * capacity + locations[i]) * hidden_size + j] = gate * in[i * hidden_size + j];
      }
    }
  }
}

template<typename T>
__global__ __launch_bounds__(1024) void MOECombine(T* out, const T* in,
                                                   const T* __restrict__ gates,
                                                   const int* __restrict__ indices,
                                                   const int* __restrict__ locations, int samples,
                                                   int hidden_size, int capacity) {
  // grid_size, blockIdx.x = 512
  // block_size, threadIdx.x = 1024
  for (int i = blockIdx.x; i < samples; i += gridDim.x) {
    if (locations[i] < capacity && indices[i] >= 0) {
      T gate = (gates == nullptr ? static_cast<T>(1.0) : gates[i]);
#pragma unroll
      for (int j = threadIdx.x; j < hidden_size; j += 1024) {
        out[i * hidden_size + j] =
            gate * in[(indices[i] * capacity + locations[i]) * hidden_size + j];
      }
    } else {
      for (int j = threadIdx.x; j < hidden_size; j += 1024) {
        out[i * hidden_size + j] = static_cast<T>(0);
      }
    }
  }
}

template <typename T>
__global__ __launch_bounds__(32) void MOEGateGrad(
    T* grad_gates,
    const T* reshaped_input,
    const T* dispatched_input,
    const int* __restrict__ indices,
    const int* __restrict__ locations,
    int samples, int hidden_size, int capacity) {
  // grid_size, blockIdx.x = 512
  // block_size, threadIdx.x = 32
  for (int index = blockIdx.x; index < samples; index += gridDim.x) {
    if (locations[index] >= capacity || indices[index] < 0) {
      if (threadIdx.x == 0) grad_gates[index] = static_cast<T>(0);
      continue;
    }

    int indice = indices[index] * capacity + locations[index];
    T grad_gates_rf = static_cast<T>(0);

    for (int i = threadIdx.x; i < hidden_size; i += 32) {
      grad_gates_rf += dispatched_input[indice * hidden_size + i] * reshaped_input[index * hidden_size + i];
    }

    T red_buf;
    unsigned int mask;
    T t;
    red_buf = grad_gates_rf;
    mask = __activemask();
    red_buf += __shfl_down_sync(mask, red_buf, 16, 32);
    red_buf += __shfl_down_sync(mask, red_buf, 8, 32);
    red_buf += __shfl_down_sync(mask, red_buf, 4, 32);
    red_buf += __shfl_down_sync(mask, red_buf, 2, 32);
    red_buf += __shfl_down_sync(mask, red_buf, 1, 32);
    red_buf = __shfl_sync(mask, red_buf, 0, 32);

    if (threadIdx.x == 0) grad_gates[index] = red_buf;
  }
}

}  // namespace

/********** GpuMOEDispatch Kernel ***********/
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
    const T* in_ptr = in->dptr<T>();
    const int32_t* locations_ptr = locations->dptr<int32_t>();
    const int32_t* indices_ptr = indices->dptr<int32_t>();

    const T* gates_ptr = nullptr;
    if (ctx->has_input("gates", 0)) {
      gates_ptr = ctx->Tensor4ArgNameAndIndex("gates", 0)->dptr<T>();
    }

    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    T* out_ptr = out->mut_dptr<T>();

    const int32_t samples = in->shape_view().At(0);
    const int32_t hidden_size = in->shape_view().At(1);
    const int32_t capacity = ctx->Attr<int32_t>("capacity");

    AutoMemset(ctx->stream(), out->mut_dptr(), 0,
               out->shape_view().elem_cnt() * sizeof(T),
               out->mem_case());

    int grid_size = 512;
    int block_size = 1024;

    MOEDispatch<<<grid_size, block_size, 0, ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
        out_ptr, in_ptr, gates_ptr, indices_ptr, locations_ptr, samples, hidden_size, capacity);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_MOE_DISPATCH_KERNEL(dtype)                       \
  REGISTER_USER_KERNEL("moe_dispatch")                                \
      .SetCreateFn<GpuMOEDispatchKernel<dtype>>()                     \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_GPU_MOE_DISPATCH_KERNEL(float)

/********** GpuMOECombine Kernel ***********/
template <typename T>
class GpuMOECombineKernel final : public user_op::OpKernel {
 public:
  GpuMOECombineKernel() = default;
  ~GpuMOECombineKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* locations = ctx->Tensor4ArgNameAndIndex("locations", 0);
    const user_op::Tensor* indices = ctx->Tensor4ArgNameAndIndex("indices", 0);
    const T* in_ptr = in->dptr<T>();
    const int32_t* locations_ptr = locations->dptr<int32_t>();
    const int32_t* indices_ptr = indices->dptr<int32_t>();

    const T* gates_ptr = nullptr;
    if (ctx->has_input("gates", 0)) {
      gates_ptr = ctx->Tensor4ArgNameAndIndex("gates", 0)->dptr<T>();
    }

    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    T* out_ptr = out->mut_dptr<T>();

    const int32_t capacity = in->shape_view().At(1);
    const int32_t hidden_size = in->shape_view().At(2);
    const int32_t samples = indices->shape_view().At(0);

    int grid_size = 512;
    int block_size = 1024;

    MOECombine<<<grid_size, block_size, 0, ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
        out_ptr, in_ptr, gates_ptr, indices_ptr, locations_ptr, samples, hidden_size, capacity);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_MOE_COMBINE_KERNEL(dtype)                         \
  REGISTER_USER_KERNEL("moe_combine")                                  \
      .SetCreateFn<GpuMOECombineKernel<dtype>>()                       \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_GPU_MOE_COMBINE_KERNEL(float)

/********** GpuMOEGateGrad Kernel ***********/
template <typename T>
class GpuMOEGateGradKernel final : public user_op::OpKernel {
 public:
  GpuMOEGateGradKernel() = default;
  ~GpuMOEGateGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* dispatched = ctx->Tensor4ArgNameAndIndex("dispatched", 0);
    const user_op::Tensor* locations = ctx->Tensor4ArgNameAndIndex("locations", 0);
    const user_op::Tensor* indices = ctx->Tensor4ArgNameAndIndex("indices", 0);
    const T* in_ptr = in->dptr<T>();
    const T* dispatched_ptr = dispatched->dptr<T>();
    const int32_t* locations_ptr = locations->dptr<int32_t>();
    const int32_t* indices_ptr = indices->dptr<int32_t>();

    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    T* out_ptr = out->mut_dptr<T>();

    const int32_t capacity = dispatched->shape_view().At(1);
    const int32_t hidden_size = in->shape_view().At(1);
    const int32_t samples = in->shape_view().At(0);

    int grid_size = 512;
    int block_size = 32;

    MOEGateGrad<<<grid_size, block_size, 0, ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
        out_ptr, in_ptr, dispatched_ptr, indices_ptr, locations_ptr, samples, hidden_size, capacity);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_MOE_GATE_GRAD_KERNEL(dtype)                        \
  REGISTER_USER_KERNEL("moe_gate_grad")                                 \
      .SetCreateFn<GpuMOEGateGradKernel<dtype>>()                       \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)  \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_GPU_MOE_GATE_GRAD_KERNEL(float)

}  // namespace oneflow
