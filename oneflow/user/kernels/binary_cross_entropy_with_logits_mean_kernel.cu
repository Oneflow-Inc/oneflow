/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/user/kernels/binary_cross_entropy_with_logits_mean_kernel_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/cuda/elementwise.cuh"
#include <cub/cub.cuh>
#include "oneflow/core/kernel/cuda_graph_support.h"

namespace oneflow {

namespace user_op {

namespace {

constexpr int32_t kBlockSize = 1024;
constexpr int32_t kReduceLocalSumBlockSize = 1024;
constexpr int32_t kSingleBlockProcessNumThreshold = 1024;

template<typename T>
struct DefaultComputeType {
  using type = T;
};

template<>
struct DefaultComputeType<half> {
  using type = float;
};

template<class Func>
inline cudaError_t GetNumBlocks(Func func, int64_t block_size, size_t dynamic_smem_size,
                                int64_t max_blocks, int64_t waves, int* num_blocks) {
  int dev;
  {
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) { return err; }
  }
  int sm_count;
  {
    cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) { return err; }
  }
  int max_active_blocks;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, func,
                                                                    block_size, dynamic_smem_size);
  }
  *num_blocks =
      std::max<int>(1, std::min<int64_t>(max_blocks, sm_count * max_active_blocks * waves));
  return cudaSuccess;
}

template<typename T>
__device__ __forceinline__ T Sigmoid(const T x) {
  const T half_of_one = static_cast<T>(0.5);
  return half_of_one * tanh(half_of_one * x) + half_of_one;
}

template<>
__device__ __forceinline__ half Sigmoid(const half x) {
  return __float2half(Sigmoid(__half2float(x)));
}

template<typename INPUT_T, typename TARGET_T, typename OUTPUT_T, typename ComputeType>
__global__ void FusedBinaryCrossEntropyWithLogitsReduceMeanKernel(const INPUT_T* input,
                                                                  const TARGET_T* target,
                                                                  OUTPUT_T* out,
                                                                  const int64_t local_elem_cnt,
                                                                  const int64_t reduce_elem_cnt) {
  ComputeType zero = static_cast<ComputeType>(0.0);
  ComputeType one = static_cast<ComputeType>(1.0);
  using BlockReduce = cub::BlockReduce<ComputeType, kBlockSize>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  ComputeType reduce_sum = static_cast<ComputeType>(0.0);
  CUDA_1D_KERNEL_LOOP(i, local_elem_cnt) {
    const ComputeType input_val = static_cast<ComputeType>(input[i]);
    const ComputeType target_val = static_cast<ComputeType>(target[i]);
    const ComputeType max_val = -input_val < zero ? zero : -input_val;
    const ComputeType result =
        (one - target_val) * input_val + max_val + (log(exp(-max_val) + exp(-input_val - max_val)));
    reduce_sum += result;
  }

  const ComputeType block_reduce_sum = BlockReduce(temp_storage).Sum(reduce_sum);
  if (threadIdx.x == 0) {
    out[blockIdx.x] = static_cast<OUTPUT_T>(block_reduce_sum / reduce_elem_cnt);
  }
}

template<typename TARGET_T, typename INPUT_T>
__global__ void ReduceLocalSumKernel(INPUT_T* block_local_sum_buf, TARGET_T* out,
                                     int64_t elem_cnt) {
  using BlockReduce = cub::BlockReduce<INPUT_T, kReduceLocalSumBlockSize>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  INPUT_T reduce_sum = 0.0;
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) { reduce_sum += block_local_sum_buf[i]; }
  const INPUT_T block_reduce_sum = BlockReduce(temp_storage).Sum(reduce_sum);
  if (threadIdx.x == 0) { out[0] = block_reduce_sum; }
}

template<typename INPUT_T, typename TARGET_T, typename ComputeType>
struct BinaryCrossEntropyWithLogitsReduceMeanGradFunctor {
  OF_DEVICE_FUNC explicit BinaryCrossEntropyWithLogitsReduceMeanGradFunctor(
      const INPUT_T elem_cnt_reciprocal, const TARGET_T dy)
      : elem_cnt_reciprocal(elem_cnt_reciprocal), dy(dy) {}
  __device__ ComputeType operator()(const INPUT_T input_val, const TARGET_T target_val) const {
    const ComputeType input_val_ = static_cast<ComputeType>(input_val);
    const ComputeType target_val_ = static_cast<ComputeType>(target_val);
    const ComputeType dy_ = static_cast<ComputeType>(dy);
    const ComputeType elem_cnt_reciprocal_ = static_cast<ComputeType>(elem_cnt_reciprocal);
    return (Sigmoid(input_val_) - target_val_) * dy_ * elem_cnt_reciprocal_;
  }
  const TARGET_T dy;
  const INPUT_T elem_cnt_reciprocal;
};

template<typename INPUT_T, typename TARGET_T, typename ComputeType>
struct BinaryCrossEntropyWithLogitsReduceMeanGradDyptrFunctor {
  OF_DEVICE_FUNC explicit BinaryCrossEntropyWithLogitsReduceMeanGradDyptrFunctor(
      const int32_t elem_cnt, const TARGET_T* dy_ptr)
      : elem_cnt_reciprocal(1.0f / elem_cnt), dy_ptr(dy_ptr) {}
  __device__ BinaryCrossEntropyWithLogitsReduceMeanGradFunctor<INPUT_T, TARGET_T, ComputeType>
  operator()() const {
    return BinaryCrossEntropyWithLogitsReduceMeanGradFunctor<INPUT_T, TARGET_T, ComputeType>(
        elem_cnt_reciprocal, *dy_ptr);
  }
  const TARGET_T* dy_ptr;
  const INPUT_T elem_cnt_reciprocal;
};

template<typename INPUT_T, typename TARGET_T, typename ComputeType>
__global__ void FusedBCEReduceMeanFwBwKernel(const INPUT_T* input, const TARGET_T* target,
                                             TARGET_T* out, INPUT_T* input_grad,
                                             const ComputeType constant_output_grad,
                                             const ComputeType elem_cnt_reciprocal,
                                             const int32_t local_elem_cnt,
                                             const int32_t reduce_elem_cnt) {
  ComputeType zero = static_cast<ComputeType>(0.0);
  ComputeType one = static_cast<ComputeType>(1.0);
  BinaryCrossEntropyWithLogitsReduceMeanGradFunctor<INPUT_T, TARGET_T, ComputeType> grad_functor(
      elem_cnt_reciprocal, constant_output_grad);
  using BlockReduce = cub::BlockReduce<ComputeType, kBlockSize>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  TARGET_T reduce_sum = 0.0;
  CUDA_1D_KERNEL_LOOP(i, local_elem_cnt) {
    const INPUT_T input_val = input[i];
    const TARGET_T target_val = target[i];
    input_grad[i] = grad_functor(input_val, target_val);
    const ComputeType input_val_ = static_cast<ComputeType>(input_val);
    const ComputeType target_val_ = static_cast<ComputeType>(target_val);
    const ComputeType max_val = -input_val_ < zero ? zero : -input_val_;
    const ComputeType result = (one - target_val_) * input_val_ + max_val
                               + (log(exp(-max_val) + exp(-input_val_ - max_val)));
    reduce_sum += result;
  }
  const ComputeType block_reduce_sum = BlockReduce(temp_storage).Sum(reduce_sum);
  if (threadIdx.x == 0) {
    out[blockIdx.x] = static_cast<TARGET_T>(block_reduce_sum / reduce_elem_cnt);
  }
}

template<typename INPUT_T, typename TARGET_T>
class FusedBCEMeanFwBwKernel final : public user_op::OpKernel, public CudaGraphSupport {
 public:
  FusedBCEMeanFwBwKernel() = default;
  ~FusedBCEMeanFwBwKernel() override = default;
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return CreateBCEWithLogitsReduceMeanKernelCache(ctx);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache* cache) const override {
    const auto* input_blob = ctx->Tensor4ArgNameAndIndex("input", 0);
    const auto* target_blob = ctx->Tensor4ArgNameAndIndex("target", 0);
    auto* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    auto* dx_blob = ctx->Tensor4ArgNameAndIndex("dx", 0);

    int64_t local_elem_cnt = input_blob->shape_view().elem_cnt();
    int64_t reduce_elem_cnt = local_elem_cnt;

    if (cache != nullptr) {
      // Because `out`'s SBP maybe P or B, we need to use reduce_elem_cnt as reduce_mean factor.
      const auto* bce_cache = dynamic_cast<const BCEWithLogitsReduceMeanKernelCache*>(cache);
      CHECK_NOTNULL(bce_cache);
      reduce_elem_cnt = bce_cache->reduce_elem_cnt();
    }

    const INPUT_T* input = input_blob->dptr<INPUT_T>();
    const TARGET_T* target = target_blob->dptr<TARGET_T>();
    using ComputeType = typename DefaultComputeType<TARGET_T>::type;
    ComputeType constant_output_grad = ctx->Attr<double>("constant_value");
    ComputeType elem_cnt_reciprocal = static_cast<ComputeType>(1) / reduce_elem_cnt;

    if (local_elem_cnt <= kSingleBlockProcessNumThreshold) {
      FusedBCEReduceMeanFwBwKernel<INPUT_T, TARGET_T, ComputeType>
          <<<1, kBlockSize, 0, ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
              input_blob->dptr<INPUT_T>(), target_blob->dptr<TARGET_T>(),
              out_blob->mut_dptr<TARGET_T>(), dx_blob->mut_dptr<INPUT_T>(), constant_output_grad,
              elem_cnt_reciprocal, local_elem_cnt, reduce_elem_cnt);
    } else {
      auto* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
      const int64_t tmp_buffer_elem_cnt = tmp_buffer->shape_view().elem_cnt() / sizeof(TARGET_T);
      const int64_t block_num = (local_elem_cnt + kBlockSize - 1) / kBlockSize;
      int launch_block = block_num;
      OF_CUDA_CHECK(GetNumBlocks(FusedBCEReduceMeanFwBwKernel<INPUT_T, ComputeType, ComputeType>,
                                 kBlockSize, 0, block_num, 32, &launch_block));
      launch_block = std::min<int32_t>(tmp_buffer_elem_cnt, launch_block);
      FusedBCEReduceMeanFwBwKernel<INPUT_T, TARGET_T, ComputeType>
          <<<launch_block, kBlockSize, 0, ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
              input_blob->dptr<INPUT_T>(), target_blob->dptr<TARGET_T>(),
              tmp_buffer->mut_dptr<TARGET_T>(), dx_blob->mut_dptr<INPUT_T>(), constant_output_grad,
              elem_cnt_reciprocal, local_elem_cnt, reduce_elem_cnt);
      ReduceLocalSumKernel<TARGET_T, ComputeType>
          <<<1, kReduceLocalSumBlockSize, 0, ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
              tmp_buffer->mut_dptr<ComputeType>(), out_blob->mut_dptr<TARGET_T>(), block_num);
    }
  }
};

template<typename INPUT_T, typename TARGET_T>
class BinaryCrossEntropyWithLogitsMeanKernel final : public user_op::OpKernel,
                                                     public CudaGraphSupport {
 public:
  BinaryCrossEntropyWithLogitsMeanKernel() = default;
  ~BinaryCrossEntropyWithLogitsMeanKernel() override = default;
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return CreateBCEWithLogitsReduceMeanKernelCache(ctx);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache* cache) const override {
    const auto* input_blob = ctx->Tensor4ArgNameAndIndex("input", 0);
    const auto* target_blob = ctx->Tensor4ArgNameAndIndex("target", 0);
    auto* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);

    int64_t local_elem_cnt = input_blob->shape_view().elem_cnt();
    int64_t reduce_elem_cnt = local_elem_cnt;

    if (cache != nullptr) {
      // Because `out`'s SBP maybe P or B, we need to use reduce_elem_cnt as reduce_mean factor.
      const auto* bce_cache = dynamic_cast<const BCEWithLogitsReduceMeanKernelCache*>(cache);
      CHECK_NOTNULL(bce_cache);
      reduce_elem_cnt = bce_cache->reduce_elem_cnt();
    }

    const INPUT_T* input = input_blob->dptr<INPUT_T>();
    const TARGET_T* target = target_blob->dptr<TARGET_T>();
    TARGET_T* out = out_blob->mut_dptr<TARGET_T>();
    using ComputeType = typename DefaultComputeType<TARGET_T>::type;

    if (local_elem_cnt <= kSingleBlockProcessNumThreshold) {
      FusedBinaryCrossEntropyWithLogitsReduceMeanKernel<INPUT_T, TARGET_T, TARGET_T, ComputeType>
          <<<1, kBlockSize, 0, ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
              input_blob->dptr<INPUT_T>(), target_blob->dptr<TARGET_T>(),
              out_blob->mut_dptr<TARGET_T>(), local_elem_cnt, reduce_elem_cnt);
    } else {
      auto* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
      const int64_t tmp_buffer_elem_cnt = tmp_buffer->shape_view().elem_cnt() / sizeof(TARGET_T);
      const int64_t block_num = (local_elem_cnt + kBlockSize - 1) / kBlockSize;
      int launch_block = block_num;
      OF_CUDA_CHECK(
          GetNumBlocks(FusedBinaryCrossEntropyWithLogitsReduceMeanKernel<INPUT_T, TARGET_T,
                                                                         ComputeType, ComputeType>,
                       kBlockSize, 0, block_num, 32, &launch_block));
      launch_block = std::min<int64_t>(tmp_buffer_elem_cnt, launch_block);
      FusedBinaryCrossEntropyWithLogitsReduceMeanKernel<INPUT_T, TARGET_T, ComputeType, ComputeType>
          <<<launch_block, kBlockSize, 0, ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
              input_blob->dptr<INPUT_T>(), target_blob->dptr<TARGET_T>(),
              tmp_buffer->mut_dptr<ComputeType>(), local_elem_cnt, reduce_elem_cnt);
      ReduceLocalSumKernel<TARGET_T, ComputeType>
          <<<1, kReduceLocalSumBlockSize, 0, ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
              tmp_buffer->mut_dptr<ComputeType>(), out_blob->mut_dptr<TARGET_T>(), block_num);
    }
  }
};

template<typename INPUT_T, typename TARGET_T>
class BinaryCrossEntropyWithLogitsReduceMeanGradKernel final : public user_op::OpKernel {
 public:
  BinaryCrossEntropyWithLogitsReduceMeanGradKernel() = default;
  ~BinaryCrossEntropyWithLogitsReduceMeanGradKernel() = default;
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return CreateBCEWithLogitsReduceMeanKernelCache(ctx);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache* cache) const override {
    const auto* input_blob = ctx->Tensor4ArgNameAndIndex("input", 0);
    const auto* target_blob = ctx->Tensor4ArgNameAndIndex("target", 0);
    const auto* dy_blob = ctx->Tensor4ArgNameAndIndex("dy", 0);
    auto* dx_blob = ctx->Tensor4ArgNameAndIndex("dx", 0);

    int64_t local_elem_cnt = input_blob->shape_view().elem_cnt();
    int64_t reduce_elem_cnt = local_elem_cnt;
    if (cache != nullptr) {
      // Because `out`'s SBP maybe P or B, we need to use reduce_elem_cnt as reduce_mean factor.
      const auto* bce_cache = dynamic_cast<const BCEWithLogitsReduceMeanKernelCache*>(cache);
      CHECK_NOTNULL(bce_cache);
      reduce_elem_cnt = bce_cache->reduce_elem_cnt();
    }

    const TARGET_T* dy = dy_blob->dptr<TARGET_T>();
    const INPUT_T* input = input_blob->dptr<INPUT_T>();
    const TARGET_T* target = target_blob->dptr<TARGET_T>();
    INPUT_T* dx = dx_blob->mut_dptr<INPUT_T>();
    using ComputeType = typename DefaultComputeType<TARGET_T>::type;

    OF_CUDA_CHECK((cuda::elementwise::BinaryWithFactory(
        BinaryCrossEntropyWithLogitsReduceMeanGradDyptrFunctor<INPUT_T, TARGET_T, ComputeType>(
            reduce_elem_cnt, dy),
        local_elem_cnt, dx, input, target, ctx->stream()->As<ep::CudaStream>()->cuda_stream())));
  }
};

}  // namespace

#define REGISTER_BINARY_CROSS_ENTROPY_REDUCE_MEAN_KERNEL(input_dtype, target_dtype)               \
  REGISTER_USER_KERNEL("binary_cross_entropy_with_logits_reduce_mean")                            \
      .SetCreateFn<BinaryCrossEntropyWithLogitsMeanKernel<input_dtype, target_dtype>>()           \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                            \
                       && (user_op::HobDataType("input", 0) == GetDataType<input_dtype>::value)   \
                       && (user_op::HobDataType("target", 0) == GetDataType<target_dtype>::value) \
                       && (user_op::HobDataType("out", 0) == GetDataType<target_dtype>::value))   \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                         \
        const int64_t elem_cnt = ctx->InputShape("input", 0).elem_cnt();                          \
        const int64_t block_num = (elem_cnt + kBlockSize - 1) / kBlockSize;                       \
        int launch_block = block_num;                                                             \
        using compute_dtype = typename DefaultComputeType<target_dtype>::type;                    \
        OF_CUDA_CHECK(GetNumBlocks(                                                               \
            FusedBinaryCrossEntropyWithLogitsReduceMeanKernel<input_dtype, target_dtype,          \
                                                              compute_dtype, compute_dtype>,      \
            kBlockSize, 0, block_num, 32, &launch_block));                                        \
        const int64_t tmp_buffer_size = GetCudaAlignedSize(launch_block * sizeof(compute_dtype)); \
        return tmp_buffer_size;                                                                   \
      });

#define REGISTER_BINARY_CROSS_ENTROPY_REDUCE_MEAN_GRAD_KERNEL(input_dtype, target_dtype)          \
  REGISTER_USER_KERNEL("binary_cross_entropy_with_logits_reduce_mean_grad")                       \
      .SetCreateFn<BinaryCrossEntropyWithLogitsReduceMeanGradKernel<input_dtype, target_dtype>>() \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                            \
                       && (user_op::HobDataType("input", 0) == GetDataType<input_dtype>::value)   \
                       && (user_op::HobDataType("target", 0) == GetDataType<target_dtype>::value) \
                       && (user_op::HobDataType("dy", 0) == GetDataType<target_dtype>::value)     \
                       && (user_op::HobDataType("dx", 0) == GetDataType<input_dtype>::value));

REGISTER_BINARY_CROSS_ENTROPY_REDUCE_MEAN_KERNEL(half, half)
REGISTER_BINARY_CROSS_ENTROPY_REDUCE_MEAN_KERNEL(half, float)
REGISTER_BINARY_CROSS_ENTROPY_REDUCE_MEAN_KERNEL(float, half)
REGISTER_BINARY_CROSS_ENTROPY_REDUCE_MEAN_KERNEL(half, double)
REGISTER_BINARY_CROSS_ENTROPY_REDUCE_MEAN_KERNEL(double, half)
REGISTER_BINARY_CROSS_ENTROPY_REDUCE_MEAN_KERNEL(float, float)
REGISTER_BINARY_CROSS_ENTROPY_REDUCE_MEAN_KERNEL(float, double)
REGISTER_BINARY_CROSS_ENTROPY_REDUCE_MEAN_KERNEL(double, float)
REGISTER_BINARY_CROSS_ENTROPY_REDUCE_MEAN_KERNEL(double, double)

REGISTER_BINARY_CROSS_ENTROPY_REDUCE_MEAN_GRAD_KERNEL(half, half)
REGISTER_BINARY_CROSS_ENTROPY_REDUCE_MEAN_GRAD_KERNEL(half, float)
REGISTER_BINARY_CROSS_ENTROPY_REDUCE_MEAN_GRAD_KERNEL(float, half)
REGISTER_BINARY_CROSS_ENTROPY_REDUCE_MEAN_GRAD_KERNEL(half, double)
REGISTER_BINARY_CROSS_ENTROPY_REDUCE_MEAN_GRAD_KERNEL(double, half)
REGISTER_BINARY_CROSS_ENTROPY_REDUCE_MEAN_GRAD_KERNEL(float, float)
REGISTER_BINARY_CROSS_ENTROPY_REDUCE_MEAN_GRAD_KERNEL(float, double)
REGISTER_BINARY_CROSS_ENTROPY_REDUCE_MEAN_GRAD_KERNEL(double, float)
REGISTER_BINARY_CROSS_ENTROPY_REDUCE_MEAN_GRAD_KERNEL(double, double)

#define REGISTER_FUSED_BCE_REDUCE_MEAN_FW_BW_KERNEL(input_dtype, target_dtype)                   \
  REGISTER_USER_KERNEL("fused_bce_reduce_mean_fw_bw")                                            \
      .SetCreateFn<FusedBCEMeanFwBwKernel<input_dtype, target_dtype>>()                          \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                           \
                       && (user_op::HobDataType("out", 0) == GetDataType<target_dtype>::value))  \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                        \
        const int64_t elem_cnt = ctx->InputShape("input", 0).elem_cnt();                         \
        const int64_t block_num = (elem_cnt + kBlockSize - 1) / kBlockSize;                      \
        int launch_block = block_num;                                                            \
        using compute_dtype = typename DefaultComputeType<target_dtype>::type;                   \
        OF_CUDA_CHECK(GetNumBlocks(                                                              \
            FusedBinaryCrossEntropyWithLogitsReduceMeanKernel<input_dtype, target_dtype,         \
                                                              compute_dtype, compute_dtype>,     \
            kBlockSize, 0, block_num, 32, &launch_block));                                       \
        const int64_t tmp_buffer_size = GetCudaAlignedSize(launch_block * sizeof(target_dtype)); \
        return tmp_buffer_size;                                                                  \
      });

REGISTER_FUSED_BCE_REDUCE_MEAN_FW_BW_KERNEL(half, half)
REGISTER_FUSED_BCE_REDUCE_MEAN_FW_BW_KERNEL(float, float)

}  // namespace user_op
}  // namespace oneflow
