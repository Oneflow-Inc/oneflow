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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/random_generator.h"
#include "oneflow/user/kernels/radix_sort.cuh"
#include "oneflow/user/kernels/op_kernel_state_wrapper.h"

namespace oneflow {

namespace {

class TmpBufferManager final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TmpBufferManager);
  TmpBufferManager(const int32_t& batch_size, const int32_t& capacity, void* ptr)
      : capacity_{capacity},
        random_value_elem_cnt_{batch_size},
        sorted_value_elem_cnt_{batch_size},
        indices_elem_cnt_{batch_size} {
    const int32_t random_value_aligned_bytes =
        GetCudaAlignedSize(random_value_elem_cnt_ * sizeof(float));
    const int32_t sorted_value_aligned_bytes =
        GetCudaAlignedSize(sorted_value_elem_cnt_ * sizeof(float));
    const int32_t indices_aligned_bytes = GetCudaAlignedSize(indices_elem_cnt_ * sizeof(int32_t));
    random_value_ptr_ = reinterpret_cast<float*>(ptr);
    sorted_value_ptr_ = reinterpret_cast<float*>(reinterpret_cast<char*>(random_value_ptr_)
                                                 + random_value_aligned_bytes);
    indices_ptr_ = reinterpret_cast<int32_t*>(reinterpret_cast<char*>(sorted_value_ptr_)
                                              + sorted_value_aligned_bytes);
    temp_storage_ptr_ =
        reinterpret_cast<void*>(reinterpret_cast<char*>(indices_ptr_) + indices_aligned_bytes);
    temp_storage_bytes_ =
        capacity_ - random_value_aligned_bytes - sorted_value_aligned_bytes - indices_aligned_bytes;
    CHECK_GE(temp_storage_bytes_, 0);
  }
  ~TmpBufferManager() = default;

  float* RandomValuePtr() const { return random_value_ptr_; }
  float* SortedValuePtr() const { return sorted_value_ptr_; }
  int32_t* IndicesPtr() const { return indices_ptr_; }
  void* TempStoragePtr() const { return temp_storage_ptr_; }

  int32_t RandomValueElemCnt() const { return random_value_elem_cnt_; }
  int32_t SortedValueElemCnt() const { return sorted_value_elem_cnt_; }
  int32_t IndicesElemCnt() const { return indices_elem_cnt_; }
  int32_t TempStorageBytes() const { return temp_storage_bytes_; }

 private:
  int32_t capacity_;

  float* random_value_ptr_;
  float* sorted_value_ptr_;
  int32_t* indices_ptr_;
  void* temp_storage_ptr_;

  int32_t random_value_elem_cnt_;
  int32_t sorted_value_elem_cnt_;
  int32_t indices_elem_cnt_;
  int32_t temp_storage_bytes_;
};

__global__ void InitializeIndices(int32_t elem_cnt, int32_t* indices_ptr) {
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) { indices_ptr[i] = i; };
}

}  // namespace

class GenerateRandomBatchPermutationIndicesGPUKernel final : public user_op::OpKernel {
 public:
  GenerateRandomBatchPermutationIndicesGPUKernel() = default;
  ~GenerateRandomBatchPermutationIndicesGPUKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    int64_t seed = ctx->Attr<int64_t>("seed");
    return std::make_shared<OpKernelStateWrapper<RandomGenerator<DeviceType::kGPU>>>(
        seed, ctx->device_ctx());
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* random_generator =
        dynamic_cast<OpKernelStateWrapper<RandomGenerator<DeviceType::kGPU>>*>(state);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const int32_t batch_size = y->shape().At(0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    TmpBufferManager buf_manager(batch_size, static_cast<int32_t>(tmp_buffer->shape().elem_cnt()),
                                 tmp_buffer->mut_dptr<void>());
    random_generator->Mutable()->Uniform(batch_size, buf_manager.RandomValuePtr());
    InitializeIndices<<<BlocksNum4ThreadsNum(batch_size), kCudaThreadsNumPerBlock, 0,
                        ctx->device_ctx()->cuda_stream()>>>(batch_size, buf_manager.IndicesPtr());
    const int32_t argsort_instance_num = 1;
    const int32_t argsort_instance_size = batch_size;
    SortPairsAscending(buf_manager.RandomValuePtr(), buf_manager.IndicesPtr(), argsort_instance_num,
                       argsort_instance_size, buf_manager.TempStoragePtr(),
                       buf_manager.TempStorageBytes(), buf_manager.SortedValuePtr(),
                       y->mut_dptr<int32_t>(), ctx->device_ctx()->cuda_stream());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("generate_random_batch_permutation_indices")
    .SetCreateFn<GenerateRandomBatchPermutationIndicesGPUKernel>()
    .SetIsMatchedHob(user_op::HobDeviceTag() == "gpu")
    .SetInferTmpSizeFn([](oneflow::user_op::InferContext* ctx) {
      const Shape* y_shape = ctx->Shape4ArgNameAndIndex("y", 0);
      const int32_t batch_size = y_shape->At(0);

      const int32_t random_value_aligned_bytes = GetCudaAlignedSize(batch_size * sizeof(float));
      const int32_t sorted_value_aligned_bytes = GetCudaAlignedSize(batch_size * sizeof(float));
      const int32_t indices_aligned_bytes = GetCudaAlignedSize(batch_size * sizeof(int32_t));
      const int32_t argsort_instance_num = 1;
      const int32_t argsort_instance_size = batch_size;
      const int32_t temp_storage_bytes = InferTempStorageForSortPairsAscending<float, int32_t>(
          argsort_instance_num, argsort_instance_size);

      return random_value_aligned_bytes + sorted_value_aligned_bytes + indices_aligned_bytes
             + temp_storage_bytes;
    });

}  // namespace oneflow
