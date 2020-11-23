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

namespace oneflow {

namespace {

template<typename T, typename K, int32_t N>
struct Param {
  const T* in;
  const K* in_num_unique;
  T* out[N];
  K* num_unique[N];
};

template<typename T, typename K>
__global__ void GetPartionBoundIndex(const int64_t n, const int64_t parallel_num,
                                     const int64_t num_classes_per_rank, const T* in_ptr,
                                     const K* in_num_unique_ptr, K* out_ptr) {
  const K num = in_num_unique_ptr[0];
  CUDA_1D_KERNEL_LOOP(i, num) {
    if (i != 0) {
      const T cur_in = in_ptr[i];
      const T pre_in = in_ptr[i - 1];
#pragma unroll
      for (int32_t j = 1; j < parallel_num; ++j) {
        const int32_t lower_bound = j * num_classes_per_rank;
        if (cur_in >= lower_bound && pre_in < lower_bound) { out_ptr[j] = static_cast<K>(i); }
      }
    }
    if (i == 0) {
      const T in = in_ptr[i];
#pragma unroll
      for (int32_t j = 0; j <= parallel_num; ++j) {
        const int32_t lower_bound = j * num_classes_per_rank;
        if (in >= lower_bound) { out_ptr[j] = 0; }
      }
    }
    if (i == num - 1) {
      const T in = in_ptr[i];
#pragma unroll
      for (int32_t j = parallel_num; j >= 0; --j) {
        const int32_t lower_bound = j * num_classes_per_rank;
        if (in < lower_bound) { out_ptr[j] = num; }
      }
    }
  }
}

template<typename T, typename K, int32_t N>
__global__ void PartitionGpu(const int64_t n, const int64_t parallel_num,
                             const int64_t num_classes_per_rank, const K* partion_bound_index,
                             Param<T, K, N> param) {
  const K num = param.in_num_unique[0];
  CUDA_1D_KERNEL_LOOP(i, num) {
#pragma unroll
    for (int32_t j = 0; j < parallel_num; ++j) {
      const int32_t partion_bound_index_start = partion_bound_index[j];
      if (i >= partion_bound_index_start && i < partion_bound_index[j + 1]) {
        const int32_t lower_bound = j * num_classes_per_rank;
        param.out[j][i - partion_bound_index_start] = param.in[i] - lower_bound;
        break;
      }
    }
  }
  CUDA_1D_KERNEL_LOOP(i, parallel_num) {
    param.num_unique[i][0] = partion_bound_index[i + 1] - partion_bound_index[i];
  }
}

}  // namespace

template<typename T, typename K>
class PartitionKernel final : public user_op::OpKernel {
 public:
  PartitionKernel() = default;
  ~PartitionKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* in_num_unique = ctx->Tensor4ArgNameAndIndex("in_num_unique", 0);
    const int64_t elem_cnt = in->shape().elem_cnt();
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const int64_t parallel_num = ctx->Attr<int64_t>("parallel_num");
    const int64_t num_classes = ctx->Attr<int64_t>("num_classes");
    CHECK_EQ(num_classes % parallel_num, 0);
    const int64_t num_classes_per_rank = num_classes / parallel_num;
    // CHECK_EQ(ctx->outputs().size(), parallel_num);

    Param<T, K, 32> para;
    para.in = in->dptr<T>();
    para.in_num_unique = in_num_unique->dptr<K>();
    for (int32_t i = 0; i < parallel_num; ++i) {
      para.out[i] = ctx->Tensor4ArgNameAndIndex("out", i)->mut_dptr<T>();
      para.num_unique[i] = ctx->Tensor4ArgNameAndIndex("num_unique", i)->mut_dptr<K>();
    }
    GetPartionBoundIndex<T, K><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                                 ctx->device_ctx()->cuda_stream()>>>(
        elem_cnt, parallel_num, num_classes_per_rank, in->dptr<T>(), in_num_unique->dptr<K>(),
        tmp_buffer->mut_dptr<K>());
    PartitionGpu<T, K, 32><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                             ctx->device_ctx()->cuda_stream()>>>(
        elem_cnt, parallel_num, num_classes_per_rank, tmp_buffer->dptr<K>(), para);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_PARTITION_KERNEL(dtype, ktype)                                                \
  REGISTER_USER_KERNEL("partition")                                                            \
      .SetCreateFn<PartitionKernel<dtype, ktype>>()                                            \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                                      \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value)         \
                       & (user_op::HobDataType("num_unique", 0) == GetDataType<ktype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                      \
        const int64_t parallel_num = ctx->Attr<int64_t>("parallel_num");                       \
        return GetCudaAlignedSize((parallel_num + 1) * sizeof(ktype));                         \
      });
REGISTER_PARTITION_KERNEL(int32_t, int32_t)
REGISTER_PARTITION_KERNEL(int64_t, int32_t)
REGISTER_PARTITION_KERNEL(int32_t, int64_t)
REGISTER_PARTITION_KERNEL(int64_t, int64_t)
}  // namespace oneflow
