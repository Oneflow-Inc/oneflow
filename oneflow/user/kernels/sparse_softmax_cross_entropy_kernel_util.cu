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
#include "oneflow/user/kernels/sparse_softmax_cross_entropy_kernel_util.cuh"
// #include "oneflow/user/kernels/sparse_softmax_cross_entropy_kernel_util.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"

namespace oneflow {
namespace user_op {
namespace {

template<typename T, typename K>
__global__ void ComputeDiffGpu(const int64_t num_instances, const int64_t num_classes,
                               const int64_t depth, const int64_t lower_bound, const T* prob,
                               const K* labels, const T* dy, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, num_instances) {
    const int32_t row_id = i / num_classes;
    const int32_t col_id = i - row_id * num_classes;
    assert(labels[row_id] >= 0);
    assert(labels[row_id] < depth);
    K label = labels[row_id] - lower_bound;
    if (label == col_id) {
      dx[i] = dy[row_id] * (prob[i] - 1);
    } else {
      dx[i] = dy[row_id] * prob[i];
    }
  }
}

template<typename K>
__global__ void ComputeDiffGpuHalf(const int64_t num_instances, const int64_t num_classes,
                                   const int64_t depth, const int64_t lower_bound, const half* prob,
                                   const K* labels, const half* dy, half* dx) {
  CUDA_1D_KERNEL_LOOP(i, num_instances) {
    const int32_t row_id = i / num_classes;
    const int32_t col_id = i - row_id * num_classes;
    assert(labels[row_id] >= 0);
    assert(labels[row_id] < depth);
    K label = labels[row_id] - lower_bound;
    if (label == col_id) {
      dx[i] = __hmul(dy[row_id], __hsub(prob[i], __float2half(1.0)));
    } else {
      dx[i] = __hmul(dy[row_id], prob[i]);
    }
  }
}

template<typename T, typename K>
void ComputeFront(DeviceCtx* ctx, const int64_t num_instances, const int64_t num_classes,
                  const T* in, T* prob, T* y, void* temp_storage, const size_t temp_storage_bytes,
                  const MemoryCase& prob_mem_case, const MemoryCase& tmp_buffer_mem_case) {
  auto Val = NdarrayUtil<DeviceType::kGPU, T>::GetValNdarrayBuilder();
  auto Var = NdarrayUtil<DeviceType::kGPU, T>::GetVarNdarrayBuilder();

  const size_t min_temp_storage_bytes =
      SparseSoftmaxCrossEntropyTempStorageSize<T>(num_instances, num_classes);
  assert(temp_storage_bytes >= min_temp_storage_bytes);

  const size_t reduce_operation_size =
      SparseSoftmaxCrossEntropyReduceOperationSize<T>(num_instances, num_classes);
  const size_t sum_result_size = SparseSoftmaxCrossEntropySumResultSize<T>(num_instances);

  T* reduce_storage = reinterpret_cast<T*>(temp_storage);

  auto reduce_storage_var =
      Var({static_cast<int64_t>(reduce_operation_size / sizeof(T))}, reduce_storage);
  T* sum_result =
      reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(temp_storage) + reduce_operation_size);
  T* sub_result = reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(temp_storage)
                                       + reduce_operation_size + sum_result_size);

  // max | tmp[i] = Max_j(in[i][j])
  NdarrayUtil<DeviceType::kGPU, T>::ReduceMax(ctx, Var({num_instances, 1}, sum_result),
                                              Val({num_instances, num_classes}, in),
                                              reduce_storage_var);
  // sub | prob[i][j] = in[i][j] - tmp[i]
  NdarrayUtil<DeviceType::kGPU, T>::BroadcastSub(ctx, Var({num_instances, num_classes}, sub_result),
                                                 Val({num_instances, num_classes}, in),
                                                 Val({num_instances, 1}, sum_result));
  // exp | prob[i][j] = exp(prob[i][j])
  cudaMemcpy(prob, sub_result, reduce_operation_size, cudaMemcpyHostToDevice);
  // AutoMemcpy(ctx, prob, new_tmp, reduce_temp_storage_bytes, prob_mem_case,
  // tmp_buffer_mem_case);
  NdarrayUtil<DeviceType::kGPU, T>::InplaceExp(ctx, Var({num_instances, num_classes}, prob));
  // sum | tmp[i] = Sum_j(prob[i][j])
  NdarrayUtil<DeviceType::kGPU, T>::ReduceSum(ctx, Var({num_instances, 1}, sum_result),
                                              Val({num_instances, num_classes}, prob),
                                              reduce_storage_var);

  NdarrayUtil<DeviceType::kGPU, T>::InplaceBroadcastDiv(
      ctx, Var({num_instances, num_classes}, prob),
      Val({num_instances, 1}, sum_result));  // for backward
}
}  // namespace

template<typename T, typename K>
struct SparseSoftmaxCrossEntropyKernelUtil<DeviceType::kGPU, T, K> {
  static void Compute(DeviceCtx* ctx, const int64_t num_instances, const int64_t num_classes,
                      const int64_t depth, const int64_t lower_bound, const T* in, T* prob,
                      const K* labels, T* y, void* temp_storage, const size_t temp_storage_bytes,
                      const MemoryCase& prob_mem_case, const MemoryCase& tmp_buffer_mem_case) {
    ComputeFront<T, K>(ctx, num_instances, num_classes, in, prob, y, temp_storage,
                       temp_storage_bytes, prob_mem_case, tmp_buffer_mem_case);

    const size_t reduce_operation_size =
        SparseSoftmaxCrossEntropyReduceOperationSize<T>(num_instances, num_classes);
    const size_t sum_result_size = SparseSoftmaxCrossEntropySumResultSize<T>(num_instances);

    T* sum_result = reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(temp_storage)
                                         + reduce_operation_size);
    T* sub_result = reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(temp_storage)
                                         + reduce_operation_size + sum_result_size);

    ComputeSparseSoftmaxCrossEntropyResultGpu<T, K>
        <<<BlocksNum4ThreadsNum(num_instances), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            num_instances, num_classes, depth, lower_bound, labels, sum_result, sub_result, y);
  }

  static void ComputeDiff(DeviceCtx* ctx, const int64_t num_instances, const int64_t num_classes,
                          const int64_t depth, const int64_t lower_bound, const T* prob,
                          const K* labels, const T* dy, T* dx) {
    ComputeDiffGpu<<<BlocksNum4ThreadsNum(num_instances), kCudaThreadsNumPerBlock, 0,
                     ctx->cuda_stream()>>>(num_instances, num_classes, depth, lower_bound, prob,
                                           labels, dy, dx);
  }
};

template<typename K>
struct SparseSoftmaxCrossEntropyKernelUtil<DeviceType::kGPU, float16, K> {
  static void Compute(DeviceCtx* ctx, const int64_t num_instances, const int64_t num_classes,
                      const int64_t depth, const int64_t lower_bound, const float16* in,
                      float16* prob, const K* labels, float16* y, void* temp_storage,
                      const size_t temp_storage_bytes, const MemoryCase& prob_mem_case,
                      const MemoryCase& tmp_buffer_mem_case) {
    ComputeFront<float16, K>(ctx, num_instances, num_classes, in, prob, y, temp_storage,
                             temp_storage_bytes, prob_mem_case, tmp_buffer_mem_case);

    const size_t reduce_operation_size =
        SparseSoftmaxCrossEntropyReduceOperationSize<float16>(num_instances, num_classes);
    const size_t sum_result_size = SparseSoftmaxCrossEntropySumResultSize<float16>(num_instances);
    float16* sum_result = reinterpret_cast<float16*>(reinterpret_cast<unsigned char*>(temp_storage)
                                                     + reduce_operation_size);
    float16* sub_result = reinterpret_cast<float16*>(reinterpret_cast<unsigned char*>(temp_storage)
                                                     + reduce_operation_size + sum_result_size);

    ComputeSparseSoftmaxCrossEntropyResultGpuHalf<K>
        <<<BlocksNum4ThreadsNum(num_instances), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
            num_instances, num_classes, depth, lower_bound, labels,
            reinterpret_cast<half*>(sum_result), reinterpret_cast<half*>(sub_result),
            reinterpret_cast<half*>(y));
  }

  static void ComputeDiff(DeviceCtx* ctx, const int64_t num_instances, const int64_t num_classes,
                          const int64_t depth, const int64_t lower_bound, const float16* prob,
                          const K* labels, const float16* dy, float16* dx) {
    ComputeDiffGpuHalf<<<BlocksNum4ThreadsNum(num_instances), kCudaThreadsNumPerBlock, 0,
                         ctx->cuda_stream()>>>(
        num_instances, num_classes, depth, lower_bound, reinterpret_cast<const half*>(prob), labels,
        reinterpret_cast<const half*>(dy), reinterpret_cast<half*>(dx));
  }
};

#define INSTANTIATE_SPARSE_SOFTMAX_CROSS_ENTROPY_KERNEL_UTIL_GPU(data_type_pair, index_type_pair) \
  template struct SparseSoftmaxCrossEntropyKernelUtil<                                            \
      DeviceType::kGPU, OF_PP_PAIR_FIRST(data_type_pair), OF_PP_PAIR_FIRST(index_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_SPARSE_SOFTMAX_CROSS_ENTROPY_KERNEL_UTIL_GPU,
                                 FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ);
#undef INSTANTIATE_SPARSE_SOFTMAX_CROSS_ENTROPY_KERNEL_UTIL_GPU

}  // namespace user_op
}  // namespace oneflow
