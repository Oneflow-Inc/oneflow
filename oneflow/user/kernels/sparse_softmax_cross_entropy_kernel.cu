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
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"
#include "oneflow/core/framework/framework.h"
#include "oneflow/user/kernels/sparse_cross_entropy_kernel_util.h"
#include "oneflow/user/kernels/sparse_softmax_cross_entropy_kernel_util.h"
#include "oneflow/core/cuda/softmax.cuh"

namespace oneflow {
namespace user_op {

namespace {

template<typename T, typename K>
__global__ void ComputeResultGpu(const int64_t n, const int64_t w, const int64_t depth,
                                 const int64_t lower_bound, const K* labels, T* tmp, T* new_tmp,
                                 T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    assert(labels[i] >= 0);
    assert(labels[i] < depth);
    K label = labels[i] - lower_bound;
    if (label >= 0 && label < w) { y[i] = SafeLog(tmp[i]) - new_tmp[i * w + label]; }
  }
}

template<typename K>
__global__ void ComputeResultGpuHalf(const int64_t n, const int64_t w, const int64_t depth,
                                     const int64_t lower_bound, const K* labels, half* tmp,
                                     half* new_tmp, half* y) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, n) {
    assert(labels[i] >= 0);
    assert(labels[i] < depth);
    K label = labels[i] - lower_bound;
    if (label >= 0 && label < w) {
      y[i] = __float2half(SafeLog(__half2float(tmp[i])) - __half2float(new_tmp[i * w + label]));
    }
  }
#else
  printf("use half need nvcc arch >= 530");
  assert(false);
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)*/
}

template<typename T>
size_t GetReduceTempStorageSize(int64_t n, int64_t w) {
  return GetCudaAlignedSize(n * w * sizeof(T));
}
template<typename T>
size_t GetProbStorageSize(int64_t n, int64_t w) {
  return GetCudaAlignedSize(n * sizeof(T));
}
template<typename T>
void ComputeProb(DeviceCtx* ctx, const int64_t row, const int64_t col, const T* in, T* prob,
                 T* sub_result, T* sum_result) {
  using ComputeType = typename cuda::softmax::DefaultComputeType<T>::type;
  cuda::softmax::DirectLoad<T, ComputeType> load(in, col);
  cuda::softmax::DirectStore<ComputeType, T> store(prob, col);
  cuda::softmax::DirectStore<ComputeType, T> sub_result_store(sub_result, col);
  cuda::softmax::DirectStore<ComputeType, T> sum_result_store(sum_result, row);
  cuda::softmax::DispatchSoftmax<decltype(load), decltype(store), ComputeType>(
      ctx->cuda_stream(), load, store, row, col, sub_result_store, sum_result_store);
}

template<>
void ComputeProb(DeviceCtx* ctx, const int64_t row, const int64_t col, const float16* in,
                 float16* prob, float16* sub_result, float16* sum_result) {
  cuda::softmax::DirectLoad<half, float> load(reinterpret_cast<const half*>(in), col);
  cuda::softmax::DirectStore<float, half> store(reinterpret_cast<half*>(prob), col);
  cuda::softmax::DirectStore<float, half> sub_result_store(reinterpret_cast<half*>(sub_result),
                                                           col);
  cuda::softmax::DirectStore<float, half> sum_result_store(reinterpret_cast<half*>(sum_result),
                                                           row);
  cuda::softmax::DispatchSoftmax<decltype(load), decltype(store), float>(
      ctx->cuda_stream(), load, store, row, col, sub_result_store, sum_result_store);
}
}  // namespace

template<typename T, typename K>
class SparseSoftmaxCrossEntropyKernel final : public user_op::OpKernel {
 public:
  SparseSoftmaxCrossEntropyKernel() = default;
  ~SparseSoftmaxCrossEntropyKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* prediction = ctx->Tensor4ArgNameAndIndex("prediction", 0);
    const user_op::Tensor* label = ctx->Tensor4ArgNameAndIndex("label", 0);
    user_op::Tensor* prob = ctx->Tensor4ArgNameAndIndex("prob", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    const int64_t num_instances = label->shape().elem_cnt();
    CHECK_EQ(prediction->shape().elem_cnt() % num_instances, 0);
    const int64_t num_classes = prediction->shape().elem_cnt() / num_instances;
    const int64_t lower_bound = 0;
    const int64_t depth = ctx->Attr<int64_t>("depth");

    void* temp_storage = tmp_buffer->mut_dptr();
    const size_t reduce_temp_storage_bytes =
        GetReduceTempStorageSize<T>(num_instances, num_classes);
    const size_t temp_storage_bytes_offset = GetProbStorageSize<T>(num_instances, num_classes);
    T* sum_result = reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(temp_storage)
                                         + reduce_temp_storage_bytes);
    T* sub_result = reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(temp_storage)
                                         + reduce_temp_storage_bytes + temp_storage_bytes_offset);

    const K* labels = label->dptr<K>();
    T* y = out->mut_dptr<T>();
    ComputeProb<T>(ctx->device_ctx(), num_instances, num_classes, prediction->dptr<T>(),
                   prob->mut_dptr<T>(), sub_result, sum_result);

    ComputeResultGpu<T, K><<<BlocksNum4ThreadsNum(num_instances), kCudaThreadsNumPerBlock, 0,
                             ctx->device_ctx()->cuda_stream()>>>(
        num_instances, num_classes, depth, lower_bound, labels, sum_result, sub_result, y);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename K>
class SparseSoftmaxCrossEntropyKernel<float16, K> final : public user_op::OpKernel {
 public:
  SparseSoftmaxCrossEntropyKernel() = default;
  ~SparseSoftmaxCrossEntropyKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* prediction = ctx->Tensor4ArgNameAndIndex("prediction", 0);
    const user_op::Tensor* label = ctx->Tensor4ArgNameAndIndex("label", 0);
    user_op::Tensor* prob = ctx->Tensor4ArgNameAndIndex("prob", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    const int64_t num_instances = label->shape().elem_cnt();
    CHECK_EQ(prediction->shape().elem_cnt() % num_instances, 0);
    const int64_t num_classes = prediction->shape().elem_cnt() / num_instances;
    const int64_t lower_bound = 0;
    const int64_t depth = ctx->Attr<int64_t>("depth");

    void* temp_storage = tmp_buffer->mut_dptr();
    const size_t reduce_temp_storage_bytes =
        GetReduceTempStorageSize<float16>(num_instances, num_classes);
    const size_t temp_storage_bytes_offset =
        GetProbStorageSize<float16>(num_instances, num_classes);
    float16* sum_result = reinterpret_cast<float16*>(reinterpret_cast<unsigned char*>(temp_storage)
                                                     + reduce_temp_storage_bytes);
    float16* sub_result =
        reinterpret_cast<float16*>(reinterpret_cast<unsigned char*>(temp_storage)
                                   + reduce_temp_storage_bytes + temp_storage_bytes_offset);

    const K* labels = label->dptr<K>();
    float16* y = out->mut_dptr<float16>();
    ComputeProb<float16>(ctx->device_ctx(), num_instances, num_classes, prediction->dptr<float16>(),
                         prob->mut_dptr<float16>(), sub_result, sum_result);
    ComputeResultGpuHalf<K><<<BlocksNum4ThreadsNum(num_instances), kCudaThreadsNumPerBlock, 0,
                              ctx->device_ctx()->cuda_stream()>>>(
        num_instances, num_classes, depth, lower_bound, labels, reinterpret_cast<half*>(sum_result),
        reinterpret_cast<half*>(sub_result), reinterpret_cast<half*>(y));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_KERNEL(dtype_pair, ltype_pair)                     \
  REGISTER_USER_KERNEL("sparse_softmax_cross_entropy")                                           \
      .SetCreateFn<SparseSoftmaxCrossEntropyKernel<OF_PP_PAIR_FIRST(dtype_pair),                 \
                                                   OF_PP_PAIR_FIRST(ltype_pair)>>()              \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kGPU)                             \
                       & (user_op::HobDataType("label", 0) == OF_PP_PAIR_SECOND(ltype_pair))     \
                       & (user_op::HobDataType("out", 0) == OF_PP_PAIR_SECOND(dtype_pair)))      \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                        \
        const Shape& prediction_shape = ctx->InputShape("prediction", 0);                        \
        const int64_t num_classes = prediction_shape.At(prediction_shape.NumAxes() - 1);         \
        const int64_t num_instances = prediction_shape.Count(0, prediction_shape.NumAxes() - 1); \
        return SparseSoftmaxCrossEntropyKernelUtil<                                              \
            DeviceType::kGPU, OF_PP_PAIR_FIRST(dtype_pair),                                      \
            OF_PP_PAIR_FIRST(ltype_pair)>::GetComputeTempStorageSizeInBytes(num_instances,       \
                                                                            num_classes);        \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_KERNEL,
                                 FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

}  // namespace user_op
}  // namespace oneflow
