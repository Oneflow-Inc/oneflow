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
#include "oneflow/user/kernels/model_update_kernel_util.h"
#include "oneflow/user/kernels/indexed_slices_reduce_sum_kernel_util.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/kernel/cuda_graph_support.h"

namespace oneflow {

namespace {

constexpr int kBlockSize = 256;

// Kernel arg size has 4K limit.
constexpr int max_tensors[5] = {160, 80, 40, 20, 15};

template<typename T, typename G, int n>
struct TensorTupleParams {
  G* model_diff_addresses[max_tensors[n - 1]];
  T* model_addresses[max_tensors[n - 1]];
  int64_t sizes[max_tensors[n - 1]];
  int64_t block_offset[max_tensors[n - 1]];
};

template<typename T, typename G, int n>
__global__ void MultiTensorSGDUpdateGpu(int64_t num_tensor, T scale, const float l1, const float l2,
                                        const float weight_decay, float learning_rate_val,
                                        const float* learning_rate, const T* scale_by_ptr,
                                        const int64_t* skip_if,
                                        TensorTupleParams<T, G, n> meta_data) {
  if (skip_if != nullptr && *skip_if != 0) { return; }
  if (learning_rate != nullptr) { learning_rate_val = *learning_rate; }
  if (scale_by_ptr != nullptr) { scale *= *scale_by_ptr; }
  int64_t v_block_id = blockIdx.x;
  for (int64_t tensor_idx = 0; tensor_idx < num_tensor; tensor_idx++) {
    for (int64_t i = v_block_id * blockDim.x + threadIdx.x; i < meta_data.sizes[tensor_idx];
         i += blockDim.x * gridDim.x) {
      SGDUpdateFunctor<T, G>()(meta_data.model_diff_addresses[tensor_idx] + i,
                               meta_data.model_addresses[tensor_idx] + i, scale, l1, l2,
                               weight_decay, learning_rate_val);
    }
    v_block_id -= meta_data.block_offset[tensor_idx];
    if (v_block_id < 0) { v_block_id += gridDim.x; }
  }
}

unsigned int ComputeGridSize(ep::Stream* stream, const int32_t block_size, const int64_t elem_cnt) {
  auto* cuda_stream = stream->As<ep::CudaStream>();
  const int32_t max_threads_multi_process =
      cuda_stream->device_properties().maxThreadsPerMultiProcessor;
  const int32_t multi_processor_count = cuda_stream->device_properties().multiProcessorCount;
  unsigned int blocks_per_sm = max_threads_multi_process / block_size;
  unsigned int grid_size = ((elem_cnt + block_size - 1) / block_size);
  grid_size = std::min((unsigned int)multi_processor_count * blocks_per_sm, grid_size);
  return grid_size;
}

template<DeviceType device_type, typename T, typename G>
class MultiTensorSGDUpdateKernel final : public user_op::OpKernel,
                                         public user_op::CudaGraphSupport {
 public:
  MultiTensorSGDUpdateKernel() = default;
  ~MultiTensorSGDUpdateKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const int64_t n_tensor = ctx->input_size("model");
    const double scale = ctx->Attr<double>("scale");
    const float l1 = ctx->Attr<float>("l1");
    const float l2 = ctx->Attr<float>("l2");
    const float weight_decay = ctx->Attr<float>("weight_decay");
    const float* learning_rate_ptr = nullptr;
    const float learning_rate_val = ctx->Attr<float>("learning_rate_val");

    if (ctx->has_input("learning_rate", 0)) {
      const user_op::Tensor* learning_rate = ctx->Tensor4ArgNameAndIndex("learning_rate", 0);
      learning_rate_ptr = learning_rate->dptr<float>();
    }
    const T* scale_by_ptr = nullptr;
    if (ctx->has_input("scale_by_tensor", 0)) {
      const user_op::Tensor* scale_by_tensor = ctx->Tensor4ArgNameAndIndex("scale_by_tensor", 0);
      CHECK_EQ(scale_by_tensor->data_type(), ctx->Tensor4ArgNameAndIndex("model", 0)->data_type());
      CHECK_EQ(scale_by_tensor->shape().elem_cnt(), 1);
      scale_by_ptr = scale_by_tensor->dptr<T>();
    }
    const int64_t* skip_if_ptr = nullptr;
    if (ctx->has_input("skip_if", 0)) {
      const user_op::Tensor* skip_if = ctx->Tensor4ArgNameAndIndex("skip_if", 0);
      CHECK_EQ(skip_if->shape().elem_cnt(), 1);
      skip_if_ptr = skip_if->dptr<int64_t>();
    }

    TensorTupleParams<T, G, 2> tensor_tuple_params{};
    int32_t count = 0;
    int32_t total_elem_cnt = 0;
    for (int tensor_idx = 0; tensor_idx < n_tensor; tensor_idx++) {
      tensor_tuple_params.model_diff_addresses[count] =
          (ctx->Tensor4ArgNameAndIndex("model_diff", tensor_idx))->mut_dptr<G>();
      tensor_tuple_params.model_addresses[count] =
          (ctx->Tensor4ArgNameAndIndex("model", tensor_idx))->mut_dptr<T>();
      const int64_t tensor_elem_cnt =
          ctx->Tensor4ArgNameAndIndex("model", tensor_idx)->shape().elem_cnt();
      tensor_tuple_params.sizes[count] = tensor_elem_cnt;

      count += 1;
      total_elem_cnt += tensor_elem_cnt;
      if (count == max_tensors[1] || tensor_idx == n_tensor - 1) {
        const unsigned int grid_size =
            ComputeGridSize(ctx->stream()->As<ep::CudaStream>(), kBlockSize, total_elem_cnt);
        for (int i = 0; i < count; i++) {
          tensor_tuple_params.block_offset[i] =
              ((tensor_tuple_params.sizes[i] + kBlockSize - 1) / kBlockSize) % grid_size;
        }
        MultiTensorSGDUpdateGpu<T, G, 2>
            <<<grid_size, kBlockSize, 0, ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
                count, static_cast<T>(scale), l1, l2, weight_decay, learning_rate_val,
                learning_rate_ptr, scale_by_ptr, skip_if_ptr, tensor_tuple_params);
        count = 0;
        total_elem_cnt = 0;
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

#define REGISTER_MULTI_TENSOR_UPDATE_SGD_UPDATE_KERNEL(device, dtype, gtype)              \
  REGISTER_USER_KERNEL("multi_tensor_sgd_update")                                         \
      .SetCreateFn<MultiTensorSGDUpdateKernel<device, dtype, gtype>>()                    \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                               \
                       && (user_op::HobDataType("model", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("model_diff", 0) == GetDataType<gtype>::value));

REGISTER_MULTI_TENSOR_UPDATE_SGD_UPDATE_KERNEL(DeviceType::kCUDA, float, half);
REGISTER_MULTI_TENSOR_UPDATE_SGD_UPDATE_KERNEL(DeviceType::kCUDA, float, float);
REGISTER_MULTI_TENSOR_UPDATE_SGD_UPDATE_KERNEL(DeviceType::kCUDA, double, double);

}  // namespace

}  // namespace oneflow
