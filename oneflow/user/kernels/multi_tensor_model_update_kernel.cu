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

// Refer from Apex
// TODO:  Kernel arg size limit may be <4KB for some other cards (ie Jetson)
constexpr int depth_to_max_tensors[5] = {110, 64, 48, 36, 30};
// constexpr int depth_to_max_tensors[5] = {110, 2, 48, 36, 30};
// constexpr int depth_to_max_blocks[5] = {320, 320, 320, 320, 320};

template<typename T, int n>
struct TensorListMetadata {
  T* addresses[n][depth_to_max_tensors[n - 1]];
  int64_t sizes[depth_to_max_tensors[n - 1]];
};

template<typename T, typename G, int n>
__global__ void MultiTensorSGDUpdateGpu(T scale, const float l1, const float l2,
                                        const float weight_decay, float learning_rate_val,
                                        const float* learning_rate, const T* scale_by_ptr,
                                        const int64_t* skip_if,
                                        TensorListMetadata<T, n> meta_data) {
  if (skip_if != nullptr && *skip_if != 0) { return; }
  if (learning_rate != nullptr) { learning_rate_val = *learning_rate; }
  if (scale_by_ptr != nullptr) { scale *= *scale_by_ptr; }
  //   CUDA_1D_KERNEL_LOOP(i, n) {
  //     SGDUpdateFunctor<T, G>()(model_diff + i, model + i, scale, l1, l2, weight_decay,
  //                              learning_rate_val);
  //   }
  // assume each block process a tensor.
  const int32_t tensor_idx = blockIdx.x;
  const int64_t tensor_size = meta_data.sizes[tensor_idx];
  for (int64_t i = threadIdx.x, step = blockDim.x; i < tensor_size; i += step) {
    SGDUpdateFunctor<T, G>()(meta_data.addresses[0][tensor_idx] + i,
                             meta_data.addresses[1][tensor_idx] + i, scale, l1, l2, weight_decay,
                             learning_rate_val);
  }
}

template<DeviceType device_type, typename T, typename G>
class MultiTensorSGDUpdateKernel final : public user_op::OpKernel,
                                         public user_op::CudaGraphSupport {
 public:
  MultiTensorSGDUpdateKernel() = default;
  ~MultiTensorSGDUpdateKernel() override = default;

 private:
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
      //   CHECK_EQ(scale_by_tensor->data_type(), model->data_type());
      CHECK_EQ(scale_by_tensor->shape().elem_cnt(), 1);
      scale_by_ptr = scale_by_tensor->dptr<T>();
    }
    const int64_t* skip_if_ptr = nullptr;
    if (ctx->has_input("skip_if", 0)) {
      const user_op::Tensor* skip_if = ctx->Tensor4ArgNameAndIndex("skip_if", 0);
      CHECK_EQ(skip_if->shape().elem_cnt(), 1);
      skip_if_ptr = skip_if->dptr<int64_t>();
    }

    TensorListMetadata<T, 2> tensor_list_meta_data{};
    int32_t count = 0;
    for (int i = 0; i < n_tensor; i++) {
      tensor_list_meta_data.addresses[0][count] =
          (ctx->Tensor4ArgNameAndIndex("model_diff", i))->mut_dptr<G>();
      tensor_list_meta_data.addresses[1][count] =
          (ctx->Tensor4ArgNameAndIndex("model", i))->mut_dptr<T>();
      tensor_list_meta_data.sizes[count] =
          (ctx->Tensor4ArgNameAndIndex("model", i))->shape().elem_cnt();
      count += 1;
      if (count == depth_to_max_tensors[1] || i == n_tensor - 1) {
        // for (int j = 0; j < n_tensor; j++) {
        //   printf("elem_cnt is: %ld \n", tensor_list_meta_data.sizes[j]);
        // }
        MultiTensorSGDUpdateGpu<T, G, 2>
            <<<count, 256, 0, ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
                static_cast<T>(scale), l1, l2, weight_decay, learning_rate_val, learning_rate_ptr,
                scale_by_ptr, skip_if_ptr, tensor_list_meta_data);
        count = 0;
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

// REGISTER_MULTI_TENSOR_UPDATE_SGD_UPDATE_KERNEL(DeviceType::kCUDA, float, float16);
REGISTER_MULTI_TENSOR_UPDATE_SGD_UPDATE_KERNEL(DeviceType::kCUDA, float, float);
REGISTER_MULTI_TENSOR_UPDATE_SGD_UPDATE_KERNEL(DeviceType::kCUDA, double, double);

}  // namespace

}  // namespace oneflow
