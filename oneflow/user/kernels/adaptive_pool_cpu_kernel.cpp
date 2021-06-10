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

namespace oneflow {

namespace {

static inline int64_t start_index(int64_t a, int64_t b, int64_t c) {
  return (int64_t)std::floor((float)(a * c) / b);
}

static inline int64_t end_index(int64_t a, int64_t b, int64_t c) {
  return (int64_t)std::ceil((float)((a + 1) * c) / b);
}

template<DeviceType device_type, typename T>
class AdaptivePoolCpuKernel final : public user_op::OpKernel {
 public:
  AdaptivePoolCpuKernel() = default;
  ~AdaptivePoolCpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
    const std::vector<int64_t> out_size = ctx->Attr<std::vector<int64_t>>("output_size");
    const T* in_ptr = in_tensor->dptr<T>();
    T* out_ptr = out_tensor->mut_dptr<T>();

    const int64_t ndims = in_tensor->shape().NumAxes();
    CHECK_EQ(ndims, 4);

    const int64_t n_idx = 0;
    const int64_t c_idx = 1;
    const int64_t h_idx = 2;
    const int64_t w_idx = 3;

    const int64_t n_batch = in_tensor->shape().At(n_idx);
    const int64_t n_channnel = in_tensor->shape().At(c_idx);
    const int input_height = in_tensor->shape().At(h_idx);
    const int input_width = in_tensor->shape().At(w_idx);
    const int output_height = out_tensor->shape().At(h_idx);
    const int output_width = out_tensor->shape().At(w_idx);

    FOR_RANGE(int64_t, b, 0, n_batch) {
      FOR_RANGE(int64_t, c, 0, n_channnel) {
        const T* input_ptr =
            in_ptr + b * n_channnel * input_height * input_width + c * input_height * input_width;
        T* output_ptr = out_ptr + b * n_channnel * output_height * output_width
                        + c * output_height * output_width;
        FOR_RANGE(int64_t, oh, 0, output_height) {
          int64_t ih0 = start_index(oh, output_height, input_height);
          int64_t ih1 = end_index(oh, output_height, input_height);
          int64_t kh = ih1 - ih0;
          FOR_RANGE(int64_t, ow, 0, output_width) {
            int64_t iw0 = start_index(ow, output_width, input_width);
            int64_t iw1 = end_index(ow, output_width, input_width);
            int64_t kw = iw1 - iw0;
            // compute local average
            T sum = static_cast<T>(0);
            FOR_RANGE(int64_t, ih, ih0, ih1) {
              FOR_RANGE(int64_t, iw, iw0, iw1) { sum += input_ptr[ih * input_width + iw]; }
            }
            output_ptr[oh * output_width + ow] = sum / kh / kw;
          }
        }
      }
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_ADAPTIVE_POOL_KERNEL(device, dtype)       \
  REGISTER_USER_KERNEL("adaptive_avg_pool2d")              \
      .SetCreateFn<AdaptivePoolCpuKernel<device, dtype>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device) \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

#define REGISTER_ADAPTIVE_POOL_KERNEL_WITH_DEVICE(device) \
  REGISTER_ADAPTIVE_POOL_KERNEL(device, float)            \
  REGISTER_ADAPTIVE_POOL_KERNEL(device, double)           \
  REGISTER_ADAPTIVE_POOL_KERNEL(device, int8_t)           \
  REGISTER_ADAPTIVE_POOL_KERNEL(device, int32_t)          \
  REGISTER_ADAPTIVE_POOL_KERNEL(device, int64_t)

REGISTER_ADAPTIVE_POOL_KERNEL_WITH_DEVICE(DeviceType::kCPU)

template<DeviceType device_type, typename T>
class AdaptivePoolCpuGradKernel final : public user_op::OpKernel {
 public:
  AdaptivePoolCpuGradKernel() = default;
  ~AdaptivePoolCpuGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* grad_output = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* grad_input = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const std::vector<int64_t> out_size = ctx->Attr<std::vector<int64_t>>("output_size");

    const T* out_ptr = grad_output->dptr<T>();
    T* in_ptr = grad_input->mut_dptr<T>();

    const int64_t ndims = grad_output->shape().NumAxes();
    CHECK_EQ(ndims, 4);

    const int64_t n_idx = 0;
    const int64_t c_idx = 1;
    const int64_t h_idx = 2;
    const int64_t w_idx = 3;

    const int64_t n_batch = grad_output->shape().At(n_idx);
    const int64_t n_channnel = grad_output->shape().At(c_idx);
    const int input_height = grad_input->shape().At(h_idx);
    const int input_width = grad_input->shape().At(w_idx);
    const int output_height = grad_output->shape().At(h_idx);
    const int output_width = grad_output->shape().At(w_idx);

    FOR_RANGE(int64_t, b, 0, n_batch) {
      FOR_RANGE(int64_t, c, 0, n_channnel) {
        T* input_ptr =
            in_ptr + b * n_channnel * input_height * input_width + c * input_height * input_width;
        const T* output_ptr = out_ptr + b * n_channnel * output_height * output_width
                              + c * output_height * output_width;
        FOR_RANGE(int64_t, oh, 0, output_height) {
          int64_t ih0 = start_index(oh, output_height, input_height);
          int64_t ih1 = end_index(oh, output_height, input_height);
          int64_t kh = ih1 - ih0;
          FOR_RANGE(int64_t, ow, 0, output_width) {
            int64_t iw0 = start_index(ow, output_width, input_width);
            int64_t iw1 = end_index(ow, output_width, input_width);
            int64_t kw = iw1 - iw0;

            T grad_delta = output_ptr[oh * output_width + ow] / kh / kw;
            FOR_RANGE(int64_t, ih, ih0, ih1) {
              FOR_RANGE(int64_t, iw, iw0, iw1) { input_ptr[ih * input_width + iw] += grad_delta; }
            }
          }
        }
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_ELU_BACKWARD_KERNEL(device, dtype)        \
  REGISTER_USER_KERNEL("adaptive_avg_pool2d_grad")             \
      .SetCreateFn<AdaptivePoolCpuGradKernel<device, dtype>>() \
      .SetIsMatchedHob((HobDeviceTag() == device)              \
                       & (HobDataType("dx", 0) == GetDataType<dtype>::value));

#define REGISTER_ADAPTIVE_POOL_BACKWARD_KERNEL_WITH_DEVICE(device) \
  REGISTER_ADAPTIVE_POOL_KERNEL(device, float)                     \
  REGISTER_ADAPTIVE_POOL_KERNEL(device, double)                    \
  REGISTER_ADAPTIVE_POOL_KERNEL(device, int8_t)                    \
  REGISTER_ADAPTIVE_POOL_KERNEL(device, int32_t)                   \
  REGISTER_ADAPTIVE_POOL_KERNEL(device, int64_t)

REGISTER_ADAPTIVE_POOL_BACKWARD_KERNEL_WITH_DEVICE(DeviceType::kCPU)

}  // namespace

}  // namespace oneflow
