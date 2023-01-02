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
#include "oneflow/user/kernels/adaptive_pool_kernel_util.h"

namespace oneflow {

namespace {
template<typename T, int32_t dim>
void AdapativeMaxPoolForward(user_op::KernelComputeContext* ctx) {
  user_op::Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
  user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
  user_op::Tensor* index_tensor = ctx->Tensor4ArgNameAndIndex("index", 0);
  const Shape& x_shape = ctx->TensorDesc4ArgNameAndIndex("x", 0)->shape();
  const Shape& y_shape = ctx->TensorDesc4ArgNameAndIndex("y", 0)->shape();

  // TODO : Support 'channels_last'
  std::string data_format = "channels_first";
  const Shape& in = GetShape5D(x_shape, data_format, dim);
  const Shape& out = GetShape5D(y_shape, data_format, dim);

  const T* in_ptr = in_tensor->dptr<T>();
  T* out_ptr = out_tensor->mut_dptr<T>();
  int64_t* index_ptr = index_tensor->mut_dptr<int64_t>();

  const int64_t input_width = in.Count(4);
  const int64_t output_width = out.Count(4);
  const int64_t input_image_size = in.Count(3);
  const int64_t output_image_size = out.Count(3);
  const int64_t input_size = in.Count(2);
  const int64_t output_size = out.Count(2);

  FOR_RANGE(int64_t, n, 0, in.At(0)) {
    FOR_RANGE(int64_t, c, 0, in.At(1)) {
      FOR_RANGE(int64_t, od, 0, out.At(2)) {
        int64_t id0 = start_index(od, out.At(2), in.At(2));
        int64_t id1 = end_index(od, out.At(2), in.At(2));
        FOR_RANGE(int64_t, oh, 0, out.At(3)) {
          int64_t ih0 = start_index(oh, out.At(3), in.At(3));
          int64_t ih1 = end_index(oh, out.At(3), in.At(3));
          FOR_RANGE(int64_t, ow, 0, out.At(4)) {
            int64_t iw0 = start_index(ow, out.At(4), in.At(4));
            int64_t iw1 = end_index(ow, out.At(4), in.At(4));

            // Find out local max
            auto start_offset = id0 * input_image_size + ih0 * input_width + iw0;
            T local_max = in_ptr[start_offset];
            int64_t local_max_index = start_offset;
            FOR_RANGE(int64_t, id, id0, id1) {
              FOR_RANGE(int64_t, ih, ih0, ih1) {
                FOR_RANGE(int64_t, iw, iw0, iw1) {
                  auto cur_index = id * input_image_size + ih * input_width + iw;
                  if (in_ptr[cur_index] > local_max) {
                    local_max_index = cur_index;
                    local_max = in_ptr[cur_index];
                  }
                }
              }
            }
            auto i = od * output_image_size + oh * output_width + ow;
            out_ptr[i] = local_max;
            index_ptr[i] = local_max_index;
          }
        }
      }
      in_ptr += input_size;
      index_ptr += output_size;
      out_ptr += output_size;
    }
  }
}

template<typename T, int32_t dim>
void AdaptiveMaxPoolBackward(user_op::KernelComputeContext* ctx) {
  user_op::Tensor* grad_input = ctx->Tensor4ArgNameAndIndex("dx", 0);
  const user_op::Tensor* grad_output = ctx->Tensor4ArgNameAndIndex("dy", 0);
  const user_op::Tensor* return_indices = ctx->Tensor4ArgNameAndIndex("index", 0);
  const Shape& dx_shape = ctx->TensorDesc4ArgNameAndIndex("dx", 0)->shape();
  const Shape& dy_shape = ctx->TensorDesc4ArgNameAndIndex("dy", 0)->shape();

  // TODO : Support 'channels_last'
  std::string data_format = "channels_first";
  const Shape& in = GetShape5D(dx_shape, data_format, dim);
  const Shape& out = GetShape5D(dy_shape, data_format, dim);

  const T* dy_ptr = grad_output->dptr<T>();
  const int64_t* indices_ptr = return_indices->dptr<int64_t>();
  T* dx_ptr = grad_input->mut_dptr<T>();

  std::fill(dx_ptr, dx_ptr + grad_input->shape_view().elem_cnt(), static_cast<T>(0));

  const int64_t output_width = out.Count(4);
  const int64_t output_image_size = out.Count(3);
  const int64_t input_size = in.Count(2);
  const int64_t output_size = out.Count(2);

  FOR_RANGE(int64_t, n, 0, in.At(0)) {
    FOR_RANGE(int64_t, c, 0, in.At(1)) {
      FOR_RANGE(int64_t, od, 0, out.At(2)) {
        FOR_RANGE(int64_t, oh, 0, out.At(3)) {
          FOR_RANGE(int64_t, ow, 0, out.At(4)) {
            auto i = od * output_image_size + oh * output_width + ow;
            dx_ptr[indices_ptr[i]] += dy_ptr[i];
          }
        }
      }
      dx_ptr += input_size;
      dy_ptr += output_size;
      indices_ptr += output_size;
    }
  }
}
}  // namespace

template<typename T, int32_t dim>
class AdaptiveMaxPoolNDCpuKernel final : public user_op::OpKernel {
 public:
  AdaptiveMaxPoolNDCpuKernel() = default;
  ~AdaptiveMaxPoolNDCpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    AdapativeMaxPoolForward<T, dim>(ctx);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T, int32_t dim>
class AdaptiveMaxPoolNDGradCpuKernel final : public user_op::OpKernel {
 public:
  AdaptiveMaxPoolNDGradCpuKernel() = default;
  ~AdaptiveMaxPoolNDGradCpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    AdaptiveMaxPoolBackward<T, dim>(ctx);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_ADAPTIVE_MAX_POOLND_CPU(op_type_name, dtype, dim)                      \
  REGISTER_USER_KERNEL(op_type_name)                                                    \
      .SetCreateFn<AdaptiveMaxPoolNDCpuKernel<dtype, dim>>()                            \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                   \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)); \
                                                                                        \
  REGISTER_USER_KERNEL(op_type_name "_grad")                                            \
      .SetCreateFn<AdaptiveMaxPoolNDGradCpuKernel<dtype, dim>>()                        \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                   \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

#define REGISTER_ADAPTIVE_MAX_POOL_CPU(op_type_name, dim)      \
  REGISTER_ADAPTIVE_MAX_POOLND_CPU(op_type_name, double, dim); \
  REGISTER_ADAPTIVE_MAX_POOLND_CPU(op_type_name, float, dim);  \
  REGISTER_ADAPTIVE_MAX_POOLND_CPU(op_type_name, int, dim);

REGISTER_ADAPTIVE_MAX_POOL_CPU("adaptive_max_pool1d", 1);
REGISTER_ADAPTIVE_MAX_POOL_CPU("adaptive_max_pool2d", 2);
REGISTER_ADAPTIVE_MAX_POOL_CPU("adaptive_max_pool3d", 3);

}  // namespace oneflow
