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
#include "oneflow/core/common/tensor_buffer.h"
#include "oneflow/core/thread/thread_manager.h"

namespace oneflow {

namespace {

class TensorBufferToTensorKernel final : public user_op::OpKernel {
 public:
  TensorBufferToTensorKernel() = default;
  ~TensorBufferToTensorKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const ShapeView& in_shape = in->shape();
    CHECK_EQ(in->data_type(), DataType::kTensorBuffer);
    const ShapeView& out_shape = out->shape();
    const auto& instance_shape = ctx->Attr<Shape>("instance_shape");
    CHECK_EQ(out_shape.NumAxes(), in_shape.NumAxes() + instance_shape.NumAxes());
    FOR_RANGE(int64_t, i, 0, in_shape.NumAxes()) { CHECK_EQ(out_shape.At(i), in_shape.At(i)); }
    FOR_RANGE(int64_t, i, 0, instance_shape.NumAxes()) {
      CHECK_EQ(out_shape.At(i + instance_shape.NumAxes()), instance_shape.At(i));
    }
    const auto data_type = ctx->Attr<DataType>("dtype");
    CHECK_EQ(out->data_type(), data_type);
    const int64_t instance_size = instance_shape.elem_cnt() * GetSizeOfDataType(data_type);
    const auto* in_ptr = in->dptr<TensorBuffer>();
    auto* out_ptr = out->mut_dptr<char>();
    MultiThreadLoop(in_shape.elem_cnt(), [&](size_t i) {
      const TensorBuffer* tensor_buffer = in_ptr + i;
      CHECK_EQ(tensor_buffer->nbytes(), instance_size);
      CHECK_EQ(tensor_buffer->data_type(), data_type);
      CHECK(tensor_buffer->shape() == instance_shape);
      Memcpy<DeviceType::kCPU>(ctx->device_ctx(), out_ptr + i * instance_size,
                               tensor_buffer->data(), instance_size);
    });
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("tensor_buffer_to_tensor")
    .SetCreateFn<TensorBufferToTensorKernel>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")
                     & (user_op::HobDataType("in", 0) == DataType::kTensorBuffer));

class TensorToTensorBufferKernel final : public user_op::OpKernel {
 public:
  TensorToTensorBufferKernel() = default;
  ~TensorToTensorBufferKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const ShapeView& in_shape = in->shape();
    const ShapeView& out_shape = out->shape();
    const auto instance_dims = ctx->Attr<int32_t>("instance_dims");
    CHECK_LT(instance_dims, in_shape.NumAxes());
    FOR_RANGE(int64_t, i, 0, in_shape.NumAxes() - instance_dims) {
      CHECK_EQ(out_shape.At(i), in_shape.At(i));
    }
    DimVector instance_dim_vec;
    FOR_RANGE(int64_t, i, in_shape.NumAxes() - instance_dims, in_shape.NumAxes()) {
      instance_dim_vec.push_back(in_shape.At(i));
    }
    const Shape instance_shape(instance_dim_vec);
    const auto data_type = in->data_type();
    CHECK(IsPODDataType(data_type));
    const int64_t instance_size = instance_shape.elem_cnt() * GetSizeOfDataType(data_type);
    const auto* in_ptr = in->dptr<char>();
    auto* out_ptr = out->mut_dptr<TensorBuffer>();
    MultiThreadLoop(in_shape.Count(0, in_shape.NumAxes() - instance_dims), [&](size_t i) {
      TensorBuffer* tensor_buffer = out_ptr + i;
      tensor_buffer->Resize(instance_shape, data_type);
      CHECK_EQ(tensor_buffer->nbytes(), instance_size);
      Memcpy<DeviceType::kCPU>(ctx->device_ctx(), tensor_buffer->mut_data(),
                               in_ptr + i * instance_size, instance_size);
    });
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("tensor_to_tensor_buffer")
    .SetCreateFn<TensorToTensorBufferKernel>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")
                     & (user_op::HobDataType("out", 0) == DataType::kTensorBuffer));

}  // namespace

}  // namespace oneflow
