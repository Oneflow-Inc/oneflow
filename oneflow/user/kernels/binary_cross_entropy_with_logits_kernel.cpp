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
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/ndarray/xpu_var_ndarray.h"
#include "oneflow/user/kernels/loss_kernel_util.h"
#include "oneflow/core/ep/include/primitive/broadcast_elementwise_binary.h"

namespace oneflow {
namespace user_op {
namespace {

using namespace loss;

template<typename T>
inline T ComputeMaxVal(const T x) {
  T y = -x;
  return y < 0 ? 0 : y;
}

template<typename T>
inline T CalSigmoid(const T x) {
  const T half_of_one = static_cast<T>(0.5);
  return half_of_one * std::tanh(half_of_one * x) + half_of_one;
}

template<typename INPUT_T, typename TARGET_T>
void ComputeBinaryCrossEntropyWithLogitsOut(int64_t elem_cnt, const INPUT_T* input,
                                            const TARGET_T* target, TARGET_T* out,
                                            const TARGET_T* weight,
                                            const TARGET_T* pos_weight_processed) {
  FOR_RANGE(int64_t, i, 0, elem_cnt) {
    TARGET_T input_val = static_cast<TARGET_T>(input[i]);
    TARGET_T target_val = target[i];
    TARGET_T max_val = ComputeMaxVal(input_val);
    if (out != nullptr) {
      if (pos_weight_processed == nullptr) {
        out[i] = (1 - target_val) * input_val + max_val
                 + (std::log(std::exp(-max_val) + std::exp(-input_val - max_val)));
      } else {
        TARGET_T pos_weight_processed_val = pos_weight_processed[i] - target_val + 1;
        out[i] = (1 - target_val) * input_val
                 + (pos_weight_processed_val
                    * (std::log(std::exp(-max_val) + std::exp(-input_val - max_val)) + max_val));
      }
    }
    if (weight != nullptr && out != nullptr) { out[i] *= weight[i]; }
  }
}

template<typename INPUT_T, typename TARGET_T>
void ComputeBinaryCrossEntropyWithLogitsGradOut(int64_t elem_cnt, const INPUT_T* input,
                                                const TARGET_T* target, const TARGET_T* dy,
                                                INPUT_T* dx, const TARGET_T* weight,
                                                const TARGET_T* pos_weight_processed) {
  FOR_RANGE(int64_t, i, 0, elem_cnt) {
    INPUT_T input_val = input[i];
    TARGET_T target_val = target[i];
    TARGET_T dy_val = dy[i];
    TARGET_T input_sigmoid = static_cast<TARGET_T>(CalSigmoid(input_val));
    TARGET_T dx_i_buffer = 0.0;
    if (pos_weight_processed == nullptr) {
      dx_i_buffer = (input_sigmoid - target_val) * dy_val;
    } else {
      dx_i_buffer =
          dy_val
          * ((pos_weight_processed[i] + 1 - target_val) * input_sigmoid - pos_weight_processed[i]);
    }

    if (weight != nullptr) { dx_i_buffer *= weight[i]; }
    dx[i] = static_cast<INPUT_T>(dx_i_buffer);
  }
}

template<typename INPUT_T, typename TARGET_T>
class BinaryCrossEntropyWithLogitsKernel final : public user_op::OpKernel {
 public:
  BinaryCrossEntropyWithLogitsKernel() = default;
  ~BinaryCrossEntropyWithLogitsKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* input_blob = ctx->Tensor4ArgNameAndIndex("input", 0);
    const auto* target_blob = ctx->Tensor4ArgNameAndIndex("target", 0);
    auto* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    auto* tmp_buffer_blob = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    const int64_t elem_cnt = input_blob->shape_view().elem_cnt();

    const INPUT_T* input = input_blob->dptr<INPUT_T>();
    const TARGET_T* target = target_blob->dptr<TARGET_T>();
    TARGET_T* out = out_blob->mut_dptr<TARGET_T>();

    const TARGET_T* weight = ctx->has_input("weight", 0)
                                 ? ctx->Tensor4ArgNameAndIndex("weight", 0)->dptr<TARGET_T>()
                                 : nullptr;

    TARGET_T* pos_weight_processed = nullptr;

    if (ctx->Attr<bool>("has_pos_weight")) {
      pos_weight_processed = tmp_buffer_blob->mut_dptr<TARGET_T>();
      const TARGET_T* pos_weight = ctx->Tensor4ArgNameAndIndex("pos_weight", 0)->dptr<TARGET_T>();

      Shape pos_weight_shape = Shape::Ones(target_blob->shape_view().NumAxes());
      pos_weight_shape.Set(pos_weight_shape.NumAxes() - 1,
                           ctx->Tensor4ArgNameAndIndex("pos_weight", 0)->shape_view().elem_cnt());
      auto bcast_mul =
          ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
              ctx->device_type(), ep::primitive::BinaryOp::kMul, target_blob->data_type(),
              target_blob->data_type(), target_blob->shape_view().NumAxes());
      CHECK(bcast_mul);
      bcast_mul->Launch(ctx->stream(), target_blob->shape_view().NumAxes(),
                        target_blob->shape_view().ptr(), target, pos_weight_shape.NumAxes(),
                        pos_weight_shape.dim_vec().data(), pos_weight, pos_weight_processed);
    }
    ComputeBinaryCrossEntropyWithLogitsOut(elem_cnt, input, target, out, weight,
                                           pos_weight_processed);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename INPUT_T, typename TARGET_T>
class BinaryCrossEntropyWithLogitsGradKernel final : public user_op::OpKernel {
 public:
  BinaryCrossEntropyWithLogitsGradKernel() = default;
  ~BinaryCrossEntropyWithLogitsGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* input_blob = ctx->Tensor4ArgNameAndIndex("input", 0);
    const auto* target_blob = ctx->Tensor4ArgNameAndIndex("target", 0);
    const auto* dy_blob = ctx->Tensor4ArgNameAndIndex("dy", 0);
    auto* dx_blob = ctx->Tensor4ArgNameAndIndex("dx", 0);
    auto* tmp_buffer_blob = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    const int64_t elem_cnt = input_blob->shape_view().elem_cnt();

    const TARGET_T* dy = dy_blob->dptr<TARGET_T>();
    const INPUT_T* input = input_blob->dptr<INPUT_T>();
    const TARGET_T* target = target_blob->dptr<TARGET_T>();
    INPUT_T* dx = dx_blob->mut_dptr<INPUT_T>();
    const TARGET_T* weight = ctx->has_input("weight", 0)
                                 ? ctx->Tensor4ArgNameAndIndex("weight", 0)->dptr<TARGET_T>()
                                 : nullptr;

    TARGET_T* pos_weight_processed = nullptr;

    if (ctx->Attr<bool>("has_pos_weight")) {
      pos_weight_processed = tmp_buffer_blob->mut_dptr<TARGET_T>();
      const TARGET_T* pos_weight = ctx->Tensor4ArgNameAndIndex("pos_weight", 0)->dptr<TARGET_T>();

      Shape pos_weight_shape = Shape::Ones(target_blob->shape_view().NumAxes());
      pos_weight_shape.Set(pos_weight_shape.NumAxes() - 1,
                           ctx->Tensor4ArgNameAndIndex("pos_weight", 0)->shape_view().elem_cnt());
      auto bcast_mul =
          ep::primitive::NewPrimitive<ep::primitive::BroadcastElementwiseBinaryFactory>(
              ctx->device_type(), ep::primitive::BinaryOp::kMul, target_blob->data_type(),
              target_blob->data_type(), target_blob->shape_view().NumAxes());
      CHECK(bcast_mul);
      bcast_mul->Launch(ctx->stream(), target_blob->shape_view().NumAxes(),
                        target_blob->shape_view().ptr(), target, pos_weight_shape.NumAxes(),
                        pos_weight_shape.dim_vec().data(), pos_weight, pos_weight_processed);
    }
    ComputeBinaryCrossEntropyWithLogitsGradOut(elem_cnt, input, target, dy, dx, weight,
                                               pos_weight_processed);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
user_op::InferTmpSizeFn GenFwInferTmpSizeFn() {
  return [](user_op::InferContext* ctx) {
    const int64_t n = ctx->InputShape("target", 0).elem_cnt();
    size_t tmp_buffer_size = 0;
    if (ctx->Attr<bool>("has_pos_weight")) { tmp_buffer_size += GetCudaAlignedSize(n * sizeof(T)); }
    return tmp_buffer_size;
  };
}

template<typename T>
user_op::InferTmpSizeFn GenBwInferTmpSizeFn() {
  return [](user_op::InferContext* ctx) {
    const int64_t n = ctx->InputShape("target", 0).elem_cnt();
    size_t tmp_buffer_size = 0;
    if (ctx->Attr<bool>("has_pos_weight")) { tmp_buffer_size += GetCudaAlignedSize(n * sizeof(T)); }
    return tmp_buffer_size;
  };
}

}  // namespace

#define REGISTER_BINARY_CROSS_ENTROPY_WITH_LOGITS_KERNEL(input_dtype, target_dtype)               \
  REGISTER_USER_KERNEL("binary_cross_entropy_with_logits")                                        \
      .SetCreateFn<BinaryCrossEntropyWithLogitsKernel<input_dtype, target_dtype>>()               \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                             \
                       && (user_op::HobDataType("input", 0) == GetDataType<input_dtype>::value)   \
                       && (user_op::HobDataType("target", 0) == GetDataType<target_dtype>::value) \
                       && (user_op::HobDataType("out", 0) == GetDataType<target_dtype>::value))   \
      .SetInferTmpSizeFn(GenFwInferTmpSizeFn<target_dtype>());

#define REGISTER_BINARY_CROSS_ENTROPY_WITH_LOGITS_GRAD_KERNEL(input_dtype, target_dtype)          \
  REGISTER_USER_KERNEL("binary_cross_entropy_with_logits_grad")                                   \
      .SetCreateFn<BinaryCrossEntropyWithLogitsGradKernel<input_dtype, target_dtype>>()           \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                             \
                       && (user_op::HobDataType("input", 0) == GetDataType<input_dtype>::value)   \
                       && (user_op::HobDataType("target", 0) == GetDataType<target_dtype>::value) \
                       && (user_op::HobDataType("dy", 0) == GetDataType<target_dtype>::value)     \
                       && (user_op::HobDataType("dx", 0) == GetDataType<input_dtype>::value))     \
      .SetInferTmpSizeFn(GenBwInferTmpSizeFn<target_dtype>());

REGISTER_BINARY_CROSS_ENTROPY_WITH_LOGITS_KERNEL(float, float)
REGISTER_BINARY_CROSS_ENTROPY_WITH_LOGITS_KERNEL(float, double)
REGISTER_BINARY_CROSS_ENTROPY_WITH_LOGITS_KERNEL(double, float)
REGISTER_BINARY_CROSS_ENTROPY_WITH_LOGITS_KERNEL(double, double)

REGISTER_BINARY_CROSS_ENTROPY_WITH_LOGITS_GRAD_KERNEL(float, float)
REGISTER_BINARY_CROSS_ENTROPY_WITH_LOGITS_GRAD_KERNEL(float, double)
REGISTER_BINARY_CROSS_ENTROPY_WITH_LOGITS_GRAD_KERNEL(double, float)
REGISTER_BINARY_CROSS_ENTROPY_WITH_LOGITS_GRAD_KERNEL(double, double)

}  // namespace user_op
}  // namespace oneflow
