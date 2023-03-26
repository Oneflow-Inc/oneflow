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
#include "oneflow/cambricon/bang/bang_kernels.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

template<typename T, typename G>
class MluAdamUpdateKernel final : public user_op::OpKernel {
 public:
  MluAdamUpdateKernel() = default;
  ~MluAdamUpdateKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* model_diff = ctx->Tensor4ArgNameAndIndex("model_diff", 0);
    user_op::Tensor* model = ctx->Tensor4ArgNameAndIndex("model", 0);
    user_op::Tensor* m = ctx->Tensor4ArgNameAndIndex("m", 0);
    user_op::Tensor* v = ctx->Tensor4ArgNameAndIndex("v", 0);

    const auto scale = ctx->Attr<double>("scale");
    const auto l1 = ctx->Attr<float>("l1");
    const auto l2 = ctx->Attr<float>("l2");
    const auto beta1 = ctx->Attr<float>("beta1");
    const auto beta2 = ctx->Attr<float>("beta2");
    const auto epsilon = ctx->Attr<float>("epsilon");
    const auto weight_decay = ctx->Attr<float>("weight_decay");
    const bool amsgrad = ctx->Attr<bool>("amsgrad");
    const bool do_bias_correction = ctx->Attr<bool>("do_bias_correction");
    const float lr_scale = ctx->Attr<float>("learning_rate_scale");

    T* max_v_ptr = nullptr;
    if (amsgrad) {
      user_op::Tensor* max_v = ctx->Tensor4ArgNameAndIndex("max_v", 0);
      max_v_ptr = max_v->mut_dptr<T>();
      CHECK(max_v_ptr != nullptr);
    }

    const float learning_rate_val = ctx->Attr<float>("learning_rate_val");
    const float* learning_rate_ptr = nullptr;
    if (ctx->has_input("learning_rate", 0)) {
      const user_op::Tensor* learning_rate = ctx->Tensor4ArgNameAndIndex("learning_rate", 0);
      learning_rate_ptr = learning_rate->dptr<float>();
    }

    const float bias_correction1_val = ctx->Attr<float>("bias_correction1_val");
    const float* bias_correction1_ptr = nullptr;
    if (ctx->has_input("bias_correction1", 0)) {
      const user_op::Tensor* bias_correction1 = ctx->Tensor4ArgNameAndIndex("bias_correction1", 0);
      CHECK_EQ(bias_correction1->shape_view().elem_cnt(),
               1);  // Just for Lazy Optional Input Check.
      bias_correction1_ptr = bias_correction1->dptr<float>();
    }

    const float bias_correction2_val = ctx->Attr<float>("bias_correction2_val");
    const float* bias_correction2_ptr = nullptr;
    if (ctx->has_input("bias_correction2", 0)) {
      const user_op::Tensor* bias_correction2 = ctx->Tensor4ArgNameAndIndex("bias_correction2", 0);
      CHECK_EQ(bias_correction2->shape_view().elem_cnt(),
               1);  // Just for Lazy Optional Input Check.
      bias_correction2_ptr = bias_correction2->dptr<float>();
    }

    const T* scale_by_ptr = nullptr;
    if (ctx->has_input("scale_by_tensor", 0)) {
      const user_op::Tensor* scale_by_tensor = ctx->Tensor4ArgNameAndIndex("scale_by_tensor", 0);
      CHECK_EQ(scale_by_tensor->data_type(), model->data_type());
      CHECK_EQ(scale_by_tensor->shape_view().elem_cnt(), 1);
      scale_by_ptr = scale_by_tensor->dptr<T>();
    }

    const int64_t* skip_if_ptr = nullptr;
    if (ctx->has_input("skip_if", 0)) {
      const user_op::Tensor* skip_if = ctx->Tensor4ArgNameAndIndex("skip_if", 0);
      CHECK_EQ(skip_if->shape_view().elem_cnt(), 1);
      skip_if_ptr = skip_if->dptr<int64_t>();
    }

    float16* model_copy_ptr = nullptr;
    if (ctx->has_input("model_copy", 0)) {
      user_op::Tensor* model_copy = ctx->Tensor4ArgNameAndIndex("model_copy", 0);
      model_copy_ptr = model_copy->mut_dptr<float16>();
    }

    auto* stream = ctx->stream()->As<ep::MluStream>();
    BangHandle handle(stream->mlu_stream(), stream->device()->nclusters(),
                      stream->device()->ncores_per_cluster());

    if constexpr (std::is_same<G, float16>::value) {
      bang_adam_update_half_kernel(
          handle, model->shape_view().elem_cnt(), static_cast<T>(scale), l1, l2, beta1, beta2,
          epsilon, weight_decay, amsgrad, do_bias_correction, learning_rate_val, lr_scale,
          bias_correction1_val, bias_correction2_val, learning_rate_ptr, scale_by_ptr, skip_if_ptr,
          bias_correction1_ptr, bias_correction2_ptr, model_diff->dptr<float16>(),
          model->mut_dptr<T>(), model_copy_ptr, m->mut_dptr<T>(), v->mut_dptr<T>(), max_v_ptr);
    } else {
      bang_adam_update_kernel(
          handle, model->shape_view().elem_cnt(), static_cast<T>(scale), l1, l2, beta1, beta2,
          epsilon, weight_decay, amsgrad, do_bias_correction, learning_rate_val, lr_scale,
          bias_correction1_val, bias_correction2_val, learning_rate_ptr, scale_by_ptr, skip_if_ptr,
          bias_correction1_ptr, bias_correction2_ptr, model_diff->dptr<T>(), model->mut_dptr<T>(),
          model_copy_ptr, m->mut_dptr<T>(), v->mut_dptr<T>(), max_v_ptr);
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

#define REGISTER_MLU_ADAM_UPDATE_KERNEL(dtype, gtype)                                     \
  REGISTER_USER_KERNEL("adam_update")                                                     \
      .SetCreateFn<MluAdamUpdateKernel<dtype, gtype>>()                                   \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU)                     \
                       && (user_op::HobDataType("model", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("model_diff", 0) == GetDataType<gtype>::value));

REGISTER_MLU_ADAM_UPDATE_KERNEL(float, float);
REGISTER_MLU_ADAM_UPDATE_KERNEL(float, float16);

#undef REGISTER_MLU_ADAM_UPDATE_KERNEL

}  // namespace
}  // namespace oneflow
