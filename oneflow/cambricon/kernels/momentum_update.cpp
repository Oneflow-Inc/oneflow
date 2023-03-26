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
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {

template<typename T, typename G>
class MluMomentumUpdateKernel final : public user_op::OpKernel {
 public:
  MluMomentumUpdateKernel() = default;
  ~MluMomentumUpdateKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    float learning_rate_val = ctx->Attr<float>("learning_rate_val");
    double scale = ctx->Attr<double>("scale");
    float l1 = ctx->Attr<float>("l1");
    float l2 = ctx->Attr<float>("l2");
    float beta = ctx->Attr<float>("beta");
    const float dampening = ctx->Attr<float>("dampening");
    const bool nesterov = ctx->Attr<bool>("nesterov");
    const bool maximize = ctx->Attr<bool>("maximize");
    float weight_decay = ctx->Attr<float>("weight_decay");
    const auto lr_scale = ctx->Attr<float>("learning_rate_scale");

    const user_op::Tensor* model_diff = ctx->Tensor4ArgNameAndIndex("model_diff", 0);
    user_op::Tensor* model = ctx->Tensor4ArgNameAndIndex("model", 0);
    user_op::Tensor* momentum = ctx->Tensor4ArgNameAndIndex("momentum", 0);
    const float* learning_rate_ptr = nullptr;
    if (ctx->has_input("learning_rate", 0)) {
      const user_op::Tensor* learning_rate = ctx->Tensor4ArgNameAndIndex("learning_rate", 0);
      learning_rate_ptr = learning_rate->dptr<float>();
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

    auto* stream = ctx->stream()->As<ep::MluStream>();
    BangHandle handle(stream->mlu_stream(), stream->device()->nclusters(),
                      stream->device()->ncores_per_cluster());

    if constexpr (std::is_same<G, float16>::value) {
      bang_momentum_update_half_kernel<T>(
          handle, model->shape_view().elem_cnt(), static_cast<T>(scale), l1, l2, beta, dampening,
          nesterov, maximize, weight_decay, learning_rate_val, lr_scale, learning_rate_ptr,
          scale_by_ptr, skip_if_ptr, model_diff->dptr<float16>(), model->mut_dptr<T>(),
          momentum->mut_dptr<T>());
    } else {
      bang_momentum_update_kernel<T>(handle, model->shape_view().elem_cnt(), static_cast<T>(scale),
                                     l1, l2, beta, dampening, nesterov, maximize, weight_decay,
                                     learning_rate_val, lr_scale, learning_rate_ptr, scale_by_ptr,
                                     skip_if_ptr, model_diff->dptr<T>(), model->mut_dptr<T>(),
                                     momentum->mut_dptr<T>());
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MLU_MOMENTUM_UPDATE_KERNEL(dtype, gtype)                                 \
  REGISTER_USER_KERNEL("momentum_update")                                                 \
      .SetCreateFn<MluMomentumUpdateKernel<dtype, gtype>>()                               \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU)                     \
                       && (user_op::HobDataType("model", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("model_diff", 0) == GetDataType<gtype>::value));

REGISTER_MLU_MOMENTUM_UPDATE_KERNEL(float, float)
REGISTER_MLU_MOMENTUM_UPDATE_KERNEL(float, float16)

#undef REGISTER_MLU_MOMENTUM_UPDATE_KERNEL

}  // namespace oneflow
