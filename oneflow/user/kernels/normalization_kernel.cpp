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

namespace oneflow {

template<typename T>
class NormalizationInferenceCpuKernel final : public user_op::OpKernel {
 public:
  NormalizationInferenceCpuKernel() = default;
  ~NormalizationInferenceCpuKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const bool training = ctx->Attr<bool>("training");
    CHECK(!training);
    const auto* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    auto* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const auto* gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);
    const auto* beta = ctx->Tensor4ArgNameAndIndex("beta", 0);
    auto* moving_mean = ctx->Tensor4ArgNameAndIndex("moving_mean", 0);
    auto* moving_variance = ctx->Tensor4ArgNameAndIndex("moving_variance", 0);
    const auto axis = ctx->Attr<int32_t>("axis");
    const auto epsilon = ctx->Attr<float>("epsilon");

    const DataType data_type = x->data_type();
    CHECK_EQ(x->shape(), y->shape());
    CHECK_EQ(y->data_type(), data_type);
    CHECK_GE(axis, 0);
    CHECK_LT(axis, x->shape().NumAxes());

    if (axis == 1) {  // NOTE(Liang Depeng): NCHW format
      const T* input_ptr = x->dptr<T>();
      const T* gamma_ptr = gamma->dptr<T>();
      const T* beta_ptr = beta->dptr<T>();

      T* output_ptr = y->mut_dptr<T>();
      T* moving_mean_ptr = moving_mean->mut_dptr<T>();
      T* moving_variance_ptr = moving_variance->mut_dptr<T>();

      const int32_t batch_size = x->shape().At(0);
      const int32_t channel_size = x->shape().At(axis);
      const int32_t spatial_size = x->shape().Count(axis + 1);

      // NOTE(Liang Depeng):
      // compute the normalization result
      const T* temp_input_ptr = input_ptr;
      T* temp_output_ptr = output_ptr;
      const int32_t all_channels = batch_size * channel_size;

      int32_t channel = -1;
      for (int ac = 0; ac < all_channels; ++ac) {
        channel += 1;
        if (channel >= channel_size) { channel = 0; }
        const T inv_variance = 1.0f / std::sqrt(moving_variance_ptr[channel] + epsilon);
        const T gamma = gamma_ptr[channel] * inv_variance;
        const T beta = beta_ptr[channel];
        const T mean = moving_mean_ptr[channel];
        for (int s = 0; s < spatial_size; ++s) {
          temp_output_ptr[s] = (temp_input_ptr[s] - mean) * gamma + beta;
        }
        temp_input_ptr += spatial_size;
        temp_output_ptr += spatial_size;
      }
    } else {  // TODO(Liang Depeng): NHWC format
    }

    // if (ctx->has_input("_add_to_output", 0)) {
    //   const user_op::Tensor* add_to_output = ctx->Tensor4ArgNameAndIndex("_add_to_output", 0);
    //   CHECK_EQ(add_to_output->data_type(), y->data_type());
    //   CHECK_EQ(add_to_output->shape(), y->shape());
    //   Memcpy<DeviceType::kGPU>(
    //       ctx->device_ctx(), y->mut_dptr<void>(), add_to_output->dptr<void>(),
    //       add_to_output->shape().elem_cnt() * GetSizeOfDataType(add_to_output->data_type()));
    //   sp_beta = CudnnSPOnePtr<T>();
    // } else {
    //   sp_beta = CudnnSPZeroPtr<T>();
    // }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BN_INFERENCE_CPU_KERNEL(dtype)                                                 \
  REGISTER_USER_KERNEL("normalization")                                                         \
      .SetCreateFn<NormalizationInferenceCpuKernel<dtype>>()                                    \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                                       \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)            \
                       & (user_op::HobAttr<bool>("training") == false))                         \
      .SetInplaceProposalFn([](const user_op::InferContext& ctx,                                \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        if (ctx.has_input("_add_to_output", 0)) {                                               \
          OF_RETURN_IF_ERROR(AddInplaceArgPairFn("y", 0, "_add_to_output", 0, true));           \
        }                                                                                       \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_BN_INFERENCE_CPU_KERNEL(float)
REGISTER_BN_INFERENCE_CPU_KERNEL(double)

#undef REGISTER_BN_INFERENCE_CPU_KERNEL

template<typename T>
class NormalizationTrainCpuKernel final : public user_op::OpKernel {
 public:
  NormalizationTrainCpuKernel() = default;
  ~NormalizationTrainCpuKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    if (ctx->op_type_name() == "normalization") { CHECK(ctx->Attr<bool>("training")); }
    const auto* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    auto* y = ctx->Tensor4ArgNameAndIndex("y", 0);

    const auto axis = ctx->Attr<int32_t>("axis");
    const auto epsilon = ctx->Attr<float>("epsilon");
    const auto momentum = ctx->Attr<float>("momentum");

    const DataType data_type = x->data_type();
    CHECK_EQ(x->shape(), y->shape());
    CHECK_EQ(y->data_type(), data_type);
    CHECK_GE(axis, 0);
    CHECK_LT(axis, x->shape().NumAxes());

    const auto* gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);
    const auto* beta = ctx->Tensor4ArgNameAndIndex("beta", 0);
    auto* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
    auto* inv_variance = ctx->Tensor4ArgNameAndIndex("inv_variance", 0);

    user_op::Tensor* moving_mean = nullptr;
    user_op::Tensor* moving_variance = nullptr;
    if (ctx->has_input("moving_mean", 0)) {
      CHECK(ctx->has_input("moving_variance", 0));
      moving_mean = ctx->Tensor4ArgNameAndIndex("moving_mean", 0);
      moving_variance = ctx->Tensor4ArgNameAndIndex("moving_variance", 0);
    }

    if (axis == 1) {  // NOTE(Liang Depeng): NCHW format
      const T* input_ptr = x->dptr<T>();
      const T* gamma_ptr = gamma->dptr<T>();
      const T* beta_ptr = beta->dptr<T>();

      T* output_ptr = y->mut_dptr<T>();
      T* mean_ptr = mean->mut_dptr<T>();
      T* inv_variance_ptr = inv_variance->mut_dptr<T>();

      T* moving_mean_ptr = nullptr;
      T* moving_variance_ptr = nullptr;
      if (moving_mean != nullptr && moving_variance != nullptr) {
        moving_mean_ptr = moving_mean->mut_dptr<T>();
        moving_variance_ptr = moving_variance->mut_dptr<T>();
      }

      const int32_t batch_size = x->shape().At(0);
      const int32_t channel_size = x->shape().At(axis);
      const int32_t spatial_size = x->shape().Count(axis + 1);
      const int32_t jump_step = spatial_size * channel_size;

      // NOTE(Liang Depeng): the following parameters were used to compute mean and var
      const int32_t reduce_count = batch_size * spatial_size;
      const int32_t unbias_reduce_count = reduce_count - 1;
      const float reduce_scale_factor = 1.0f / reduce_count;
      const float unbias_reduce_scale_factor = 1.0f / unbias_reduce_count;
      const float unbias_reduce_scale_factor_m2 = unbias_reduce_scale_factor * -2.0f;
      const float unbias_reduce_scale_factor_mn = reduce_count * unbias_reduce_scale_factor;

      const float exponential_average_factor = 1.0f - momentum;

      // NOTE(Liang Depeng):
      // compute mean & inv_variance and update moving_mean & moving_variance for each channel
      for (int channel = 0; channel < channel_size; ++channel) {
        const T* temp_input_ptr = input_ptr + channel * spatial_size;
        T sum = 0;
        T sum_square = 0;
        for (int batch = 0; batch < batch_size; ++batch) {
          for (int s = 0; s < spatial_size; ++s) {
            const T x = temp_input_ptr[s];
            sum += x;
            sum_square += x * x;
          }
          temp_input_ptr += jump_step;
        }

        const T temp_mean = sum * reduce_scale_factor;
        mean_ptr[channel] = temp_mean;

        const T temp_mean_square = temp_mean * temp_mean;
        const T temp_variance = sum_square * reduce_scale_factor - temp_mean_square;

        const T temp_unbias_variance = sum_square * unbias_reduce_scale_factor
                                       + unbias_reduce_scale_factor_m2 * temp_mean * sum
                                       + unbias_reduce_scale_factor_mn * temp_mean_square;

        inv_variance_ptr[channel] = 1.0f / std::sqrt(temp_variance + epsilon);

        if (moving_mean_ptr != nullptr && moving_variance_ptr != nullptr) {
          moving_mean_ptr[channel] =
              moving_mean_ptr[channel] * momentum + temp_mean * exponential_average_factor;
          moving_variance_ptr[channel] = moving_variance_ptr[channel] * momentum
                                         + temp_unbias_variance * exponential_average_factor;
        }
      }

      // NOTE(Liang Depeng):
      // compute the normalization result
      const T* temp_input_ptr = input_ptr;
      T* temp_output_ptr = output_ptr;
      const int32_t all_channels = batch_size * channel_size;

      int32_t channel = -1;
      for (int ac = 0; ac < all_channels; ++ac) {
        channel += 1;
        if (channel >= channel_size) { channel = 0; }
        const T gamma = gamma_ptr[channel] * inv_variance_ptr[channel];
        const T beta = beta_ptr[channel];
        const T mean = mean_ptr[channel];
        for (int s = 0; s < spatial_size; ++s) {
          temp_output_ptr[s] = (temp_input_ptr[s] - mean) * gamma + beta;
        }
        temp_input_ptr += spatial_size;
        temp_output_ptr += spatial_size;
      }
    } else {  // TODO(Liang Depeng): NHWC format
    }

    // if (ctx->op_type_name() == "normalization_add_relu") {
    //   CHECK(!ctx->has_input("_add_to_output", 0));
    //   const int64_t elem_cnt = x->shape().elem_cnt();
    //   auto* mask = ctx->Tensor4ArgNameAndIndex("reserve_space", 0);
    //   if (ctx->has_input("addend", 0)) {
    //     const auto* addend = ctx->Tensor4ArgNameAndIndex("addend", 0);
    //     AddRelu(ctx->device_ctx(), elem_cnt, y->dptr<T>(), addend->dptr<T>(), y->mut_dptr<T>(),
    //             mask->mut_dptr<int32_t>());
    //   } else {
    //     Relu(ctx->device_ctx(), elem_cnt, y->dptr<T>(), y->mut_dptr<T>(),
    //          mask->mut_dptr<int32_t>());
    //   }
    // }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BN_TRAIN_CPU_KERNEL(dtype)                                                     \
  REGISTER_USER_KERNEL("normalization")                                                         \
      .SetCreateFn<NormalizationTrainCpuKernel<dtype>>()                                        \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                                       \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)            \
                       & (user_op::HobAttr<bool>("training") == true))                          \
      .SetInplaceProposalFn([](const user_op::InferContext& ctx,                                \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        if (ctx.has_input("_add_to_output", 0)) {                                               \
          OF_RETURN_IF_ERROR(AddInplaceArgPairFn("y", 0, "_add_to_output", 0, true));           \
        }                                                                                       \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_BN_TRAIN_CPU_KERNEL(float)
REGISTER_BN_TRAIN_CPU_KERNEL(double)

#undef REGISTER_BN_TRAIN_CPU_KERNEL

template<typename T>
class NormalizationGradCpuKernel final : public user_op::OpKernel {
 public:
  NormalizationGradCpuKernel() = default;
  ~NormalizationGradCpuKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    auto* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const auto* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const auto* gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);
    auto* gamma_diff = ctx->Tensor4ArgNameAndIndex("gamma_diff", 0);
    auto* beta_diff = ctx->Tensor4ArgNameAndIndex("beta_diff", 0);
    const auto* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
    const auto* inv_variance = ctx->Tensor4ArgNameAndIndex("inv_variance", 0);
    auto* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const auto axis = ctx->Attr<int32_t>("axis");
    const auto epsilon = ctx->Attr<float>("epsilon");

    const DataType data_type = x->data_type();
    CHECK_EQ(dy->shape(), x->shape());
    CHECK_EQ(dy->data_type(), data_type);
    CHECK_EQ(dx->shape(), x->shape());
    CHECK_EQ(dx->data_type(), data_type);
    CHECK_GE(axis, 0);
    CHECK_LT(axis, x->shape().NumAxes());
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BN_GRAD_CPU_KERNEL(dtype)                \
  REGISTER_USER_KERNEL("normalization_grad")              \
      .SetCreateFn<NormalizationGradCpuKernel<dtype>>()   \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu") \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_BN_GRAD_CPU_KERNEL(float)
REGISTER_BN_GRAD_CPU_KERNEL(double)

#undef REGISTER_BN_GRAD_CPU_KERNEL

}  // namespace oneflow