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
static void ComputeMeanAndVar(const T* input_ptr, T* mean_ptr, T* inv_variance_ptr,
                              T* moving_mean_ptr, T* moving_variance_ptr, const int64_t batch_size,
                              const int64_t channel_size, const int64_t spatial_size,
                              const float epsilon, const float momentum) {
  // NOTE(Liang Depeng): the following parameters were used to compute mean and var
  const int64_t jump_step = spatial_size * channel_size;
  const int64_t reduce_count = batch_size * spatial_size;
  const int64_t unbias_reduce_count = reduce_count - 1;
  const T reduce_scale_factor = static_cast<T>(1) / reduce_count;
  const T unbias_reduce_scale_factor = static_cast<T>(1) / unbias_reduce_count;
  const T unbias_reduce_scale_factor_m2 = unbias_reduce_scale_factor * -static_cast<T>(2);
  const T unbias_reduce_scale_factor_mn = reduce_count * unbias_reduce_scale_factor;

  const T exponential_average_factor = 1.0f - momentum;

  for (int64_t channel = 0; channel < channel_size; ++channel) {
    const T* temp_input_ptr = input_ptr + channel * spatial_size;
    T sum = 0;
    T sum_square = 0;
    for (int64_t batch = 0; batch < batch_size; ++batch) {
      for (int64_t s = 0; s < spatial_size; ++s) {
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

    inv_variance_ptr[channel] = static_cast<T>(1) / std::sqrt(temp_variance + epsilon);

    if (moving_mean_ptr != nullptr && moving_variance_ptr != nullptr) {
      moving_mean_ptr[channel] =
          moving_mean_ptr[channel] * momentum + temp_mean * exponential_average_factor;
      moving_variance_ptr[channel] = moving_variance_ptr[channel] * momentum
                                     + temp_unbias_variance * exponential_average_factor;
    }
  }
}

template<typename T>
static void Normalize(const T* input_ptr, const T* mean_ptr, const T* variance_ptr,
                      const T* gamma_ptr, const T* beta_ptr, T* output_ptr,
                      const int64_t batch_size, const int64_t channel_size,
                      const int64_t spatial_size, const float epsilon, const bool training) {
  const T* temp_input_ptr = input_ptr;
  T* temp_output_ptr = output_ptr;
  const int64_t all_channels = batch_size * channel_size;
  int64_t channel = -1;
  for (int64_t ac = 0; ac < all_channels; ++ac) {
    channel += 1;
    if (channel >= channel_size) { channel = 0; }
    T inv_variance = variance_ptr[channel];
    if (!training) { inv_variance = 1.0f / std::sqrt(inv_variance + epsilon); }
    const T gamma = gamma_ptr[channel] * inv_variance;
    const T beta = beta_ptr[channel];
    const T mean = mean_ptr[channel];
    for (int64_t s = 0; s < spatial_size; ++s) {
      temp_output_ptr[s] = (temp_input_ptr[s] - mean) * gamma + beta;
    }
    temp_input_ptr += spatial_size;
    temp_output_ptr += spatial_size;
  }
}

template<typename T>
static void AddToOutput(const T* add_to_output_ptr, T* output_ptr, const int64_t elem_count) {
  for (int64_t i = 0; i < elem_count; ++i) { output_ptr[i] += add_to_output_ptr[i]; }
}

template<typename T>
static void AddRelu(const T* addend_ptr, int32_t* mask_ptr, T* output_ptr, const int64_t elem_cnt) {
  const int32_t step = 32;
  const int64_t outer_loop = elem_cnt / step;
  const int64_t remain_loop_start_idx = outer_loop * step;

  T* temp_output_ptr = output_ptr;
  for (int64_t outer = 0; outer < outer_loop; ++outer) {
    int32_t mask = 0;
    for (int32_t s = 0; s < step; ++s) {
      const T sum = temp_output_ptr[s] + addend_ptr[s];
      const bool is_positive = (sum > 0);
      mask = mask | (static_cast<int32_t>(is_positive) << s);
      temp_output_ptr[s] = is_positive ? sum : 0;
    }
    mask_ptr[outer] = mask;
    addend_ptr += step;
    temp_output_ptr += step;
  }
  if (remain_loop_start_idx < elem_cnt) {
    int32_t mask_val = 0;
    const int32_t remain = elem_cnt - remain_loop_start_idx;
    for (int32_t i = 0; i < remain; ++i) {
      const T sum = temp_output_ptr[i] + addend_ptr[i];
      const bool is_positive = (sum > 0);
      mask_val = mask_val | (static_cast<int32_t>(is_positive) << i);
      temp_output_ptr[i] = is_positive ? sum : 0;
    }
    mask_ptr[outer_loop] = mask_val;
  }
}

template<typename T>
static void Relu(int32_t* mask_ptr, T* output_ptr, const int64_t elem_cnt) {
  const int32_t step = 32;
  const int64_t outer_loop = elem_cnt / step;
  const int64_t remain_loop_start_idx = outer_loop * step;

  T* temp_output_ptr = output_ptr;
  for (int64_t outer = 0; outer < outer_loop; ++outer) {
    int32_t mask_val = 0;
    for (int32_t s = 0; s < step; ++s) {
      const T output = temp_output_ptr[s];
      const bool is_positive = (output > 0);
      mask_val = mask_val | (static_cast<int32_t>(is_positive) << s);
      temp_output_ptr[s] = is_positive ? output : 0;
    }
    mask_ptr[outer] = mask_val;
    temp_output_ptr += step;
  }
  if (remain_loop_start_idx < elem_cnt) {
    int32_t mask_val = 0;
    const int32_t remain = elem_cnt - remain_loop_start_idx;
    for (int32_t i = 0; i < remain; ++i) {
      const T output = temp_output_ptr[i];
      const bool is_positive = (output > 0);
      mask_val = mask_val | (static_cast<int32_t>(is_positive) << i);
      temp_output_ptr[i] = is_positive ? output : 0;
    }
    mask_ptr[outer_loop] = mask_val;
  }
}

template<typename T>
static void AddReluGrad(const T* dy_ptr, const int32_t* mask_ptr, T* addend_diff_ptr,
                        const int64_t elem_cnt) {
  const int32_t step = 32;
  const int64_t outer_loop = elem_cnt / step;
  const int64_t remain_loop_start_idx = outer_loop * step;

  for (int64_t outer = 0; outer < outer_loop; ++outer) {
    const int32_t mask_val = mask_ptr[outer];
    for (int32_t s = 0; s < step; ++s) {
      bool is_positive = mask_val & (1 << s);
      addend_diff_ptr[s] = static_cast<T>(is_positive) * dy_ptr[s];
    }
    addend_diff_ptr += step;
    dy_ptr += step;
  }

  if (remain_loop_start_idx < elem_cnt) {
    const int32_t mask_val = mask_ptr[outer_loop];
    const int32_t remain = elem_cnt - remain_loop_start_idx;
    for (int32_t i = 0; i < remain; ++i) {
      bool is_positive = mask_val & (1 << i);
      addend_diff_ptr[i] = static_cast<T>(is_positive) * dy_ptr[i];
    }
  }
}

template<typename T>
static void ReluGrad(const T* dy_ptr, const int32_t* mask_ptr, T* relu_dx_ptr,
                     const int64_t elem_cnt) {
  const int32_t step = 32;
  const int64_t outer_loop = elem_cnt / step;
  const int64_t remain_loop_start_idx = outer_loop * step;

  for (int64_t outer = 0; outer < outer_loop; ++outer) {
    const int32_t mask_val = mask_ptr[outer];
    for (int32_t s = 0; s < step; ++s) {
      bool is_positive = mask_val & (1 << s);
      relu_dx_ptr[s] = static_cast<T>(is_positive) * dy_ptr[s];
    }
    relu_dx_ptr += step;
    dy_ptr += step;
  }

  if (remain_loop_start_idx < elem_cnt) {
    const int32_t mask_val = mask_ptr[outer_loop];
    const int32_t remain = elem_cnt - remain_loop_start_idx;
    for (int32_t i = 0; i < remain; ++i) {
      bool is_positive = mask_val & (1 << i);
      relu_dx_ptr[i] = static_cast<T>(is_positive) * dy_ptr[i];
    }
  }
}

static size_t InferGradTmpSizeForCpuKernel(user_op::InferContext* ctx) {
  const auto& dy = ctx->InputTensorDesc("dy", 0);
  size_t tmp_size = 0;
  if (ctx->op_type_name() == "normalization_add_relu_grad" && !ctx->has_output("addend_diff", 0)) {
    tmp_size += dy.shape().elem_cnt() * GetSizeOfDataType(dy.data_type());
  }
  return tmp_size;
}

// NOTE(Liang Depeng): helper functions to process datas for specific channel over all samples.
template<typename T, typename DataProcessor>
static inline void ForEachFast(const T* data, const int64_t batch_size, const int64_t spatial_size,
                               const int64_t jump_step, const int64_t channel_idx,
                               DataProcessor data_processor) {
  const int64_t start_offset = channel_idx * spatial_size;
  const T* tmp_data = data + start_offset;
  for (int64_t outer = 0; outer < batch_size; ++outer) {
    for (int64_t i = 0; i < spatial_size; ++i) { data_processor(&tmp_data[i]); }
    tmp_data += jump_step;
  }
}

template<typename T, typename DataProcessor>
static inline void ForEachFast(const T* in_data1, const T* in_data2, const int64_t batch_size,
                               const int64_t spatial_size, const int64_t jump_step,
                               const int64_t channel_idx, DataProcessor data_processor) {
  const int64_t start_offset = channel_idx * spatial_size;
  const T* tmp_in_data1 = in_data1 + start_offset;
  const T* tmp_in_data2 = in_data2 + start_offset;
  for (int64_t outer = 0; outer < batch_size; ++outer) {
    for (int64_t i = 0; i < spatial_size; ++i) {
      data_processor(&tmp_in_data1[i], &tmp_in_data2[i]);
    }
    tmp_in_data1 += jump_step;
    tmp_in_data2 += jump_step;
  }
}

template<typename T, typename DataProcessor>
static inline void ForEachFast(const T* in_data, T* out_data, const int64_t batch_size,
                               const int64_t spatial_size, const int64_t jump_step,
                               const int64_t channel_idx, DataProcessor data_processor) {
  const int64_t start_offset = channel_idx * spatial_size;
  const T* tmp_in_data = in_data + start_offset;
  T* tmp_out_data = out_data + start_offset;
  for (int64_t outer = 0; outer < batch_size; ++outer) {
    for (int64_t i = 0; i < spatial_size; ++i) {
      data_processor(&tmp_in_data[i], &tmp_out_data[i]);
    }
    tmp_in_data += jump_step;
    tmp_out_data += jump_step;
  }
}

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
    CHECK_EQ(x->shape_view(), y->shape_view());
    CHECK_EQ(y->data_type(), data_type);
    CHECK_GE(axis, 0);
    CHECK_LT(axis, x->shape_view().NumAxes());

    if (axis == 1) {  // NOTE(Liang Depeng): NCHW format
      const T* input_ptr = x->dptr<T>();
      const T* gamma_ptr = gamma->dptr<T>();
      const T* beta_ptr = beta->dptr<T>();

      T* output_ptr = y->mut_dptr<T>();
      T* moving_mean_ptr = moving_mean->mut_dptr<T>();
      T* moving_variance_ptr = moving_variance->mut_dptr<T>();

      const int64_t batch_size = x->shape_view().At(0);
      const int64_t channel_size = x->shape_view().At(axis);
      const int64_t spatial_size = x->shape_view().Count(axis + 1);

      // NOTE(Liang Depeng):
      // compute the normalization result
      Normalize(input_ptr, moving_mean_ptr, moving_variance_ptr, gamma_ptr, beta_ptr, output_ptr,
                batch_size, channel_size, spatial_size, epsilon, false);

      if (ctx->has_input("_add_to_output", 0)) {
        const user_op::Tensor* add_to_output = ctx->Tensor4ArgNameAndIndex("_add_to_output", 0);
        CHECK_EQ(add_to_output->data_type(), y->data_type());
        CHECK_EQ(add_to_output->shape_view(), y->shape_view());
        AddToOutput(add_to_output->dptr<T>(), output_ptr, x->shape_view().elem_cnt());
      }

    } else {  // TODO(Liang Depeng): NHWC format
      UNIMPLEMENTED() << "cpu normalization op only support nchw data_format now!";
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BN_INFERENCE_CPU_KERNEL(dtype)                                           \
  REGISTER_USER_KERNEL("normalization")                                                   \
      .SetCreateFn<NormalizationInferenceCpuKernel<dtype>>()                              \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                     \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)     \
                       && (user_op::HobAttr<bool>("training") == false))                  \
      .SetInplaceProposalFn(                                                              \
          [](const user_op::InferContext& ctx,                                            \
             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {      \
            if (ctx.has_input("_add_to_output", 0)) {                                     \
              OF_RETURN_IF_ERROR(AddInplaceArgPairFn("y", 0, "_add_to_output", 0, true)); \
            }                                                                             \
            return Maybe<void>::Ok();                                                     \
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
    CHECK_EQ(x->shape_view(), y->shape_view());
    CHECK_EQ(y->data_type(), data_type);
    CHECK_GE(axis, 0);
    CHECK_LT(axis, x->shape_view().NumAxes());

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

      const int64_t batch_size = x->shape_view().At(0);
      const int64_t channel_size = x->shape_view().At(axis);
      const int64_t spatial_size = x->shape_view().Count(axis + 1);

      // NOTE(Liang Depeng):
      // Compute mean & inv_variance and update moving_mean & moving_variance for each channel.
      ComputeMeanAndVar(input_ptr, mean_ptr, inv_variance_ptr, moving_mean_ptr, moving_variance_ptr,
                        batch_size, channel_size, spatial_size, epsilon, momentum);

      // NOTE(Liang Depeng):
      // compute the normalization result
      Normalize(input_ptr, mean_ptr, inv_variance_ptr, gamma_ptr, beta_ptr, output_ptr, batch_size,
                channel_size, spatial_size, epsilon, true);

      if (ctx->has_input("_add_to_output", 0)) {
        const user_op::Tensor* add_to_output = ctx->Tensor4ArgNameAndIndex("_add_to_output", 0);
        CHECK_EQ(add_to_output->data_type(), y->data_type());
        CHECK_EQ(add_to_output->shape_view(), y->shape_view());
        AddToOutput(add_to_output->dptr<T>(), output_ptr, x->shape_view().elem_cnt());
      }

      if (ctx->op_type_name() == "normalization_add_relu") {
        CHECK(!ctx->has_input("_add_to_output", 0));
        auto* mask = ctx->Tensor4ArgNameAndIndex("reserve_space", 0);

        if (ctx->has_input("addend", 0)) {
          const auto* addend = ctx->Tensor4ArgNameAndIndex("addend", 0);
          AddRelu(addend->dptr<T>(), mask->mut_dptr<int32_t>(), output_ptr,
                  x->shape_view().elem_cnt());
        } else {
          Relu(mask->mut_dptr<int32_t>(), output_ptr, x->shape_view().elem_cnt());
        }
      }
    } else {  // TODO(Liang Depeng): NHWC format
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BN_TRAIN_CPU_KERNEL(dtype)                                               \
  REGISTER_USER_KERNEL("normalization")                                                   \
      .SetCreateFn<NormalizationTrainCpuKernel<dtype>>()                                  \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                     \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)     \
                       && (user_op::HobAttr<bool>("training") == true))                   \
      .SetInplaceProposalFn(                                                              \
          [](const user_op::InferContext& ctx,                                            \
             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {      \
            if (ctx.has_input("_add_to_output", 0)) {                                     \
              OF_RETURN_IF_ERROR(AddInplaceArgPairFn("y", 0, "_add_to_output", 0, true)); \
            }                                                                             \
            return Maybe<void>::Ok();                                                     \
          });

REGISTER_BN_TRAIN_CPU_KERNEL(float)
REGISTER_BN_TRAIN_CPU_KERNEL(double)

#undef REGISTER_BN_TRAIN_CPU_KERNEL

#define REGISTER_BN_ADD_RELU_CPU_KERNEL(dtype)                        \
  REGISTER_USER_KERNEL("normalization_add_relu")                      \
      .SetCreateFn<NormalizationTrainCpuKernel<dtype>>()              \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU) \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_BN_ADD_RELU_CPU_KERNEL(float)
REGISTER_BN_ADD_RELU_CPU_KERNEL(double)

#undef REGISTER_BN_ADD_RELU_CPU_KERNEL

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

    const DataType data_type = x->data_type();
    CHECK_EQ(dy->shape_view(), x->shape_view());
    CHECK_EQ(dy->data_type(), data_type);
    CHECK_EQ(dx->shape_view(), x->shape_view());
    CHECK_EQ(dx->data_type(), data_type);
    CHECK_GE(axis, 0);
    CHECK_LT(axis, x->shape_view().NumAxes());

    const T* dy_ptr = nullptr;
    if (ctx->op_type_name() == "normalization_grad") {
      dy_ptr = dy->dptr<T>();
    } else if (ctx->op_type_name() == "normalization_add_relu_grad") {
      const auto* mask = ctx->Tensor4ArgNameAndIndex("reserve_space", 0);
      if (ctx->has_output("addend_diff", 0)) {
        user_op::Tensor* addend_diff = ctx->Tensor4ArgNameAndIndex("addend_diff", 0);
        AddReluGrad(dy->dptr<T>(), mask->dptr<int32_t>(), addend_diff->mut_dptr<T>(),
                    dy->shape_view().elem_cnt());
        dy_ptr = addend_diff->dptr<T>();
      } else {
        ReluGrad(dy->dptr<T>(), mask->dptr<int32_t>(), tmp_buffer->mut_dptr<T>(),
                 dy->shape_view().elem_cnt());
        dy_ptr = tmp_buffer->dptr<T>();
      }

    } else {
      UNIMPLEMENTED();
    }

    if (axis == 1) {  // NOTE(Liang Depeng): NCHW format
      const T* x_ptr = x->dptr<T>();
      const T* gamma_ptr = gamma->dptr<T>();
      const T* mean_ptr = mean->dptr<T>();
      const T* inv_variance_ptr = inv_variance->dptr<T>();

      T* dx_ptr = dx->mut_dptr<T>();
      T* gamma_diff_ptr = gamma_diff->mut_dptr<T>();
      T* beta_diff_ptr = beta_diff->mut_dptr<T>();

      const int64_t batch_size = x->shape_view().At(0);
      const int64_t channel_size = x->shape_view().At(axis);
      const int64_t spatial_size = x->shape_view().Count(axis + 1);
      const int64_t jump_step = spatial_size * channel_size;
      const int64_t reduce_count = batch_size * spatial_size;

      // NOTE(Liang Depeng):
      // Borrow the MXNet implementation to compute dx, gamma_diff and beta_diff.
      // For more details pls refers to:
      // https://github.com/apache/incubator-mxnet/blob/master/src/operator/nn/batch_norm.cc
      for (int64_t channel = 0; channel < channel_size; ++channel) {
        const T gamma_c = gamma_ptr[channel];
        const T mean_c = mean_ptr[channel];
        const T inv_variance_c = inv_variance_ptr[channel];

        // NOTE(Liang Depeng): sum dy for specific channel over all samples
        T sum_dy_out = 0;
        ForEachFast(dy_ptr, batch_size, spatial_size, jump_step, channel,
                    [&sum_dy_out](const T* dy_data) { sum_dy_out += *dy_data; });

        // NOTE(Liang Depeng): dot product of the x and dy
        T dotp = 0;
        ForEachFast(x_ptr, dy_ptr, batch_size, spatial_size, jump_step, channel,
                    [&dotp, mean_c](const T* x_data, const T* dy_data) {
                      dotp += (*x_data - mean_c) * (*dy_data);
                    });

        // NOTE(Liang Depeng): projection of dy on to output scaled by std
        const T k = dotp * inv_variance_c * inv_variance_c / reduce_count;
        const T iw = inv_variance_c * gamma_c;
        const T grad_mean_c = sum_dy_out / reduce_count;
        ForEachFast(
            x_ptr, dx_ptr, batch_size, spatial_size, jump_step, channel,
            [&mean_c, &k](const T* x_data, T* dx_data) { *dx_data = (*x_data - mean_c) * k; });

        ForEachFast(dy_ptr, dx_ptr, batch_size, spatial_size, jump_step, channel,
                    [iw, grad_mean_c](const T* dy_data, T* dx_data) {
                      *dx_data = (*dy_data - grad_mean_c - *dx_data) * iw;
                    });

        gamma_diff_ptr[channel] = dotp * inv_variance_c;
        beta_diff_ptr[channel] = sum_dy_out;
      }

    } else {  // TODO(Liang Depeng): NHWC format
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BN_GRAD_CPU_KERNEL(dtype)                            \
  REGISTER_USER_KERNEL("normalization_grad")                          \
      .SetCreateFn<NormalizationGradCpuKernel<dtype>>()               \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU) \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_BN_GRAD_CPU_KERNEL(float)
REGISTER_BN_GRAD_CPU_KERNEL(double)

#undef REGISTER_BN_GRAD_CPU_KERNEL

#define REGISTER_BN_ADD_RELU_GRAD_CPU_KERNEL(dtype)                                     \
  REGISTER_USER_KERNEL("normalization_add_relu_grad")                                   \
      .SetCreateFn<NormalizationGradCpuKernel<dtype>>()                                 \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                   \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn(InferGradTmpSizeForCpuKernel);

REGISTER_BN_ADD_RELU_GRAD_CPU_KERNEL(float)
REGISTER_BN_ADD_RELU_GRAD_CPU_KERNEL(double)

#undef REGISTER_BN_ADD_RELU_GRAD_CPU_KERNEL

}  // namespace oneflow
