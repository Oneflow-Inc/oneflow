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
#include "oneflow/core/kernel/indexed_slices_reduce_sum_kernel_util.h"
#include "oneflow/user/kernels/model_update_kernel_util.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

namespace {

class IndexedSlicesUpdateOpKernelState final : public user_op::OpKernelState {
 public:
  IndexedSlicesUpdateOpKernelState(int64_t lower, int64_t upper) : lower_(lower), upper_(upper) {}
  ~IndexedSlicesUpdateOpKernelState() override = default;

  int64_t lower() const { return lower_; }
  int64_t upper() const { return upper_; }

 private:
  const int64_t lower_;
  const int64_t upper_;
};

std::shared_ptr<user_op::OpKernelState> CreateIndexedSlicesUpdateOpKernelState(
    user_op::KernelInitContext* ctx) {
  const SbpParallel& model_sbp = ctx->SbpParallel4ArgNameAndIndex("model", 0);
  const user_op::TensorDesc* model_logical_desc =
      ctx->LogicalTensorDesc4ArgNameAndIndex("model", 0);
  const int64_t num_model_instances = model_logical_desc->shape().At(0);
  if (model_sbp.has_split_parallel() && model_sbp.split_parallel().axis() == 0
      && ctx->parallel_ctx().parallel_num() > 1) {
    CHECK(ctx->SbpParallel4ArgNameAndIndex("model_diff_indices", 0).has_broadcast_parallel());
    CHECK(ctx->SbpParallel4ArgNameAndIndex("model_diff_values", 0).has_broadcast_parallel());
    BalancedSplitter bs(num_model_instances, ctx->parallel_ctx().parallel_num());
    return std::make_shared<IndexedSlicesUpdateOpKernelState>(
        bs.At(ctx->parallel_ctx().parallel_id()).begin(),
        bs.At(ctx->parallel_ctx().parallel_id()).end());
  } else {
    return std::make_shared<IndexedSlicesUpdateOpKernelState>(0, num_model_instances);
  }
}

template<DeviceType device_type, typename T, typename K>
class IndexedSlicesSGDUpdateKernel final : public user_op::OpKernel {
 public:
  IndexedSlicesSGDUpdateKernel() = default;
  ~IndexedSlicesSGDUpdateKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return CreateIndexedSlicesUpdateOpKernelState(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    const user_op::Tensor* learning_rate = ctx->Tensor4ArgNameAndIndex("learning_rate", 0);
    const user_op::Tensor* model_diff_indices =
        ctx->Tensor4ArgNameAndIndex("model_diff_indices", 0);
    const user_op::Tensor* model_diff_values = ctx->Tensor4ArgNameAndIndex("model_diff_values", 0);
    user_op::Tensor* model = ctx->Tensor4ArgNameAndIndex("model", 0);
    auto* indexed_slices_update_state = dynamic_cast<IndexedSlicesUpdateOpKernelState*>(state);
    CHECK_NOTNULL(indexed_slices_update_state);
    CHECK_EQ(model->shape().At(0),
             indexed_slices_update_state->upper() - indexed_slices_update_state->lower());
    IndexedSlicesSGDUpdateKernelUtil<device_type, T, K>::Update(
        ctx->device_ctx(), model_diff_indices->dptr<K>(), model_diff_values->dptr<T>(),
        learning_rate->dptr<float>(), model_diff_indices->shape().elem_cnt(), model->shape().At(0),
        model->shape().Count(1), indexed_slices_update_state->lower(), model->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

#define REGISTER_INDEXED_SLICES_SGD_UPDATE_KERNEL(device_type_v, data_type_pair,                 \
                                                  indices_type_pair)                             \
  REGISTER_USER_KERNEL("indexed_slices_sgd_update")                                              \
      .SetCreateFn<IndexedSlicesSGDUpdateKernel<device_type_v, OF_PP_PAIR_FIRST(data_type_pair), \
                                                OF_PP_PAIR_FIRST(indices_type_pair)>>()          \
      .SetIsMatchedHob(                                                                          \
          (user_op::HobDeviceTag() == ToString(device_type_v))                                   \
          & (user_op::HobDataType("model", 0) == OF_PP_PAIR_SECOND(data_type_pair))              \
          & (user_op::HobDataType("model_diff_values", 0) == OF_PP_PAIR_SECOND(data_type_pair))  \
          & (user_op::HobDataType("model_diff_indices", 0)                                       \
             == OF_PP_PAIR_SECOND(indices_type_pair)));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_INDEXED_SLICES_SGD_UPDATE_KERNEL, DEVICE_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

template<DeviceType device_type, typename T, typename K>
user_op::InferTmpSizeFn GenInferTmpSizeFn() {
  return [](user_op::InferContext* ctx) {
    const user_op::TensorDesc* indices = ctx->TensorDesc4ArgNameAndIndex("model_diff_indices", 0);
    const user_op::TensorDesc* values = ctx->TensorDesc4ArgNameAndIndex("model_diff_values", 0);
    const int64_t num_indices = indices->shape().elem_cnt();
    const int64_t num_values = values->shape().elem_cnt();
    const size_t unique_diff_indices_bytes = GetCudaAlignedSize(num_indices * sizeof(K));
    const size_t unique_diff_values_bytes = GetCudaAlignedSize(num_values * sizeof(T));
    const size_t num_unique_diff_indices_bytes = GetCudaAlignedSize(1 * sizeof(int32_t));
    int64_t unique_workspace_size = 0;
    IndexedSlicesReduceSumKernelUtil<device_type, K, T, int64_t>::GetReduceSumWorkspaceSizeInBytes(
        nullptr, num_indices, values->shape().Count(indices->shape().NumAxes()),
        &unique_workspace_size);

    return unique_diff_indices_bytes + unique_diff_values_bytes + num_unique_diff_indices_bytes
           + static_cast<size_t>(unique_workspace_size);
  };
}

template<DeviceType device_type, typename T, typename K>
class IndexedSlicesMomentumUpdateKernel final : public user_op::OpKernel {
 public:
  IndexedSlicesMomentumUpdateKernel() = default;
  ~IndexedSlicesMomentumUpdateKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return CreateIndexedSlicesUpdateOpKernelState(ctx);
  }

 private:
  using ReduceSumUtilT = IndexedSlicesReduceSumKernelUtil<device_type, K, T, int32_t>;
  using MdUpdateUtilT = IndexedSlicesMomentumMdUpdateKernelUtil<device_type, T, K, int32_t>;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    const user_op::Tensor* learning_rate = ctx->Tensor4ArgNameAndIndex("learning_rate", 0);
    const user_op::Tensor* model_diff_indices =
        ctx->Tensor4ArgNameAndIndex("model_diff_indices", 0);
    const user_op::Tensor* model_diff_values = ctx->Tensor4ArgNameAndIndex("model_diff_values", 0);
    user_op::Tensor* model = ctx->Tensor4ArgNameAndIndex("model", 0);
    user_op::Tensor* momentum = ctx->Tensor4ArgNameAndIndex("momentum", 0);
    const auto beta = ctx->Attr<float>("beta");
    const int64_t num_indices = model_diff_indices->shape().elem_cnt();
    const int64_t num_values = model_diff_values->shape().elem_cnt();
    CHECK_EQ(num_values % num_indices, 0);
    const int64_t feature_size = num_values / num_indices;
    auto* indexed_slices_update_state = dynamic_cast<IndexedSlicesUpdateOpKernelState*>(state);
    CHECK_NOTNULL(indexed_slices_update_state);
    CHECK_EQ(model->shape().At(0),
             indexed_slices_update_state->upper() - indexed_slices_update_state->lower());

    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    K* unique_diff_indices_ptr = tmp_buffer->mut_dptr<K>();
    const size_t unique_diff_indices_bytes = GetCudaAlignedSize(num_indices * sizeof(K));
    T* unique_diff_values_ptr =
        reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>() + unique_diff_indices_bytes);
    const size_t unique_diff_values_bytes = GetCudaAlignedSize(num_values * sizeof(T));
    int32_t* num_unique_diff_indices_ptr = reinterpret_cast<int32_t*>(
        reinterpret_cast<char*>(unique_diff_values_ptr) + unique_diff_values_bytes);
    const size_t num_unique_diff_indices_bytes = GetCudaAlignedSize(1 * sizeof(int32_t));
    char* unique_workspace_ptr =
        reinterpret_cast<char*>(num_unique_diff_indices_ptr) + num_unique_diff_indices_bytes;
    const size_t unique_workspace_bytes = tmp_buffer->shape().elem_cnt() - unique_diff_indices_bytes
                                          - unique_diff_values_bytes
                                          - num_unique_diff_indices_bytes;

    ReduceSumUtilT::ReduceSum(ctx->device_ctx(), num_indices, feature_size,
                              model_diff_indices->dptr<K>(), model_diff_values->dptr<T>(),
                              num_unique_diff_indices_ptr, unique_diff_indices_ptr,
                              unique_diff_values_ptr, unique_workspace_ptr, unique_workspace_bytes);
    MdUpdateUtilT::Update(ctx->device_ctx(), beta, num_indices, feature_size,
                          indexed_slices_update_state->lower(),
                          indexed_slices_update_state->upper(), num_unique_diff_indices_ptr,
                          learning_rate->dptr<float>(), unique_diff_indices_ptr,
                          unique_diff_values_ptr, model->mut_dptr<T>(), momentum->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

#define REGISTER_INDEXED_SLICES_MOMENTUM_UPDATE_KERNEL(device_type_v, data_type_pair,              \
                                                       indices_type_pair)                          \
  REGISTER_USER_KERNEL("indexed_slices_momentum_update")                                           \
      .SetCreateFn<IndexedSlicesMomentumUpdateKernel<                                              \
          device_type_v, OF_PP_PAIR_FIRST(data_type_pair), OF_PP_PAIR_FIRST(indices_type_pair)>>() \
      .SetIsMatchedHob(                                                                            \
          (user_op::HobDeviceTag() == ToString(device_type_v))                                     \
          & (user_op::HobDataType("model", 0) == OF_PP_PAIR_SECOND(data_type_pair))                \
          & (user_op::HobDataType("model_diff_values", 0) == OF_PP_PAIR_SECOND(data_type_pair))    \
          & (user_op::HobDataType("model_diff_indices", 0)                                         \
             == OF_PP_PAIR_SECOND(indices_type_pair)))                                             \
      .SetInferTmpSizeFn(GenInferTmpSizeFn<device_type_v, OF_PP_PAIR_FIRST(data_type_pair),        \
                                           OF_PP_PAIR_FIRST(indices_type_pair)>());

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_INDEXED_SLICES_MOMENTUM_UPDATE_KERNEL, DEVICE_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

template<DeviceType device_type, typename T, typename K>
class IndexedSlicesAdamUpdateKernel final : public user_op::OpKernel {
 public:
  IndexedSlicesAdamUpdateKernel() = default;
  ~IndexedSlicesAdamUpdateKernel() override = default;
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return CreateIndexedSlicesUpdateOpKernelState(ctx);
  }

 private:
  using ReduceSumUtilT = IndexedSlicesReduceSumKernelUtil<device_type, K, T, int32_t>;
  using MdUpdateUtilT = IndexedSlicesAdamMdUpdateKernelUtil<device_type, T, K, int32_t>;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    const user_op::Tensor* learning_rate = ctx->Tensor4ArgNameAndIndex("learning_rate", 0);
    const user_op::Tensor* model_diff_indices =
        ctx->Tensor4ArgNameAndIndex("model_diff_indices", 0);
    const user_op::Tensor* model_diff_values = ctx->Tensor4ArgNameAndIndex("model_diff_values", 0);
    user_op::Tensor* model = ctx->Tensor4ArgNameAndIndex("model", 0);
    user_op::Tensor* m = ctx->Tensor4ArgNameAndIndex("m", 0);
    user_op::Tensor* v = ctx->Tensor4ArgNameAndIndex("v", 0);
    const auto beta1 = ctx->Attr<float>("beta1");
    const auto beta2 = ctx->Attr<float>("beta2");
    const auto epsilon = ctx->Attr<float>("epsilon");
    const auto do_bias_correction = ctx->Attr<bool>("do_bias_correction");
    T* beta1_t_ptr = nullptr;
    T* beta2_t_ptr = nullptr;
    if (do_bias_correction) {
      user_op::Tensor* beta1_t = ctx->Tensor4ArgNameAndIndex("beta1_t", 0);
      beta1_t_ptr = beta1_t->mut_dptr<T>();
      user_op::Tensor* beta2_t = ctx->Tensor4ArgNameAndIndex("beta2_t", 0);
      beta2_t_ptr = beta2_t->mut_dptr<T>();
    }
    auto* indexed_slices_update_state = dynamic_cast<IndexedSlicesUpdateOpKernelState*>(state);
    CHECK_NOTNULL(indexed_slices_update_state);
    CHECK_EQ(model->shape().At(0),
             indexed_slices_update_state->upper() - indexed_slices_update_state->lower());
    const int64_t num_indices = model_diff_indices->shape().elem_cnt();
    const int64_t num_values = model_diff_values->shape().elem_cnt();
    CHECK_EQ(num_values % num_indices, 0);
    const int64_t feature_size = num_values / num_indices;
    // tmp buffer
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    K* unique_diff_indices_ptr = tmp_buffer->mut_dptr<K>();
    const size_t unique_diff_indices_bytes = GetCudaAlignedSize(num_indices * sizeof(K));
    T* unique_diff_values_ptr =
        reinterpret_cast<T*>(tmp_buffer->mut_dptr<char>() + unique_diff_indices_bytes);
    const size_t unique_diff_values_bytes = GetCudaAlignedSize(num_values * sizeof(T));
    int32_t* num_unique_diff_indices_ptr = reinterpret_cast<int32_t*>(
        reinterpret_cast<char*>(unique_diff_values_ptr) + unique_diff_values_bytes);
    const size_t num_unique_diff_indices_bytes = GetCudaAlignedSize(1 * sizeof(int32_t));
    char* unique_workspace_ptr =
        reinterpret_cast<char*>(num_unique_diff_indices_ptr) + num_unique_diff_indices_bytes;
    const size_t unique_workspace_bytes = tmp_buffer->shape().elem_cnt() - unique_diff_indices_bytes
                                          - unique_diff_values_bytes
                                          - num_unique_diff_indices_bytes;

    ReduceSumUtilT::ReduceSum(ctx->device_ctx(), num_indices, feature_size,
                              model_diff_indices->dptr<K>(), model_diff_values->dptr<T>(),
                              num_unique_diff_indices_ptr, unique_diff_indices_ptr,
                              unique_diff_values_ptr, unique_workspace_ptr, unique_workspace_bytes);

    MdUpdateUtilT::Update(ctx->device_ctx(), beta1, beta2, epsilon, do_bias_correction, num_indices,
                          feature_size, indexed_slices_update_state->lower(),
                          indexed_slices_update_state->upper(), num_unique_diff_indices_ptr,
                          learning_rate->dptr<float>(), unique_diff_indices_ptr,
                          unique_diff_values_ptr, model->mut_dptr<T>(), m->mut_dptr<T>(),
                          v->mut_dptr<T>(), beta1_t_ptr, beta2_t_ptr);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

#define REGISTER_INDEXED_SLICES_ADAM_UPDATE_KERNEL(device_type_v, data_type_pair,                 \
                                                   indices_type_pair)                             \
  REGISTER_USER_KERNEL("indexed_slices_adam_update")                                              \
      .SetCreateFn<IndexedSlicesAdamUpdateKernel<device_type_v, OF_PP_PAIR_FIRST(data_type_pair), \
                                                 OF_PP_PAIR_FIRST(indices_type_pair)>>()          \
      .SetIsMatchedHob(                                                                           \
          (user_op::HobDeviceTag() == ToString(device_type_v))                                    \
          & (user_op::HobDataType("model", 0) == OF_PP_PAIR_SECOND(data_type_pair))               \
          & (user_op::HobDataType("model_diff_values", 0) == OF_PP_PAIR_SECOND(data_type_pair))   \
          & (user_op::HobDataType("model_diff_indices", 0)                                        \
             == OF_PP_PAIR_SECOND(indices_type_pair)))                                            \
      .SetInferTmpSizeFn(GenInferTmpSizeFn<device_type_v, OF_PP_PAIR_FIRST(data_type_pair),       \
                                           OF_PP_PAIR_FIRST(indices_type_pair)>());

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_INDEXED_SLICES_ADAM_UPDATE_KERNEL, DEVICE_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

}  // namespace

}  // namespace oneflow
