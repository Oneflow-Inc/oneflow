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
#include "oneflow/core/kernel/indexed_slices_reduce_sum_kernel_util.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T, typename K>
class TmpBufferManager final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TmpBufferManager);
  TmpBufferManager(void* ptr, const int64_t num_indices, const int64_t num_values) : ptr_(ptr) {
    const size_t unique_diff_indices_bytes = GetCudaAlignedSize(num_indices * sizeof(K));
    const size_t unique_diff_values_bytes = GetCudaAlignedSize(num_values * sizeof(T));
    const size_t num_unique_diff_indices_bytes = GetCudaAlignedSize(1 * sizeof(int32_t));
    CHECK_EQ(num_values % num_indices, 0);
    IndexedSlicesReduceSumKernelUtil<device_type, K, T, int64_t>::GetReduceSumWorkspaceSizeInBytes(
        nullptr, num_indices, num_values / num_indices, &unique_workspace_bytes_);
    unique_diff_indices_offset_ = 0;
    unique_diff_values_offset_ = unique_diff_indices_offset_ + unique_diff_indices_bytes;
    num_unique_diff_indices_offset_ = unique_diff_values_offset_ + unique_diff_values_bytes;
    unique_workspace_offset_ = num_unique_diff_indices_offset_ + num_unique_diff_indices_bytes;
    CHECK_GE(unique_workspace_bytes_, 0);
    total_buffer_size_ = unique_diff_indices_bytes + unique_diff_values_bytes
                         + num_unique_diff_indices_bytes
                         + static_cast<size_t>(unique_workspace_bytes_);
  }
  ~TmpBufferManager() = default;

  int64_t UniqueWorkspaceBytes() const { return unique_workspace_bytes_; }
  size_t GetTotalBufferSize() const { return total_buffer_size_; }
  K* UniqueDiffIndicesPtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<K*>(reinterpret_cast<char*>(ptr_) + unique_diff_indices_offset_);
  }
  T* UniqueDiffValuesPtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<T*>(reinterpret_cast<char*>(ptr_) + unique_diff_values_offset_);
  }
  int32_t* NumUniqueDiffIndicesPtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<int32_t*>(reinterpret_cast<char*>(ptr_)
                                      + num_unique_diff_indices_offset_);
  }
  char* UniqueWorkspacePtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<char*>(ptr_) + unique_workspace_offset_;
  }

 private:
  size_t unique_diff_indices_offset_;
  size_t unique_diff_values_offset_;
  size_t num_unique_diff_indices_offset_;
  size_t unique_workspace_offset_;

  int64_t unique_workspace_bytes_;
  size_t total_buffer_size_;
  void* ptr_;
};

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

template<DeviceType device_type, typename T, typename G>
class SGDUpdateKernel final : public user_op::OpKernel {
 public:
  SGDUpdateKernel() = default;
  ~SGDUpdateKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* learning_rate = ctx->Tensor4ArgNameAndIndex("learning_rate", 0);
    const user_op::Tensor* model_diff = ctx->Tensor4ArgNameAndIndex("model_diff", 0);
    user_op::Tensor* model = ctx->Tensor4ArgNameAndIndex("model", 0);
    const auto scale = ctx->Attr<double>("scale");
    const auto l1 = ctx->Attr<float>("l1");
    const auto l2 = ctx->Attr<float>("l2");
    const auto weight_decay = ctx->Attr<float>("weight_decay");
    const T* scale_by_ptr = nullptr;
    if (ctx->user_op_conf().has_input("scale_by_tensor", 0)) {
      const user_op::Tensor* scale_by_tensor = ctx->Tensor4ArgNameAndIndex("scale_by_tensor", 0);
      CHECK_EQ(scale_by_tensor->data_type(), model->data_type());
      CHECK_EQ(scale_by_tensor->shape().elem_cnt(), 1);
      scale_by_ptr = scale_by_tensor->dptr<T>();
    }
    const int64_t* skip_if_ptr = nullptr;
    if (ctx->user_op_conf().has_input("skip_if", 0)) {
      const user_op::Tensor* skip_if = ctx->Tensor4ArgNameAndIndex("skip_if", 0);
      CHECK_EQ(skip_if->shape().elem_cnt(), 1);
      skip_if_ptr = skip_if->dptr<int64_t>();
    }
    SGDUpdateKernelUtil<device_type, T, G>::Update(
        ctx->device_ctx(), model->shape().elem_cnt(), static_cast<T>(scale), l1, l2, weight_decay,
        learning_rate->dptr<float>(), scale_by_ptr, skip_if_ptr, model_diff->dptr<G>(),
        model->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

#define REGISTER_SGD_UPDATE_KERNEL(device, dtype, gtype)                                 \
  REGISTER_USER_KERNEL("sgd_update")                                                     \
      .SetCreateFn<SGDUpdateKernel<device, dtype, gtype>>()                              \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                               \
                       & (user_op::HobDataType("model", 0) == GetDataType<dtype>::value) \
                       & (user_op::HobDataType("model_diff", 0) == GetDataType<gtype>::value));

REGISTER_SGD_UPDATE_KERNEL(DeviceType::kCPU, float, float);
REGISTER_SGD_UPDATE_KERNEL(DeviceType::kCPU, double, double);
#ifdef WITH_CUDA
REGISTER_SGD_UPDATE_KERNEL(DeviceType::kGPU, float, float16);
REGISTER_SGD_UPDATE_KERNEL(DeviceType::kGPU, float, float);
REGISTER_SGD_UPDATE_KERNEL(DeviceType::kGPU, double, double);
#endif  // WITH_CUDA

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
    auto* kernel_state = dynamic_cast<IndexedSlicesUpdateOpKernelState*>(state);
    CHECK_NOTNULL(kernel_state);
    CHECK_EQ(model->shape().At(0), kernel_state->upper() - kernel_state->lower());
    IndexedSlicesSGDUpdateKernelUtil<device_type, T, K>::Update(
        ctx->device_ctx(), model_diff_indices->shape().elem_cnt(), model->shape().At(0),
        model->shape().Count(1), kernel_state->lower(), learning_rate->dptr<float>(),
        model_diff_indices->dptr<K>(), model_diff_values->dptr<T>(), model->mut_dptr<T>());
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
    TmpBufferManager<device_type, T, K> buffer_manager(nullptr, num_indices, num_values);
    return buffer_manager.GetTotalBufferSize();
  };
}

template<DeviceType device_type, typename T, typename G>
class MomentumUpdateKernel final : public user_op::OpKernel {
 public:
  explicit MomentumUpdateKernel(user_op::KernelCreateContext* ctx) {
    scale_ = ctx->Attr<double>("scale");
    l1_ = ctx->Attr<float>("l1");
    l2_ = ctx->Attr<float>("l2");
    beta_ = ctx->Attr<float>("beta");
    weight_decay_ = ctx->Attr<float>("weight_decay");
    has_scale_by_ptr_ = ctx->user_op_conf().has_input("scale_by_tensor", 0);
    has_skip_if_ = ctx->user_op_conf().has_input("skip_if", 0);
  };
  ~MomentumUpdateKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* learning_rate = ctx->Tensor4ArgNameAndIndex("learning_rate", 0);
    const user_op::Tensor* model_diff = ctx->Tensor4ArgNameAndIndex("model_diff", 0);
    user_op::Tensor* model = ctx->Tensor4ArgNameAndIndex("model", 0);
    user_op::Tensor* momentum = ctx->Tensor4ArgNameAndIndex("momentum", 0);
    const T* scale_by_ptr = nullptr;
    if (has_scale_by_ptr_) {
      const user_op::Tensor* scale_by_tensor = ctx->Tensor4ArgNameAndIndex("scale_by_tensor", 0);
      CHECK_EQ(scale_by_tensor->data_type(), model->data_type());
      CHECK_EQ(scale_by_tensor->shape().elem_cnt(), 1);
      scale_by_ptr = scale_by_tensor->dptr<T>();
    }
    const int64_t* skip_if_ptr = nullptr;
    if (has_skip_if_) {
      const user_op::Tensor* skip_if = ctx->Tensor4ArgNameAndIndex("skip_if", 0);
      CHECK_EQ(skip_if->shape().elem_cnt(), 1);
      skip_if_ptr = skip_if->dptr<int64_t>();
    }
    MomentumUpdateKernelUtil<device_type, T, G>::Update(
        ctx->device_ctx(), model->shape().elem_cnt(), static_cast<T>(scale_), l1_, l2_, beta_,
        weight_decay_, learning_rate->dptr<float>(), scale_by_ptr, skip_if_ptr,
        model_diff->dptr<G>(), model->mut_dptr<T>(), momentum->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }

 private:
  double scale_;
  float l1_;
  float l2_;
  float beta_;
  float weight_decay_;
  bool has_scale_by_ptr_;
  bool has_skip_if_;
};

#define REGISTER_MOMENTUM_UPDATE_KERNEL(device, dtype, gtype)                            \
  REGISTER_USER_KERNEL("momentum_update")                                                \
      .SetCreateWithCtxFn<MomentumUpdateKernel<device, dtype, gtype>>()                  \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                               \
                       & (user_op::HobDataType("model", 0) == GetDataType<dtype>::value) \
                       & (user_op::HobDataType("model_diff", 0) == GetDataType<gtype>::value));

REGISTER_MOMENTUM_UPDATE_KERNEL(DeviceType::kCPU, float, float);
REGISTER_MOMENTUM_UPDATE_KERNEL(DeviceType::kCPU, double, double);
#ifdef WITH_CUDA
REGISTER_MOMENTUM_UPDATE_KERNEL(DeviceType::kGPU, float, float16);
REGISTER_MOMENTUM_UPDATE_KERNEL(DeviceType::kGPU, float, float);
REGISTER_MOMENTUM_UPDATE_KERNEL(DeviceType::kGPU, double, double);
#endif  // WITH_CUDA

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
    CHECK_EQ(feature_size, model_diff_values->shape().Count(model_diff_indices->shape().NumAxes()));
    auto* kernel_state = dynamic_cast<IndexedSlicesUpdateOpKernelState*>(state);
    CHECK_NOTNULL(kernel_state);
    CHECK_EQ(model->shape().At(0), kernel_state->upper() - kernel_state->lower());
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    TmpBufferManager<device_type, T, K> buffer_manager(tmp_buffer->mut_dptr(), num_indices,
                                                       num_values);
    CHECK_EQ(tmp_buffer->shape().elem_cnt(), buffer_manager.GetTotalBufferSize());
    ReduceSumUtilT::ReduceSum(
        ctx->device_ctx(), num_indices, feature_size, model_diff_indices->dptr<K>(),
        model_diff_values->dptr<T>(), buffer_manager.NumUniqueDiffIndicesPtr(),
        buffer_manager.UniqueDiffIndicesPtr(), buffer_manager.UniqueDiffValuesPtr(),
        buffer_manager.UniqueWorkspacePtr(), buffer_manager.UniqueWorkspaceBytes());
    MdUpdateUtilT::Update(ctx->device_ctx(), beta, num_indices, feature_size, kernel_state->lower(),
                          kernel_state->upper(), buffer_manager.NumUniqueDiffIndicesPtr(),
                          learning_rate->dptr<float>(), buffer_manager.UniqueDiffIndicesPtr(),
                          buffer_manager.UniqueDiffValuesPtr(), model->mut_dptr<T>(),
                          momentum->mut_dptr<T>());
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

template<DeviceType device_type, typename T, typename G>
class AdamUpdateKernel final : public user_op::OpKernel {
 public:
  AdamUpdateKernel() = default;
  ~AdamUpdateKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* learning_rate = ctx->Tensor4ArgNameAndIndex("learning_rate", 0);
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
    const T* scale_by_ptr = nullptr;
    if (ctx->user_op_conf().has_input("scale_by_tensor", 0)) {
      const user_op::Tensor* scale_by_tensor = ctx->Tensor4ArgNameAndIndex("scale_by_tensor", 0);
      CHECK_EQ(scale_by_tensor->data_type(), model->data_type());
      CHECK_EQ(scale_by_tensor->shape().elem_cnt(), 1);
      scale_by_ptr = scale_by_tensor->dptr<T>();
    }
    const int64_t* skip_if_ptr = nullptr;
    if (ctx->user_op_conf().has_input("skip_if", 0)) {
      const user_op::Tensor* skip_if = ctx->Tensor4ArgNameAndIndex("skip_if", 0);
      CHECK_EQ(skip_if->shape().elem_cnt(), 1);
      skip_if_ptr = skip_if->dptr<int64_t>();
    }
    AdamUpdateKernelUtil<device_type, T, G>::Update(
        ctx->device_ctx(), model->shape().elem_cnt(), static_cast<T>(scale), l1, l2, beta1, beta2,
        epsilon, weight_decay, learning_rate->dptr<float>(), scale_by_ptr, skip_if_ptr,
        model_diff->dptr<G>(), model->mut_dptr<T>(), m->mut_dptr<T>(), v->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

#define REGISTER_ADAM_UPDATE_KERNEL(device, dtype, gtype)                                \
  REGISTER_USER_KERNEL("adam_update")                                                    \
      .SetCreateFn<AdamUpdateKernel<device, dtype, gtype>>()                             \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                               \
                       & (user_op::HobDataType("model", 0) == GetDataType<dtype>::value) \
                       & (user_op::HobDataType("model_diff", 0) == GetDataType<gtype>::value));

REGISTER_ADAM_UPDATE_KERNEL(DeviceType::kCPU, float, float);
REGISTER_ADAM_UPDATE_KERNEL(DeviceType::kCPU, double, double);
#ifdef WITH_CUDA
REGISTER_ADAM_UPDATE_KERNEL(DeviceType::kGPU, float, float16);
REGISTER_ADAM_UPDATE_KERNEL(DeviceType::kGPU, float, float);
REGISTER_ADAM_UPDATE_KERNEL(DeviceType::kGPU, double, double);
#endif  // WITH_CUDA

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
    auto* kernel_state = dynamic_cast<IndexedSlicesUpdateOpKernelState*>(state);
    CHECK_NOTNULL(kernel_state);
    CHECK_EQ(model->shape().At(0), kernel_state->upper() - kernel_state->lower());
    const int64_t num_indices = model_diff_indices->shape().elem_cnt();
    const int64_t num_values = model_diff_values->shape().elem_cnt();
    CHECK_EQ(num_values % num_indices, 0);
    const int64_t feature_size = num_values / num_indices;
    CHECK_EQ(feature_size, model_diff_values->shape().Count(model_diff_indices->shape().NumAxes()));
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    TmpBufferManager<device_type, T, K> buffer_manager(tmp_buffer->mut_dptr(), num_indices,
                                                       num_values);
    CHECK_EQ(tmp_buffer->shape().elem_cnt(), buffer_manager.GetTotalBufferSize());

    ReduceSumUtilT::ReduceSum(
        ctx->device_ctx(), num_indices, feature_size, model_diff_indices->dptr<K>(),
        model_diff_values->dptr<T>(), buffer_manager.NumUniqueDiffIndicesPtr(),
        buffer_manager.UniqueDiffIndicesPtr(), buffer_manager.UniqueDiffValuesPtr(),
        buffer_manager.UniqueWorkspacePtr(), buffer_manager.UniqueWorkspaceBytes());

    MdUpdateUtilT::Update(ctx->device_ctx(), beta1, beta2, epsilon, num_indices, feature_size,
                          kernel_state->lower(), kernel_state->upper(),
                          buffer_manager.NumUniqueDiffIndicesPtr(), learning_rate->dptr<float>(),
                          buffer_manager.UniqueDiffIndicesPtr(),
                          buffer_manager.UniqueDiffValuesPtr(), model->mut_dptr<T>(),
                          m->mut_dptr<T>(), v->mut_dptr<T>());
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

template<DeviceType device_type, typename T>
class LambTmpBufferManager final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LambTmpBufferManager);
  LambTmpBufferManager(void* ptr, const int64_t n) : ptr_(ptr) {
    const size_t adam_diff_bytes = GetCudaAlignedSize(n * sizeof(T));
    norm_buffer_bytes_ = GetCudaAlignedSize(2 * sizeof(T));
    adam_diff_offset_ = 0;
    norm_buffer_offset_ = adam_diff_offset_ + adam_diff_bytes;
    total_buffer_size_ = adam_diff_bytes + norm_buffer_bytes_;
  }
  ~LambTmpBufferManager() = default;

  size_t GetNormBufferSize() const { return norm_buffer_bytes_; }
  size_t GetTotalBufferSize() const { return total_buffer_size_; }

  T* AdamDiffPtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<T*>(reinterpret_cast<char*>(ptr_) + adam_diff_offset_);
  }
  T* NormBufferPtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<T*>(reinterpret_cast<char*>(ptr_) + norm_buffer_offset_);
  }

 private:
  size_t adam_diff_offset_;
  size_t norm_buffer_offset_;

  size_t total_buffer_size_;
  size_t norm_buffer_bytes_;
  void* ptr_;
};

template<DeviceType device_type, typename T, typename G>
class LambUpdateKernel final : public user_op::OpKernel {
 public:
  LambUpdateKernel() = default;
  ~LambUpdateKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* learning_rate = ctx->Tensor4ArgNameAndIndex("learning_rate", 0);
    const user_op::Tensor* model_diff = ctx->Tensor4ArgNameAndIndex("model_diff", 0);
    user_op::Tensor* model = ctx->Tensor4ArgNameAndIndex("model", 0);
    user_op::Tensor* m = ctx->Tensor4ArgNameAndIndex("m", 0);
    user_op::Tensor* v = ctx->Tensor4ArgNameAndIndex("v", 0);
    user_op::Tensor* beta1_t = ctx->Tensor4ArgNameAndIndex("beta1_t", 0);
    user_op::Tensor* beta2_t = ctx->Tensor4ArgNameAndIndex("beta2_t", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    LambTmpBufferManager<device_type, T> tbm(tmp_buffer->mut_dptr(), model->shape().elem_cnt());
    const auto scale = ctx->Attr<double>("scale");
    const auto l1 = ctx->Attr<float>("l1");
    const auto l2 = ctx->Attr<float>("l2");
    const auto beta1 = ctx->Attr<float>("beta1");
    const auto beta2 = ctx->Attr<float>("beta2");
    const auto epsilon = ctx->Attr<float>("epsilon");
    const auto weight_decay = ctx->Attr<float>("weight_decay");
    const T* scale_by_ptr = nullptr;
    if (ctx->user_op_conf().has_input("scale_by_tensor", 0)) {
      const user_op::Tensor* scale_by_tensor = ctx->Tensor4ArgNameAndIndex("scale_by_tensor", 0);
      CHECK_EQ(scale_by_tensor->data_type(), model->data_type());
      CHECK_EQ(scale_by_tensor->shape().elem_cnt(), 1);
      scale_by_ptr = scale_by_tensor->dptr<T>();
    }
    const int64_t* skip_if_ptr = nullptr;
    if (ctx->user_op_conf().has_input("skip_if", 0)) {
      const user_op::Tensor* skip_if = ctx->Tensor4ArgNameAndIndex("skip_if", 0);
      CHECK_EQ(skip_if->shape().elem_cnt(), 1);
      skip_if_ptr = skip_if->dptr<int64_t>();
    }
    LambUpdateKernelUtil<device_type, T, G>::Update(
        ctx->device_ctx(), m->shape().elem_cnt(), scale, l1, l2, beta1, beta2, epsilon,
        weight_decay, learning_rate->dptr<float>(), scale_by_ptr, skip_if_ptr,
        model_diff->dptr<G>(), tbm.AdamDiffPtr(), model->mut_dptr<T>(), m->mut_dptr<T>(),
        v->mut_dptr<T>(), tbm.NormBufferPtr(), beta1_t->mut_dptr<T>(), beta2_t->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

template<DeviceType device_type, typename T>
user_op::InferTmpSizeFn LambGenInferTmpSizeFn() {
  return [](user_op::InferContext* ctx) {
    const user_op::TensorDesc* model = ctx->TensorDesc4ArgNameAndIndex("model", 0);
    LambTmpBufferManager<device_type, T> tbm(nullptr, model->shape().elem_cnt());
    return tbm.GetTotalBufferSize();
  };
}

#define REGISTER_LAMB_UPDATE_KERNEL(device, dtype, gtype)                                      \
  REGISTER_USER_KERNEL("lamb_update")                                                          \
      .SetCreateFn<LambUpdateKernel<device, dtype, gtype>>()                                   \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                     \
                       & (user_op::HobDataType("model", 0) == GetDataType<dtype>::value)       \
                       & (user_op::HobDataType("model_diff", 0) == GetDataType<gtype>::value)) \
      .SetInferTmpSizeFn(LambGenInferTmpSizeFn<device, dtype>());

REGISTER_LAMB_UPDATE_KERNEL(DeviceType::kCPU, float, float);
REGISTER_LAMB_UPDATE_KERNEL(DeviceType::kCPU, double, double);
#ifdef WITH_CUDA
REGISTER_LAMB_UPDATE_KERNEL(DeviceType::kGPU, float, float16);
REGISTER_LAMB_UPDATE_KERNEL(DeviceType::kGPU, float, float);
REGISTER_LAMB_UPDATE_KERNEL(DeviceType::kGPU, double, double);
#endif  // WITH_CUDA

template<DeviceType device_type>
class AdamBiasCorrectionLearningRateKernel final : public user_op::OpKernel {
 public:
  AdamBiasCorrectionLearningRateKernel() = default;
  ~AdamBiasCorrectionLearningRateKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* learning_rate = ctx->Tensor4ArgNameAndIndex("learning_rate", 0);
    const user_op::Tensor* train_step = ctx->Tensor4ArgNameAndIndex("train_step", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const auto beta1 = ctx->Attr<float>("beta1");
    const auto beta2 = ctx->Attr<float>("beta2");
    AdamBiasCorrectionLearningRateKernelUtil<device_type>::AdamBiasCorrectionLearningRate(
        ctx->device_ctx(), beta1, beta2, learning_rate->dptr<float>(), train_step->dptr<int64_t>(),
        out->mut_dptr<float>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

#define REGISTER_ADAM_BIAS_CORRECTION_LEARNING_RATE_KERNEL(device) \
  REGISTER_USER_KERNEL("adam_bias_correction_learning_rate")       \
      .SetCreateFn<AdamBiasCorrectionLearningRateKernel<device>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device));
REGISTER_ADAM_BIAS_CORRECTION_LEARNING_RATE_KERNEL(DeviceType::kCPU)
#ifdef WITH_CUDA
REGISTER_ADAM_BIAS_CORRECTION_LEARNING_RATE_KERNEL(DeviceType::kGPU)
#endif  // WITH_CUDA

template<DeviceType device_type, typename T, typename G>
class RmsPropUpdateKernel final : public user_op::OpKernel {
 public:
  RmsPropUpdateKernel() = default;
  ~RmsPropUpdateKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* learning_rate = ctx->Tensor4ArgNameAndIndex("learning_rate", 0);
    const user_op::Tensor* model_diff = ctx->Tensor4ArgNameAndIndex("model_diff", 0);
    user_op::Tensor* model = ctx->Tensor4ArgNameAndIndex("model", 0);
    user_op::Tensor* mean_square = ctx->Tensor4ArgNameAndIndex("mean_square", 0);
    const auto scale = ctx->Attr<double>("scale");
    const auto l1 = ctx->Attr<float>("l1");
    const auto l2 = ctx->Attr<float>("l2");
    const auto decay_rate = ctx->Attr<float>("decay_rate");
    const auto epsilon = ctx->Attr<float>("epsilon");
    const auto centered = ctx->Attr<bool>("centered");
    const auto weight_decay = ctx->Attr<float>("weight_decay");
    const T* scale_by_ptr = nullptr;
    if (ctx->user_op_conf().has_input("scale_by_tensor", 0)) {
      const user_op::Tensor* scale_by_tensor = ctx->Tensor4ArgNameAndIndex("scale_by_tensor", 0);
      CHECK_EQ(scale_by_tensor->data_type(), model->data_type());
      CHECK_EQ(scale_by_tensor->shape().elem_cnt(), 1);
      scale_by_ptr = scale_by_tensor->dptr<T>();
    }
    const int64_t* skip_if_ptr = nullptr;
    if (ctx->user_op_conf().has_input("skip_if", 0)) {
      const user_op::Tensor* skip_if = ctx->Tensor4ArgNameAndIndex("skip_if", 0);
      CHECK_EQ(skip_if->shape().elem_cnt(), 1);
      skip_if_ptr = skip_if->dptr<int64_t>();
    }
    T* mean_gradient_ptr = nullptr;
    if (centered) {
      user_op::Tensor* mean_gradient = ctx->Tensor4ArgNameAndIndex("mean_gradient", 0);
      mean_gradient_ptr = mean_gradient->mut_dptr<T>();
    }
    RmsPropUpdateKernelUtil<device_type, T, G>::Update(
        ctx->device_ctx(), model->shape().elem_cnt(), static_cast<T>(scale), l1, l2, centered,
        epsilon, weight_decay, decay_rate, learning_rate->dptr<float>(), scale_by_ptr, skip_if_ptr,
        model_diff->dptr<G>(), model->mut_dptr<T>(), mean_square->mut_dptr<T>(), mean_gradient_ptr);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

#define REGISTER_RMSPROP_UPDATE_KERNEL(device, dtype, gtype)                             \
  REGISTER_USER_KERNEL("rmsprop_update")                                                 \
      .SetCreateFn<RmsPropUpdateKernel<device, dtype, gtype>>()                          \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                               \
                       & (user_op::HobDataType("model", 0) == GetDataType<dtype>::value) \
                       & (user_op::HobDataType("model_diff", 0) == GetDataType<gtype>::value));

REGISTER_RMSPROP_UPDATE_KERNEL(DeviceType::kCPU, float, float);
REGISTER_RMSPROP_UPDATE_KERNEL(DeviceType::kCPU, double, double);
#ifdef WITH_CUDA
REGISTER_RMSPROP_UPDATE_KERNEL(DeviceType::kGPU, float, float16);
REGISTER_RMSPROP_UPDATE_KERNEL(DeviceType::kGPU, float, float);
REGISTER_RMSPROP_UPDATE_KERNEL(DeviceType::kGPU, double, double);
#endif  // WITH_CUDA

template<DeviceType device_type, typename T>
class LarsTmpBufferManager final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LarsTmpBufferManager);
  LarsTmpBufferManager(void* ptr, const int64_t n) : ptr_(ptr) {
    model_diff_size_ = GetCudaAlignedSize(n * sizeof(T));
    model_diff_offset_ = 0;
    data_tmp_size_ = GetCudaAlignedSize(3 * sizeof(T));
    data_tmp_offset_ = model_diff_offset_ + model_diff_size_;
    total_buffer_size_ = model_diff_size_ + data_tmp_size_;
  }
  ~LarsTmpBufferManager() = default;

  size_t GetTotalBufferSize() const { return total_buffer_size_; }
  size_t GetDataTmpBufferSize() const { return data_tmp_size_; }

  T* ModelDiffPtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<T*>(reinterpret_cast<char*>(ptr_) + model_diff_offset_);
  }

  T* DataTmpPtr() const {
    CHECK(ptr_ != nullptr);
    return reinterpret_cast<T*>(reinterpret_cast<char*>(ptr_) + data_tmp_offset_);
  }

 private:
  size_t model_diff_offset_;
  size_t model_diff_size_;
  size_t data_tmp_offset_;
  size_t data_tmp_size_;
  size_t total_buffer_size_;
  void* ptr_;
};

template<DeviceType device_type, typename T, typename G>
class LarsUpdateKernel final : public user_op::OpKernel {
 public:
  LarsUpdateKernel() = default;
  ~LarsUpdateKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* learning_rate = ctx->Tensor4ArgNameAndIndex("learning_rate", 0);
    const user_op::Tensor* train_step = ctx->Tensor4ArgNameAndIndex("train_step", 0);
    const user_op::Tensor* model_diff = ctx->Tensor4ArgNameAndIndex("model_diff", 0);
    user_op::Tensor* model = ctx->Tensor4ArgNameAndIndex("model", 0);
    user_op::Tensor* momentum = ctx->Tensor4ArgNameAndIndex("momentum", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    LarsTmpBufferManager<device_type, T> tlm(tmp_buffer->mut_dptr(), model->shape().elem_cnt());
    const auto scale = ctx->Attr<double>("scale");
    const auto l1 = ctx->Attr<float>("l1");
    const auto l2 = ctx->Attr<float>("l2");
    const auto momentum_beta = ctx->Attr<float>("momentum_beta");
    const auto epsilon = ctx->Attr<float>("epsilon");
    const auto lars_coefficient = ctx->Attr<float>("lars_coefficient");
    const auto weight_decay = ctx->Attr<float>("weight_decay");
    const T* scale_by_ptr = nullptr;
    if (ctx->user_op_conf().has_input("scale_by_tensor", 0)) {
      const user_op::Tensor* scale_by_tensor = ctx->Tensor4ArgNameAndIndex("scale_by_tensor", 0);
      CHECK_EQ(scale_by_tensor->data_type(), model->data_type());
      CHECK_EQ(scale_by_tensor->shape().elem_cnt(), 1);
      scale_by_ptr = scale_by_tensor->dptr<T>();
    }
    const int64_t* skip_if_ptr = nullptr;
    if (ctx->user_op_conf().has_input("skip_if", 0)) {
      const user_op::Tensor* skip_if = ctx->Tensor4ArgNameAndIndex("skip_if", 0);
      CHECK_EQ(skip_if->shape().elem_cnt(), 1);
      skip_if_ptr = skip_if->dptr<int64_t>();
    }
    LarsUpdateKernelUtil<device_type, T, G>::Update(
        ctx->device_ctx(), model->shape().elem_cnt(), static_cast<T>(scale), l1, l2, momentum_beta,
        epsilon, lars_coefficient, weight_decay, learning_rate->dptr<float>(),
        train_step->dptr<int64_t>(), scale_by_ptr, skip_if_ptr, model_diff->dptr<G>(),
        model->mut_dptr<T>(), momentum->mut_dptr<T>(), tlm.DataTmpPtr(), tlm.ModelDiffPtr());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

template<DeviceType device_type, typename T>
user_op::InferTmpSizeFn LarsGenInferTmpSizeFn() {
  return [](user_op::InferContext* ctx) {
    const user_op::TensorDesc* model = ctx->TensorDesc4ArgNameAndIndex("model", 0);
    LarsTmpBufferManager<device_type, T> tlm(nullptr, model->shape().elem_cnt());
    return tlm.GetTotalBufferSize();
  };
}

#define REGISTER_LARS_UPDATE_KERNEL(device, dtype, gtype)                                      \
  REGISTER_USER_KERNEL("lars_update")                                                          \
      .SetCreateFn<LarsUpdateKernel<device, dtype, gtype>>()                                   \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                                     \
                       & (user_op::HobDataType("model", 0) == GetDataType<dtype>::value)       \
                       & (user_op::HobDataType("model_diff", 0) == GetDataType<gtype>::value)) \
      .SetInferTmpSizeFn(LarsGenInferTmpSizeFn<device, dtype>());

REGISTER_LARS_UPDATE_KERNEL(DeviceType::kCPU, float, float);
REGISTER_LARS_UPDATE_KERNEL(DeviceType::kCPU, double, double);
#ifdef WITH_CUDA
REGISTER_LARS_UPDATE_KERNEL(DeviceType::kGPU, float, float16);
REGISTER_LARS_UPDATE_KERNEL(DeviceType::kGPU, float, float);
REGISTER_LARS_UPDATE_KERNEL(DeviceType::kGPU, double, double);
#endif  // WITH_CUDA

}  // namespace

}  // namespace oneflow
