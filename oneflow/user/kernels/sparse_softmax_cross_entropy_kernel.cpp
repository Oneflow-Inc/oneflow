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
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/user/kernels/sparse_cross_entropy_kernel_util.h"
#include "oneflow/user/kernels/sparse_softmax_cross_entropy_kernel_util.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/job/nd_sbp_util.h"
#include "oneflow/core/ep/include/primitive/log_softmax.h"

namespace oneflow {
namespace user_op {

namespace {

template<typename Context>
std::unique_ptr<ep::primitive::LogSoftmax> NewLogSoftmaxPrimitive(Context* ctx) {
  const DataType data_type = ctx->TensorDesc4ArgNameAndIndex("prediction", 0)->data_type();
  return ep::primitive::NewPrimitive<ep::primitive::LogSoftmaxFactory>(ctx->device_type(),
                                                                       data_type);
}

auto LogSoftmaxPrimitiveExists() {
  return hob::make_custom("LogSoftmaxPrimitiveExists", [](const user_op::KernelRegContext& ctx) {
    return NewLogSoftmaxPrimitive(&ctx).operator bool();
  });
}

class SparseSoftmaxCrossEntropyOpKernelCache final : public user_op::OpKernelCache {
 public:
  SparseSoftmaxCrossEntropyOpKernelCache(int64_t lower, int64_t upper)
      : lower_(lower), upper_(upper) {}
  ~SparseSoftmaxCrossEntropyOpKernelCache() override = default;

  int64_t lower() const { return lower_; }
  int64_t upper() const { return upper_; }

 private:
  const int64_t lower_;
  const int64_t upper_;
};

}  // namespace

template<DeviceType device_type, typename T, typename K>
class SparseSoftmaxCrossEntropyKernel final : public user_op::OpKernel,
                                              public user_op::CudaGraphSupport {
 public:
  SparseSoftmaxCrossEntropyKernel() = default;
  ~SparseSoftmaxCrossEntropyKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* prediction = ctx->Tensor4ArgNameAndIndex("prediction", 0);
    const user_op::Tensor* label = ctx->Tensor4ArgNameAndIndex("label", 0);
    user_op::Tensor* prob = ctx->Tensor4ArgNameAndIndex("prob", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int64_t num_instances = label->shape_view().elem_cnt();
    CHECK_EQ(prediction->shape_view().elem_cnt() % num_instances, 0);
    const int64_t num_classes = prediction->shape_view().elem_cnt() / num_instances;
    const int64_t lower_bound = 0;
    const int64_t depth = ctx->Attr<int64_t>("depth");

    std::unique_ptr<ep::primitive::LogSoftmax> primitive = NewLogSoftmaxPrimitive(ctx);
    CHECK(primitive);
    primitive->Launch(ctx->stream(), num_instances, num_classes, prediction->dptr(),
                      prob->mut_dptr());

    const K* labels = label->dptr<K>();
    const T* prob_ptr = prob->dptr<T>();
    T* out_ptr = out->mut_dptr<T>();

    FOR_RANGE(int64_t, i, 0, num_instances) {
      CHECK_GE(labels[i], 0);
      CHECK_LT(labels[i], depth);
      K _label = labels[i] - lower_bound;
      if (_label >= 0 && _label < num_classes) { out_ptr[i] = -prob_ptr[i * num_classes + _label]; }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename T, typename K>
class SparseSoftmaxCrossEntropyMsKernel final : public user_op::OpKernel {
 public:
  SparseSoftmaxCrossEntropyMsKernel() = default;
  ~SparseSoftmaxCrossEntropyMsKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    LOG(FATAL) << "SparseSoftmaxCrossEntropyMsKernel should be split to ops";
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_KERNEL(kernel_class, kernel_name, device_type_v, \
                                                     dtype_pair, ltype_pair)                   \
  REGISTER_USER_KERNEL(kernel_name)                                                            \
      .SetCreateFn<kernel_class<device_type_v, OF_PP_PAIR_FIRST(dtype_pair),                   \
                                OF_PP_PAIR_FIRST(ltype_pair)>>()                               \
      .SetIsMatchedHob((user_op::HobDeviceType() == device_type_v)                             \
                       && (user_op::HobDataType("label", 0) == OF_PP_PAIR_SECOND(ltype_pair))  \
                       && (user_op::HobDataType("out", 0) == OF_PP_PAIR_SECOND(dtype_pair))    \
                       && LogSoftmaxPrimitiveExists());

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_KERNEL,
                                 (SparseSoftmaxCrossEntropyKernel),
                                 ("sparse_softmax_cross_entropy"),
                                 OF_PP_MAKE_TUPLE_SEQ(DeviceType::kCPU), FLOATING_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ)
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_KERNEL,
                                 (SparseSoftmaxCrossEntropyMsKernel),
                                 ("sparse_softmax_cross_entropy_ms"),
                                 OF_PP_MAKE_TUPLE_SEQ(DeviceType::kCPU), FLOATING_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ)
#ifdef WITH_CUDA
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_KERNEL,
                                 (SparseSoftmaxCrossEntropyMsKernel),
                                 ("sparse_softmax_cross_entropy_ms"),
                                 OF_PP_MAKE_TUPLE_SEQ(DeviceType::kCUDA),
                                 FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)
#endif

template<DeviceType device_type, typename T, typename K>
class SparseSoftmaxCrossEntropyGradKernel final : public user_op::OpKernel,
                                                  public user_op::CudaGraphSupport {
 public:
  SparseSoftmaxCrossEntropyGradKernel() = default;
  ~SparseSoftmaxCrossEntropyGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* label = ctx->Tensor4ArgNameAndIndex("label", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* prob = ctx->Tensor4ArgNameAndIndex("prob", 0);
    user_op::Tensor* prediction_diff = ctx->Tensor4ArgNameAndIndex("prediction_diff", 0);
    const int64_t num_instances = label->shape_view().elem_cnt();
    CHECK_EQ(prob->shape_view().elem_cnt() % num_instances, 0);
    const int64_t num_classes = prob->shape_view().elem_cnt() / num_instances;
    const int64_t lower_bound = 0;
    const int64_t depth = ctx->Attr<int64_t>("depth");
    SparseSoftmaxCrossEntropyKernelUtil<device_type, T, K>::ComputeDiff(
        ctx->stream(), prediction_diff->shape_view().elem_cnt(), num_classes, depth, lower_bound,
        prob->dptr<T>(), label->dptr<K>(), dy->dptr<T>(), prediction_diff->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename T, typename K>
class SparseSoftmaxCrossEntropyMsGradKernel final : public user_op::OpKernel {
 public:
  SparseSoftmaxCrossEntropyMsGradKernel() = default;
  ~SparseSoftmaxCrossEntropyMsGradKernel() = default;
  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    if (ctx->parallel_ctx().parallel_num() > 1) {
      const NdSbp& nd_sbp = ctx->NdSbp4ArgNameAndIndex("prob", 0);
      const Shape& hierarchy = *ctx->parallel_desc().hierarchy();
      const TensorDesc* prob_logical_desc = ctx->LogicalTensorDesc4ArgNameAndIndex("prob", 0);
      const int64_t class_axis = prob_logical_desc->shape().NumAxes() - 1;
      TensorSliceView view = GetTensorSliceView4ParallelId(
          hierarchy, nd_sbp, prob_logical_desc->shape(), ctx->parallel_ctx().parallel_id());
      return std::make_shared<SparseSoftmaxCrossEntropyOpKernelCache>(view.At(class_axis).begin(),
                                                                      view.At(class_axis).end());
    } else {
      return nullptr;
    }
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    const user_op::Tensor* label = ctx->Tensor4ArgNameAndIndex("label", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* prob = ctx->Tensor4ArgNameAndIndex("prob", 0);
    user_op::Tensor* prediction_diff = ctx->Tensor4ArgNameAndIndex("prediction_diff", 0);
    const int64_t num_instances = label->shape_view().elem_cnt();
    CHECK_EQ(prob->shape_view().elem_cnt() % num_instances, 0);
    const int64_t num_classes = prob->shape_view().elem_cnt() / num_instances;
    const int64_t depth = ctx->Attr<int64_t>("depth");
    int64_t lower_bound = 0;
    if (cache != nullptr) {
      auto* kernel_cache = dynamic_cast<const SparseSoftmaxCrossEntropyOpKernelCache*>(cache);
      CHECK_NOTNULL(kernel_cache);
      CHECK_EQ(num_classes, kernel_cache->upper() - kernel_cache->lower());
      lower_bound = kernel_cache->lower();
    }
    SparseCrossEntropyKernelUtil<device_type, T, K>::ComputeDiffWithSoftmax(
        ctx->stream(), prediction_diff->shape_view().elem_cnt(), num_classes, depth, lower_bound,
        prob->dptr<T>(), label->dptr<K>(), dy->dptr<T>(), prediction_diff->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_GRAD_KERNEL(kernel_class, kernel_name,             \
                                                          device_type_v, dtype_pair, ltype_pair) \
  REGISTER_USER_KERNEL(kernel_name)                                                              \
      .SetCreateFn<kernel_class<device_type_v, OF_PP_PAIR_FIRST(dtype_pair),                     \
                                OF_PP_PAIR_FIRST(ltype_pair)>>()                                 \
      .SetIsMatchedHob(                                                                          \
          (user_op::HobDeviceType() == device_type_v)                                            \
          && (user_op::HobDataType("label", 0) == OF_PP_PAIR_SECOND(ltype_pair))                 \
          && (user_op::HobDataType("prediction_diff", 0) == OF_PP_PAIR_SECOND(dtype_pair)))      \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                     \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> {  \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("prediction_diff", 0, "prob", 0, true));          \
        return Maybe<void>::Ok();                                                                \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_GRAD_KERNEL,
                                 (SparseSoftmaxCrossEntropyGradKernel),
                                 ("sparse_softmax_cross_entropy_grad"),
                                 OF_PP_MAKE_TUPLE_SEQ(DeviceType::kCPU), FLOATING_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ)
#ifdef WITH_CUDA
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_GRAD_KERNEL,
                                 (SparseSoftmaxCrossEntropyGradKernel),
                                 ("sparse_softmax_cross_entropy_grad"),
                                 OF_PP_MAKE_TUPLE_SEQ(DeviceType::kCUDA),
                                 FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)
#endif
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_GRAD_KERNEL,
                                 (SparseSoftmaxCrossEntropyMsGradKernel),
                                 ("sparse_softmax_cross_entropy_ms_grad"),
                                 OF_PP_MAKE_TUPLE_SEQ(DeviceType::kCPU), FLOATING_DATA_TYPE_SEQ,
                                 INDEX_DATA_TYPE_SEQ)
#ifdef WITH_CUDA
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SPARSE_SOFTMAX_CROSS_ENTROPY_GRAD_KERNEL,
                                 (SparseSoftmaxCrossEntropyMsGradKernel),
                                 ("sparse_softmax_cross_entropy_ms_grad"),
                                 OF_PP_MAKE_TUPLE_SEQ(DeviceType::kCUDA),
                                 FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)
#endif
}  // namespace user_op
}  // namespace oneflow
