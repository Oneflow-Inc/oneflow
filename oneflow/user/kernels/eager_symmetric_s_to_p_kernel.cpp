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
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/nd_sbp_util.h"
#include "oneflow/core/register/tensor_slice_copier.h"
#include "oneflow/core/control/global_process_ctx.h"

namespace oneflow {

namespace {

template<typename Context>
std::unique_ptr<ep::primitive::Memset> NewMemsetPrimitive(Context* ctx) {
  return ep::primitive::NewPrimitive<ep::primitive::MemsetFactory>(ctx->device_type());
}

auto MemsetPrimitiveExists() {
  return hob::make_custom("MemsetPrimitiveExists", [](const user_op::KernelRegContext& ctx) {
    return NewMemsetPrimitive(&ctx).operator bool();
  });
}

Maybe<Symbol<NdSbp>> GetAllSplitNdSbp(int64_t axis, int64_t ndim) {
  NdSbp split_nd_sbp;
  for (int64_t i = 0; i < ndim; ++i) {
    split_nd_sbp.mutable_sbp_parallel()->Add()->mutable_split_parallel()->set_axis(axis);
  }
  return SymbolOf(split_nd_sbp);
}

auto* CachedGetAllSplitNdSbp = DECORATE(&GetAllSplitNdSbp, ThreadLocal);

Maybe<Symbol<NdSbp>> GetAllPartialSumNdSbp(int64_t ndim) {
  NdSbp split_nd_sbp;
  for (int64_t i = 0; i < ndim; ++i) {
    split_nd_sbp.mutable_sbp_parallel()->Add()->mutable_partial_sum_parallel();
  }
  return SymbolOf(split_nd_sbp);
}

auto* CachedGetAllPartialSumNdSbp = DECORATE(&GetAllPartialSumNdSbp, ThreadLocal);

class EagerSymmetricSToPOpKernelCache final : public user_op::OpKernelCache {
 public:
  explicit EagerSymmetricSToPOpKernelCache(user_op::KernelCacheContext* ctx) { Init(ctx); }
  ~EagerSymmetricSToPOpKernelCache() override = default;

  const std::shared_ptr<TensorSliceCopier>& tensor_slice_copier() const {
    return tensor_slice_copier_;
  }

 private:
  void Init(user_op::KernelCacheContext* ctx) {
    const std::string& parallel_conf_txt = ctx->Attr<std::string>("parallel_conf");
    const int64_t in_split_axis = ctx->Attr<int64_t>("in_split_axis");
    const user_op::TensorDesc* in_logical_desc = ctx->LogicalTensorDesc4ArgNameAndIndex("in", 0);
    const Shape& shape = in_logical_desc->shape();
    DeviceType device_type = ctx->device_type();
    DataType data_type = ctx->TensorDesc4ArgNameAndIndex("in", 0)->data_type();
    ParallelConf parallel_conf;
    CHECK(TxtString2PbMessage(parallel_conf_txt, &parallel_conf));
    Symbol<ParallelDesc> parallel_desc = SymbolOf(ParallelDesc(parallel_conf));

    const TensorSliceView& in_slice = GetTensorSliceView4ParallelId(
        *parallel_desc->hierarchy(),
        *CHECK_JUST(CachedGetAllSplitNdSbp(in_split_axis, parallel_desc->hierarchy()->NumAxes())),
        shape, ctx->parallel_ctx().parallel_id());
    CHECK(!in_slice.IsEmpty());
    const TensorSliceView& out_slice = GetTensorSliceView4ParallelId(
        *parallel_desc->hierarchy(),
        *CHECK_JUST(CachedGetAllPartialSumNdSbp(parallel_desc->hierarchy()->NumAxes())), shape,
        ctx->parallel_ctx().parallel_id());
    CHECK(!out_slice.IsEmpty());
    const TensorSliceView& intersection = out_slice.Intersect(in_slice);
    CHECK(!intersection.IsEmpty());
    tensor_slice_copier_ =
        std::make_shared<TensorSliceCopier>(out_slice, in_slice, data_type, device_type);
  }

  std::shared_ptr<TensorSliceCopier> tensor_slice_copier_;
};

}  // namespace

class EagerSymmetricSToPKernel final : public user_op::OpKernel {
 public:
  EagerSymmetricSToPKernel() = default;
  ~EagerSymmetricSToPKernel() override = default;

  void InitOpKernelCacheWithFlags(
      user_op::KernelCacheContext* ctx, int8_t flag,
      std::shared_ptr<user_op::OpKernelCache>* cache_ptr) const override {
    if (*cache_ptr == nullptr) {
      *cache_ptr = std::make_shared<EagerSymmetricSToPOpKernelCache>(ctx);
    }
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState*,
               const user_op::OpKernelCache* cache) const override {
    auto* kernel_cache = dynamic_cast<const EagerSymmetricSToPOpKernelCache*>(cache);
    CHECK(kernel_cache != nullptr);
    auto primitive = NewMemsetPrimitive(ctx);
    CHECK(primitive);  // NOLINT
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const auto& out_shape_view = out->shape_view();

    const void* in_ptr = in->dptr();
    void* out_ptr = out->mut_dptr();

    primitive->Launch(ctx->stream(), out->mut_dptr(), 0,
                      out_shape_view.elem_cnt() * GetSizeOfDataType(out->data_type()));
    const auto& tensor_slice_copier = kernel_cache->tensor_slice_copier();
    tensor_slice_copier->Copy(ctx->stream(), out_ptr, in_ptr);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("eager_symmetric_s_to_p")
    .SetCreateFn<EagerSymmetricSToPKernel>()
    .SetIsMatchedHob(MemsetPrimitiveExists() == true);

}  // namespace oneflow
