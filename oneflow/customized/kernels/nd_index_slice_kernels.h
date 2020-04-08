#ifndef ONEFLOW_CUSTOMIZED_KERNELS_ND_INDEX_SLICE_KERNELS_H_
#define ONEFLOW_CUSTOMIZED_KERNELS_ND_INDEX_SLICE_KERNELS_H_

#include "oneflow/customized/kernels/nd_index_slice_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename I>
class GatherNdKernel final : public user_op::OpKernel {
 public:
  GatherNdKernel() = default;
  ~GatherNdKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override;
};

template<DeviceType device_type, typename T, typename I>
class ScatterNdKernel final : public user_op::OpKernel {
 public:
  ScatterNdKernel() = default;
  ~ScatterNdKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override;
};

template<DeviceType device_type, typename T, typename I>
class TensorScatterNdUpdateKernel final : public user_op::OpKernel {
 public:
  TensorScatterNdUpdateKernel() = default;
  ~TensorScatterNdUpdateKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override;
};

template<DeviceType device_type, typename T, typename I>
class TensorScatterNdAddKernel final : public user_op::OpKernel {
 public:
  TensorScatterNdAddKernel() = default;
  ~TensorScatterNdAddKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override;
};

template<DeviceType device_type, typename T, typename I>
void GatherNdKernel<device_type, T, I>::Compute(user_op::KernelComputeContext* ctx) const {
  const user_op::Tensor* indices = ctx->Tensor4ArgNameAndIndex("indices", 0);
  const user_op::Tensor* params = ctx->Tensor4ArgNameAndIndex("params", 0);
  user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
  if (indices->shape().elem_cnt() == 0) { return; }
  auto args = ConstructNdIndexSliceArgs<T, I>(*params, *out, *indices);
  GatherNdFunctor<device_type, T, I>()(ctx->device_ctx(), args, indices->dptr<I>(),
                                       params->dptr<T>(), out->mut_dptr<T>());
}

template<DeviceType device_type, typename T, typename I>
void ScatterNdKernel<device_type, T, I>::Compute(user_op::KernelComputeContext* ctx) const {
  const user_op::Tensor* indices = ctx->Tensor4ArgNameAndIndex("indices", 0);
  const user_op::Tensor* updates = ctx->Tensor4ArgNameAndIndex("updates", 0);
  user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
  size_t out_bytes_size = out->shape().elem_cnt() * GetSizeOfDataType(out->data_type());
  Memset<device_type>(ctx->device_ctx(), out->mut_dptr<T>(), 0, out_bytes_size);
  if (indices->shape().elem_cnt() == 0) { return; }
  auto args = ConstructNdIndexSliceArgs<T, I>(*out, *updates, *indices);
  ScatterNdAddFunctor<device_type, T, I>()(ctx->device_ctx(), args, indices->dptr<I>(),
                                           updates->dptr<T>(), out->mut_dptr<T>());
}

template<DeviceType device_type, typename T, typename I>
void TensorScatterNdUpdateKernel<device_type, T, I>::Compute(
    user_op::KernelComputeContext* ctx) const {
  const user_op::Tensor* params = ctx->Tensor4ArgNameAndIndex("params", 0);
  const user_op::Tensor* indices = ctx->Tensor4ArgNameAndIndex("indices", 0);
  const user_op::Tensor* updates = ctx->Tensor4ArgNameAndIndex("updates", 0);
  user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
  size_t out_bytes_size = out->shape().elem_cnt() * GetSizeOfDataType(out->data_type());
  Memcpy<device_type>(ctx->device_ctx(), out->mut_dptr<T>(), params->dptr<T>(), out_bytes_size);
  if (indices->shape().elem_cnt() == 0) { return; }
  auto args = ConstructNdIndexSliceArgs<T, I>(*params, *updates, *indices);
  ZeroByNdIndexFunctor<device_type, T, I>()(ctx->device_ctx(), args, indices->dptr<I>(),
                                            out->mut_dptr<T>());
  ScatterNdAddFunctor<device_type, T, I>()(ctx->device_ctx(), args, indices->dptr<I>(),
                                           updates->dptr<T>(), out->mut_dptr<T>());
}

template<DeviceType device_type, typename T, typename I>
void TensorScatterNdAddKernel<device_type, T, I>::Compute(
    user_op::KernelComputeContext* ctx) const {
  const user_op::Tensor* params = ctx->Tensor4ArgNameAndIndex("params", 0);
  const user_op::Tensor* indices = ctx->Tensor4ArgNameAndIndex("indices", 0);
  const user_op::Tensor* updates = ctx->Tensor4ArgNameAndIndex("updates", 0);
  user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
  size_t out_bytes_size = out->shape().elem_cnt() * GetSizeOfDataType(out->data_type());
  Memcpy<device_type>(ctx->device_ctx(), out->mut_dptr<T>(), params->dptr<T>(), out_bytes_size);
  if (indices->shape().elem_cnt() == 0) { return; }
  auto args = ConstructNdIndexSliceArgs<T, I>(*params, *updates, *indices);
  ScatterNdAddFunctor<device_type, T, I>()(ctx->device_ctx(), args, indices->dptr<I>(),
                                           updates->dptr<T>(), out->mut_dptr<T>());
}

namespace {

template<DeviceType device_type, typename T, typename I>
std::function<bool(const oneflow::user_op::KernelRegContext&)>
MakeNdIndexSliceKernelMatchedPredictor() {
  return [](const oneflow::user_op::KernelRegContext& ctx) {
    const user_op::TensorDesc* indices_desc = ctx.TensorDesc4ArgNameAndIndex("indices", 0);
    const user_op::TensorDesc* out_desc = ctx.TensorDesc4ArgNameAndIndex("out", 0);
    if (ctx.device_type() == device_type && indices_desc->data_type() == GetDataType<I>::value
        && out_desc->data_type() == GetDataType<T>::value) {
      return true;
    }
    return false;
  };
}

}  // namespace

#define REGISTER_GATHER_SCATTER_ND_KERNELS(op_type_name, op, device_type_v, dtype_pair,            \
                                           itype_pair)                                             \
  REGISTER_USER_KERNEL(#op_type_name)                                                              \
      .SetCreateFn<                                                                                \
          op##Kernel<device_type_v, OF_PP_PAIR_FIRST(dtype_pair), OF_PP_PAIR_FIRST(itype_pair)>>() \
      .SetIsMatchedPred(                                                                           \
          MakeNdIndexSliceKernelMatchedPredictor<device_type_v, OF_PP_PAIR_FIRST(dtype_pair),      \
                                                 OF_PP_PAIR_FIRST(itype_pair)>());

#define REGISTER_TENSOR_SCATTER_ND_OPT_KERNELS(op_type_name, opt, device_type_v, dtype_pair,    \
                                               itype_pair)                                      \
  REGISTER_USER_KERNEL(#op_type_name)                                                           \
      .SetCreateFn<TensorScatterNd##opt##Kernel<device_type_v, OF_PP_PAIR_FIRST(dtype_pair),    \
                                                OF_PP_PAIR_FIRST(itype_pair)>>()                \
      .SetIsMatchedPred(                                                                        \
          MakeNdIndexSliceKernelMatchedPredictor<device_type_v, OF_PP_PAIR_FIRST(dtype_pair),   \
                                                 OF_PP_PAIR_FIRST(itype_pair)>())               \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "params", 0, true));                   \
        return Maybe<void>::Ok();                                                               \
      });

#define REGISTER_GATHER_ND_KERNELS(device_type_v, dtype_pair, itype_pair) \
  REGISTER_GATHER_SCATTER_ND_KERNELS(gather_nd, GatherNd, device_type_v, dtype_pair, itype_pair)

#define REGISTER_SCATTER_ND_KERNELS(device_type_v, dtype_pair, itype_pair) \
  REGISTER_GATHER_SCATTER_ND_KERNELS(scatter_nd, ScatterNd, device_type_v, dtype_pair, itype_pair)

#define REGISTER_SCATTER_ND_LIKE_KERNELS(device_type_v, dtype_pair, itype_pair)             \
  REGISTER_GATHER_SCATTER_ND_KERNELS(scatter_nd_like, ScatterNd, device_type_v, dtype_pair, \
                                     itype_pair)

#define REGISTER_TENSOR_GATHER_ND_UPDATE_KERNELS(device_type_v, dtype_pair, itype_pair)   \
  REGISTER_TENSOR_SCATTER_ND_OPT_KERNELS(tensor_scatter_nd_update, Update, device_type_v, \
                                         dtype_pair, itype_pair)

#define REGISTER_TENSOR_GATHER_ND_ADD_KERNELS(device_type_v, dtype_pair, itype_pair)            \
  REGISTER_TENSOR_SCATTER_ND_OPT_KERNELS(tensor_scatter_nd_add, Add, device_type_v, dtype_pair, \
                                         itype_pair)

#define REGISTER_ND_INDEX_SLICE_KERNELS(device_type_v, dtype_pair, itype_pair)    \
  REGISTER_GATHER_ND_KERNELS(device_type_v, dtype_pair, itype_pair)               \
  REGISTER_SCATTER_ND_KERNELS(device_type_v, dtype_pair, itype_pair)              \
  REGISTER_SCATTER_ND_LIKE_KERNELS(device_type_v, dtype_pair, itype_pair)         \
  REGISTER_TENSOR_GATHER_ND_UPDATE_KERNELS(device_type_v, dtype_pair, itype_pair) \
  REGISTER_TENSOR_GATHER_ND_ADD_KERNELS(device_type_v, dtype_pair, itype_pair)

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_KERNELS_ND_INDEX_SLICE_KERNELS_H_
