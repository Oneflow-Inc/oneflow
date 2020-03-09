#ifndef ONEFLOW_CUSTOMIZED_KERNELS_ND_INDEX_SLICE_KERNELS_H_
#define ONEFLOW_CUSTOMIZED_KERNELS_ND_INDEX_SLICE_KERNELS_H_

#include "oneflow/customized/kernels/nd_index_slice_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename I>
class GatherNdKernel final : public user_op::OpKernel {
 public:
  GatherNdKernel(const user_op::KernelInitContext& ctx) : user_op::OpKernel(ctx) {}
  GatherNdKernel() = default;
  ~GatherNdKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override;
};

template<DeviceType device_type, typename T, typename I>
class ScatterNdKernel final : public user_op::OpKernel {
 public:
  ScatterNdKernel(const user_op::KernelInitContext& ctx) : user_op::OpKernel(ctx) {}
  ScatterNdKernel() = default;
  ~ScatterNdKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override;
};

template<DeviceType device_type, typename T, typename I>
class ScatterNdUpdateKernel final : public user_op::OpKernel {
 public:
  ScatterNdUpdateKernel(const user_op::KernelInitContext& ctx) : user_op::OpKernel(ctx) {}
  ScatterNdUpdateKernel() = default;
  ~ScatterNdUpdateKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override;
};

template<DeviceType device_type, typename T, typename I>
class ScatterNdAddKernel final : public user_op::OpKernel {
 public:
  ScatterNdAddKernel(const user_op::KernelInitContext& ctx) : user_op::OpKernel(ctx) {}
  ScatterNdAddKernel() = default;
  ~ScatterNdAddKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override;
};

template<DeviceType device_type, typename T, typename I>
void GatherNdKernel<device_type, T, I>::Compute(user_op::KernelContext* ctx) {
  const user_op::Tensor* indices = ctx->Tensor4ArgNameAndIndex("indices", 0);
  user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("params", 0);
  user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("out", 0);
  auto params = ConstructNdIndexSliceParams<T, I>(x, y, indices);
  NdIndicesSliceUtil<device_type, T, I>::GatherNd(ctx->device_ctx(), &params);
}

template<DeviceType device_type, typename T, typename I>
void ScatterNdKernel<device_type, T, I>::Compute(user_op::KernelContext* ctx) {
  const user_op::Tensor* indices = ctx->Tensor4ArgNameAndIndex("indices", 0);
  user_op::Tensor* updates = ctx->Tensor4ArgNameAndIndex("updates", 0);
  user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
  size_t out_bytes_size = out->shape().elem_cnt() * GetSizeOfDataType(out->data_type());
  Memset<device_type>(ctx->device_ctx(), out->mut_dptr<T>(), 0, out_bytes_size);
  auto params = ConstructNdIndexSliceParams<T, I>(out, updates, indices);
  NdIndicesSliceUtil<device_type, T, I>::ScatterNdAdd(ctx->device_ctx(), &params);
}

template<DeviceType device_type, typename T, typename I>
void ScatterNdUpdateKernel<device_type, T, I>::Compute(user_op::KernelContext* ctx) {
  const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("params", 0);
  const user_op::Tensor* indices = ctx->Tensor4ArgNameAndIndex("indices", 0);
  user_op::Tensor* updates = ctx->Tensor4ArgNameAndIndex("updates", 0);
  user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
  if (in->dptr<T>() != out->mut_dptr<T>()) {
    size_t out_bytes_size = out->shape().elem_cnt() * GetSizeOfDataType(out->data_type());
    Memcpy<device_type>(ctx->device_ctx(), out->mut_dptr<T>(), in->dptr<T>(), out_bytes_size);
  }
  auto params = ConstructNdIndexSliceParams<T, I>(out, updates, indices);
  NdIndicesSliceUtil<device_type, T, I>::ScatterNdUpdate(ctx->device_ctx(), &params);
}

template<DeviceType device_type, typename T, typename I>
void ScatterNdAddKernel<device_type, T, I>::Compute(user_op::KernelContext* ctx) {
  const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("params", 0);
  const user_op::Tensor* indices = ctx->Tensor4ArgNameAndIndex("indices", 0);
  user_op::Tensor* updates = ctx->Tensor4ArgNameAndIndex("updates", 0);
  user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
  if (in->dptr<T>() != out->mut_dptr<T>()) {
    size_t out_bytes_size = out->shape().elem_cnt() * GetSizeOfDataType(out->data_type());
    Memcpy<device_type>(ctx->device_ctx(), out->mut_dptr<T>(), in->dptr<T>(), out_bytes_size);
  }
  auto params = ConstructNdIndexSliceParams<T, I>(out, updates, indices);
  NdIndicesSliceUtil<device_type, T, I>::ScatterNdAdd(ctx->device_ctx(), &params);
}

namespace {

template<DeviceType device_type, typename T, typename I>
std::function<bool(const oneflow::user_op::KernelRegContext&)>
MakeGatherScatterNdKernelMatchedPredictor() {
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

#define GATHER_ND_DATA_TYPE_SEQ                   \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32) \
  OF_PP_MAKE_TUPLE_SEQ(float, DataType::kFloat)   \
  OF_PP_MAKE_TUPLE_SEQ(double, DataType::kDouble)

#define GATHER_ND_INDEX_TYPE_SEQ                  \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32) \
  OF_PP_MAKE_TUPLE_SEQ(int64_t, DataType::kInt64)

#define REGISTER_GATHER_SCATTER_ND_KERNELS(op_type_name, op, device_type_v, dtype_pair,          \
                                           itype_pair)                                           \
  REGISTER_USER_KERNEL(#op_type_name)                                                            \
      .SetCreateFn([](const oneflow::user_op::KernelInitContext& ctx) {                          \
        return new op##Kernel<device_type_v, OF_PP_PAIR_FIRST(dtype_pair),                       \
                              OF_PP_PAIR_FIRST(itype_pair)>(ctx);                                \
      })                                                                                         \
      .SetIsMatchedPred(                                                                         \
          MakeGatherScatterNdKernelMatchedPredictor<device_type_v, OF_PP_PAIR_FIRST(dtype_pair), \
                                                    OF_PP_PAIR_FIRST(itype_pair)>());

#define REGISTER_SCATTER_ND_OPT_KERNELS(op_type_name, opt, device_type_v, dtype_pair, itype_pair) \
  REGISTER_USER_KERNEL(#op_type_name)                                                             \
      .SetCreateFn([](const oneflow::user_op::KernelInitContext& ctx) {                           \
        return new ScatterNd##opt##Kernel<device_type_v, OF_PP_PAIR_FIRST(dtype_pair),            \
                                          OF_PP_PAIR_FIRST(itype_pair)>(ctx);                     \
      })                                                                                          \
      .SetIsMatchedPred(                                                                          \
          MakeGatherScatterNdKernelMatchedPredictor<device_type_v, OF_PP_PAIR_FIRST(dtype_pair),  \
                                                    OF_PP_PAIR_FIRST(itype_pair)>())              \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                      \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> {   \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "params", 0, true));                     \
        return Maybe<void>::Ok();                                                                 \
      });

#define REGISTER_ND_INDEX_SLICE_KERNELS(device_type_v, dtype_pair, itype_pair)                     \
  REGISTER_GATHER_SCATTER_ND_KERNELS(gather_nd, GatherNd, device_type_v, dtype_pair, itype_pair)   \
  REGISTER_GATHER_SCATTER_ND_KERNELS(scatter_nd, ScatterNd, device_type_v, dtype_pair, itype_pair) \
  REGISTER_SCATTER_ND_OPT_KERNELS(tensor_scatter_nd_update, Update, device_type_v, dtype_pair,     \
                                  itype_pair)                                                      \
  REGISTER_SCATTER_ND_OPT_KERNELS(tensor_scatter_nd_add, Add, device_type_v, dtype_pair, itype_pair)

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_KERNELS_ND_INDEX_SLICE_KERNELS_H_
