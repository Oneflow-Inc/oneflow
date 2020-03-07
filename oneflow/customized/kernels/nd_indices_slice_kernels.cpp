#include "oneflow/customized/kernels/nd_indices_slice_util.h"

namespace oneflow {

template<typename T, typename I>
struct GatherNdOnDevice<DeviceType::kCPU, T, I> {
  static void Run(DeviceCtx* ctx, NdIndicesSliceParams<T, I>* params) {
    GatherNdFunctor<T, I>::Invoke(params->num_segms * params->segm_size, params->segm_size,
                                  params->segm_dim, params->shape, params->indices, params->dense,
                                  params->sparse);
  }
};

template<typename T, typename I, template<DeviceType, typename> class Opt>
struct ScatterNdOnDevice<DeviceType::kCPU, T, I, Opt> {
  static void Run(DeviceCtx* ctx, NdIndicesSliceParams<T, I>* params) {
    ScatterNdFunctor<T, I, Opt<DeviceType::kCPU, T>>::Invoke(
        params->num_segms * params->segm_size, params->segm_size, params->segm_dim, params->shape,
        params->indices, params->sparse, params->dense);
  }
};

template<typename T>
struct ScatterNdAddOpt<DeviceType::kCPU, T> {
  OF_DEVICE_FUNC static void Invoke(const T* x, T* y) { *y += *x; }
};

template<DeviceType device_type, typename T, typename I>
class GatherNdKernel final : public user_op::OpKernel {
 public:
  GatherNdKernel(const user_op::KernelInitContext& ctx) : user_op::OpKernel(ctx) {}
  GatherNdKernel() = default;
  ~GatherNdKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* indices = ctx->Tensor4ArgNameAndIndex("indices", 0);
    user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    auto params = ConstructNdIndicesSliceParams<T, I>(x, y, indices);
    NdIndicesSliceUtil<device_type, T, I>::GatherNd(ctx->device_ctx(), &params);
  }
};

template<DeviceType device_type, typename T, typename I>
class ScatterNdUpdateKernel final : public user_op::OpKernel {
 public:
  ScatterNdUpdateKernel(const user_op::KernelInitContext& ctx) : user_op::OpKernel(ctx) {}
  ScatterNdUpdateKernel() = default;
  ~ScatterNdUpdateKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* indices = ctx->Tensor4ArgNameAndIndex("indices", 0);
    user_op::Tensor* updates = ctx->Tensor4ArgNameAndIndex("updates", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    if (in->dptr<T>() != out->mut_dptr<T>()) {
      size_t out_bytes_size = out->shape().elem_cnt() * GetSizeOfDataType(out->data_type());
      Memcpy<device_type>(ctx->device_ctx(), out->mut_dptr<T>(), in->dptr<T>(), out_bytes_size);
    }
    auto params = ConstructNdIndicesSliceParams<T, I>(out, updates, indices);
    NdIndicesSliceUtil<device_type, T, I>::ScatterNdUpdate(ctx->device_ctx(), &params);
  }
};

template<DeviceType device_type, typename T, typename I>
class ScatterNdAddKernel final : public user_op::OpKernel {
 public:
  ScatterNdAddKernel(const user_op::KernelInitContext& ctx) : user_op::OpKernel(ctx) {}
  ScatterNdAddKernel() = default;
  ~ScatterNdAddKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* indices = ctx->Tensor4ArgNameAndIndex("indices", 0);
    user_op::Tensor* updates = ctx->Tensor4ArgNameAndIndex("updates", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    if (in->dptr<T>() != out->mut_dptr<T>()) {
      size_t out_bytes_size = out->shape().elem_cnt() * GetSizeOfDataType(out->data_type());
      Memcpy<device_type>(ctx->device_ctx(), out->mut_dptr<T>(), in->dptr<T>(), out_bytes_size);
    }
    auto params = ConstructNdIndicesSliceParams<T, I>(out, updates, indices);
    NdIndicesSliceUtil<device_type, T, I>::ScatterNdAdd(ctx->device_ctx(), &params);
  }
};

#define REGISTER_SCATTER_ND_OPT_KERNELS(opt, opt_name, device_type_v, dtype_pair, itype_pair)   \
  REGISTER_USER_KERNEL(#opt_name)                                                               \
      .SetCreateFn([](const oneflow::user_op::KernelInitContext& ctx) {                         \
        return new ScatterNd##opt##Kernel<device_type_v, OF_PP_PAIR_FIRST(dtype_pair),          \
                                          OF_PP_PAIR_FIRST(itype_pair)>(ctx);                   \
      })                                                                                        \
      .SetIsMatchedPred([](const oneflow::user_op::KernelRegContext& ctx) {                     \
        const user_op::TensorDesc* indices_desc = ctx.TensorDesc4ArgNameAndIndex("indices", 0); \
        const user_op::TensorDesc* out_desc = ctx.TensorDesc4ArgNameAndIndex("out", 0);         \
        if (ctx.device_type() == device_type_v                                                  \
            && indices_desc->data_type() == OF_PP_PAIR_SECOND(itype_pair)                       \
            && out_desc->data_type() == OF_PP_PAIR_SECOND(dtype_pair)) {                        \
          return true;                                                                          \
        }                                                                                       \
        return false;                                                                           \
      })                                                                                        \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));                       \
        return Maybe<void>::Ok();                                                               \
      });

#define REGISTER_GATHER_SCATTER_ND_KERNELS(device_type_v, dtype_pair, itype_pair)               \
  REGISTER_USER_KERNEL("gather_nd")                                                             \
      .SetCreateFn([](const oneflow::user_op::KernelInitContext& ctx) {                         \
        return new GatherNdKernel<device_type_v, OF_PP_PAIR_FIRST(dtype_pair),                  \
                                  OF_PP_PAIR_FIRST(itype_pair)>(ctx);                           \
      })                                                                                        \
      .SetIsMatchedPred([](const oneflow::user_op::KernelRegContext& ctx) {                     \
        const user_op::TensorDesc* indices_desc = ctx.TensorDesc4ArgNameAndIndex("indices", 0); \
        const user_op::TensorDesc* y_desc = ctx.TensorDesc4ArgNameAndIndex("y", 0);             \
        if (ctx.device_type() == device_type_v                                                  \
            && indices_desc->data_type() == OF_PP_PAIR_SECOND(itype_pair)                       \
            && y_desc->data_type() == OF_PP_PAIR_SECOND(dtype_pair)) {                          \
          return true;                                                                          \
        }                                                                                       \
        return false;                                                                           \
      });                                                                                       \
  REGISTER_SCATTER_ND_OPT_KERNELS(Update, scatter_nd_update, device_type_v, dtype_pair,         \
                                  itype_pair)                                                   \
  REGISTER_SCATTER_ND_OPT_KERNELS(Add, scatter_nd_add, device_type_v, dtype_pair, itype_pair)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_GATHER_SCATTER_ND_KERNELS, DEVICE_TYPE_SEQ,
                                 GATHER_ND_DATA_TYPE_SEQ, GATHER_ND_INDEX_TYPE_SEQ)

}  // namespace oneflow
