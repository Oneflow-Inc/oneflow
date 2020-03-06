// #include "oneflow/core/framework/framework.h"
// #include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/customized/kernels/nd_indices_slice_util.h"
#include "oneflow/core/kernel/util/cuda_kernel_util.h"

namespace oneflow {

namespace {

template<typename T, typename I>
__global__ void CudaGatherNd(NdIndicesSliceParams<T, I> params) {
  GatherNdFunctor<T, I>::Invoke(params.num_segms * params.segm_size, params.segm_size,
                                params.segm_dim, params.shape, params.indices, params.dense,
                                params.sparse);
}

template<typename T, typename I, template<DeviceType, typename> class Opt>
__global__ void CudaScatterNd(NdIndicesSliceParams<T, I> params) {
  ScatterNdFunctor<T, I, Opt<DeviceType::kGPU, T>>::Invoke(
      params.num_segms * params.segm_size, params.segm_size, params.segm_dim, params.shape,
      params.indices, params.sparse, params.dense);
}

}  // namespace

template<typename T, typename I>
struct GatherNdOnDevice<DeviceType::kGPU, T, I> {
  static void Run(DeviceCtx* ctx, NdIndicesSliceParams<T, I>* params) {
    RUN_CUDA_KERNEL((CudaGatherNd<T, I>), ctx, params->num_segms * params->segm_size, *params);
  }
};

template<typename T, typename I, template<DeviceType, typename> class Opt>
struct ScatterNdOnDevice<DeviceType::kGPU, T, I, Opt> {
  static void Run(DeviceCtx* ctx, NdIndicesSliceParams<T, I>* params) {
    RUN_CUDA_KERNEL((CudaScatterNd<T, I, Opt>), ctx, params->num_segms * params->segm_size,
                    *params);
  }
};

template<typename T>
struct ScatterNdAddOpt<DeviceType::kGPU, T> {
  OF_DEVICE_FUNC static void Invoke(const T* x, T* y) { gpu_atomic_add(y, *x); }
};

template<typename T, typename I>
class GatherNdGpuKernel final : public user_op::OpKernel {
 public:
  GatherNdGpuKernel(const user_op::KernelInitContext& ctx) : user_op::OpKernel(ctx) {}
  GatherNdGpuKernel() = default;
  ~GatherNdGpuKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* indices = ctx->Tensor4ArgNameAndIndex("indices", 0);
    user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    auto params = ConstructNdIndicesSliceParams<T, I>(x, y, indices);
    NdIndicesSliceUtil<DeviceType::kGPU, T, I>::GatherNd(ctx->device_ctx(), &params);
  }
};

template<typename T, typename I>
class ScatterNdUpdateGpuKernel final : public user_op::OpKernel {
 public:
  ScatterNdUpdateGpuKernel(const user_op::KernelInitContext& ctx) : user_op::OpKernel(ctx) {}
  ScatterNdUpdateGpuKernel() = default;
  ~ScatterNdUpdateGpuKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* indices = ctx->Tensor4ArgNameAndIndex("indices", 0);
    user_op::Tensor* updates = ctx->Tensor4ArgNameAndIndex("updates", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    if (in->dptr<T>() != out->mut_dptr<T>()) {
      size_t out_bytes_size = out->shape().elem_cnt() * GetSizeOfDataType(out->data_type());
      Memcpy<DeviceType::kGPU>(ctx->device_ctx(), out->mut_dptr<T>(), in->dptr<T>(),
                               out_bytes_size);
    }
    auto params = ConstructNdIndicesSliceParams<T, I>(out, updates, indices);
    NdIndicesSliceUtil<DeviceType::kGPU, T, I>::ScatterNdUpdate(ctx->device_ctx(), &params);
  }
};

template<typename T, typename I>
class ScatterNdAddGpuKernel final : public user_op::OpKernel {
 public:
  ScatterNdAddGpuKernel(const user_op::KernelInitContext& ctx) : user_op::OpKernel(ctx) {}
  ScatterNdAddGpuKernel() = default;
  ~ScatterNdAddGpuKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* indices = ctx->Tensor4ArgNameAndIndex("indices", 0);
    user_op::Tensor* updates = ctx->Tensor4ArgNameAndIndex("updates", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    if (in->dptr<T>() != out->mut_dptr<T>()) {
      size_t out_bytes_size = out->shape().elem_cnt() * GetSizeOfDataType(out->data_type());
      Memcpy<DeviceType::kGPU>(ctx->device_ctx(), out->mut_dptr<T>(), in->dptr<T>(),
                               out_bytes_size);
    }
    auto params = ConstructNdIndicesSliceParams<T, I>(out, updates, indices);
    NdIndicesSliceUtil<DeviceType::kGPU, T, I>::ScatterNdAdd(ctx->device_ctx(), &params);
  }
};

#define REGISTER_SCATTER_ND_OPT_GPU_KERNELS(opt, opt_name, dtype, itype)                        \
  REGISTER_USER_KERNEL(#opt_name)                                                               \
      .SetCreateFn([](const oneflow::user_op::KernelInitContext& ctx) {                         \
        return new ScatterNd##opt##GpuKernel<dtype, itype>(ctx);                                \
      })                                                                                        \
      .SetIsMatchedPred([](const oneflow::user_op::KernelRegContext& ctx) {                     \
        const user_op::TensorDesc* indices_desc = ctx.TensorDesc4ArgNameAndIndex("indices", 0); \
        const user_op::TensorDesc* out_desc = ctx.TensorDesc4ArgNameAndIndex("out", 0);         \
        if (ctx.device() == DeviceType::kGPU                                                    \
            && indices_desc->data_type() == GetDataType<itype>::value                           \
            && out_desc->data_type() == GetDataType<dtype>::value) {                            \
          return true;                                                                          \
        }                                                                                       \
        return false;                                                                           \
      })                                                                                        \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));                       \
        return Maybe<void>::Ok();                                                               \
      });

#define REGISTER_GATHER_SCATTER_ND_GPU_KERNELS(dtype, itype)                                    \
  REGISTER_USER_KERNEL("gather_nd")                                                             \
      .SetCreateFn([](const oneflow::user_op::KernelInitContext& ctx) {                         \
        return new GatherNdGpuKernel<dtype, itype>(ctx);                                        \
      })                                                                                        \
      .SetIsMatchedPred([](const oneflow::user_op::KernelRegContext& ctx) {                     \
        const user_op::TensorDesc* indices_desc = ctx.TensorDesc4ArgNameAndIndex("indices", 0); \
        const user_op::TensorDesc* y_desc = ctx.TensorDesc4ArgNameAndIndex("y", 0);             \
        if (ctx.device() == DeviceType::kGPU                                                    \
            && indices_desc->data_type() == GetDataType<itype>::value                           \
            && y_desc->data_type() == GetDataType<dtype>::value) {                              \
          return true;                                                                          \
        }                                                                                       \
        return false;                                                                           \
      });                                                                                       \
  REGISTER_SCATTER_ND_OPT_GPU_KERNELS(Update, scatter_nd_update, dtype, itype)                  \
  REGISTER_SCATTER_ND_OPT_GPU_KERNELS(Add, scatter_nd_add, dtype, itype)

REGISTER_GATHER_SCATTER_ND_GPU_KERNELS(float, int32_t)

}  // namespace oneflow
