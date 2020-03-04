// #include "oneflow/core/framework/framework.h"
// #include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/customized/kernels/nd_indices_slice_util.h"

namespace oneflow {

namespace {

template<typename T, typename I>
__global__ void CudaGatherNd(NdIndicesSliceParams<T, I> params) {
  GatherNdFunctor<T, I>::Invoke(params.num_segms * params.segm_size, params.segm_size,
                                params.segm_dim, params.shape, params.indices, params.dense,
                                params.sparse);
}

}  // namespace

template<typename T, typename I>
struct GatherNdOnDevice<DeviceType::kGPU, T, I> {
  static void Run(DeviceCtx* ctx, NdIndicesSliceParams<T, I>* params) {
    RUN_CUDA_KERNEL((CudaGatherNd<T, I>), ctx, params->num_segms * params->segm_size, *params);
  }
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

}  // namespace oneflow
