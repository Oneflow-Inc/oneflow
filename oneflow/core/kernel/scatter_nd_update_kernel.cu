#include "oneflow/core/kernel/nd_indices_util.h"
// #include "kernel.h"

namespace oneflow {

namespace {

template<typename T, typename I, template<typename> class func>
__global__ void ScatterNdUpdateGpu(int64_t num_segms, int64_t segms_size, int64_t segm_dims,
                                   const I* indices, const int64_t* shape, const T* sparse,
                                   T* dense) {
  ScatterNdFunctor<T, I, func>::Invoke(num_segms, segms_size, segm_dims, indices, shape, sparse,
                                       dense);
}

}  // namespace

template<typename T, typename I, template<typename> class func>
struct ScatterNdOnDevice<DeviceType::kGPU, T, I, func> {
  static void Run(DeviceCtx* ctx, int64_t num_segms, int64_t segms_size, int64_t segm_dims,
                  const I* indices, const int64_t* shape, const T* sparse, T* dense) {
    int64_t elem_cnt = num_segms * segms_size;
    RUN_CUDA_KERNEL((ScatterNdUpdateGpu<T, I, func>), ctx, elem_cnt, elem_cnt, segms_size,
                    segm_dims, indices, shape, sparse, dense);
  }
};

template<typename T, typename I>
class ScatterNdUpdateGpuKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ScatterNdUpdateGpuKernel);
  ScatterNdUpdateGpuKernel() = default;
  ~ScatterNdUpdateGpuKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* indices_blob = BnInOp2Blob("indices");
    const Blob* updates_blob = BnInOp2Blob("updates");
    Blob* shape_blob = BnInOp2Blob("shape");
    Blob* out_blob = BnInOp2Blob("out");

    out_blob->CopyDataContentFrom(ctx.device_ctx, BnInOp2Blob("in"));

    FOR_RANGE(size_t, i, 0, out_blob->shape().NumAxes()) {
      KernelUtil<DeviceType::kGPU, int64_t>::Set(ctx.device_ctx, out_blob->shape().At(i),
                                                 shape_blob->mut_dptr<int64_t>() + i);
    }

    NdIndicesUtil<DeviceType::kGPU, T, I>::ScatterNdUpdate(
        ctx.device_ctx, indices_blob, updates_blob, shape_blob->dptr<int64_t>(), out_blob);
  }
};

#define REGISTER_SCATTER_ND_UPDATE_GPU_KERNEL(dtype, itype)                                  \
  NEW_REGISTER_KERNEL(OperatorConf::kLocalScatterNdUpdateConf,                               \
                      ScatterNdUpdateGpuKernel<dtype, itype>)                                \
      .SetIsMatchedPred([](const KernelConf& conf) {                                         \
        return ((conf.op_attribute().op_conf().device_type() == DeviceType::kGPU)            \
                && (conf.data_type() == GetDataType<dtype>::value)                           \
                && (GetDataType<itype>::value == conf.scatter_nd_update_conf().idx_type())); \
      })

REGISTER_SCATTER_ND_UPDATE_GPU_KERNEL(int32_t, int32_t);
REGISTER_SCATTER_ND_UPDATE_GPU_KERNEL(float, int32_t);
REGISTER_SCATTER_ND_UPDATE_GPU_KERNEL(double, int32_t);
REGISTER_SCATTER_ND_UPDATE_GPU_KERNEL(float, int8_t);
REGISTER_SCATTER_ND_UPDATE_GPU_KERNEL(double, int8_t);

}  // namespace oneflow
