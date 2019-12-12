#include "oneflow/core/kernel/nd_indices_util.h"

namespace oneflow {

namespace {

template<typename T, typename I>
__global__ void GatherNdGpu(int64_t num_segms, int64_t segms_size, int64_t segm_dims,
                            const I* indices, const int64_t* dense_shape, const T* dense,
                            T* sparse) {
  GatherNdFunctor<T, I>::Invoke(num_segms, segms_size, segm_dims, indices, dense_shape, dense,
                                sparse);
}

}  // namespace

template<typename T, typename I>
struct GatherNdOnDevice<DeviceType::kGPU, T, I> {
  static void Run(DeviceCtx* ctx, int64_t num_segms, int64_t segms_size, int64_t segm_dims,
                  const I* indices, const int64_t* dense_shape, const T* dense, T* sparse) {
    int64_t elem_cnt = num_segms * segms_size;
    RUN_CUDA_KERNEL((GatherNdGpu<T, I>), ctx, elem_cnt, elem_cnt, segms_size, segm_dims, indices,
                    dense_shape, dense, sparse);
  }
};

template<typename T, typename I>
class GatherNdGpuKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GatherNdGpuKernel);
  GatherNdGpuKernel() = default;
  ~GatherNdGpuKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("in");
    const Blob* indices_blob = BnInOp2Blob("indices");
    Blob* shape_blob = BnInOp2Blob("shape");
    Blob* out_blob = BnInOp2Blob("out");

    FOR_RANGE(size_t, i, 0, out_blob->shape().NumAxes()) {
      KernelUtil<DeviceType::kGPU, int64_t>::Set(ctx.device_ctx, in_blob->shape().At(i),
                                                 shape_blob->mut_dptr<int64_t>() + i);
    }

    NdIndicesUtil<DeviceType::kGPU, T, I>::GatherNd(ctx.device_ctx, indices_blob, in_blob,
                                                    shape_blob->dptr<int64_t>(), out_blob);
  }
};

#define REGISTER_GATHER_ND_GPU_KERNEL(dtype, itype)                                         \
  NEW_REGISTER_KERNEL(OperatorConf::kLocalGatherNdConf, GatherNdGpuKernel<dtype, itype>)    \
      .SetIsMatchedPred([](const KernelConf& conf) {                                        \
        return ((conf.op_attribute().op_conf().device_type() == DeviceType::kGPU)           \
                && (conf.data_type() == GetDataType<dtype>::value)                          \
                && (GetDataType<itype>::value == conf.gather_nd_update_conf().idx_type())); \
      })

REGISTER_GATHER_ND_GPU_KERNEL(float, int32_t);
REGISTER_GATHER_ND_GPU_KERNEL(double, int32_t);
REGISTER_GATHER_ND_GPU_KERNEL(float, int8_t);
REGISTER_GATHER_ND_GPU_KERNEL(double, int8_t);

}  // namespace oneflow
