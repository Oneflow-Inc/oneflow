#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/nonzero_kernel_util.cuh"
#include <cub/cub.cuh>

namespace oneflow {

namespace {

template<typename T>
struct IsNonzero {
  bool operator()(const T& x) const { return (x != static_cast<T>(0)); }
};

}  // namespace

template<typename T>
cudaError_t CubReduceCount(void* tmp, size_t& tmp_bytes, const T* in, int32_t* out, int num_items,
                           cudaStream_t stream) {
  IsNonzero<T> is_nonzero;
  cub::TransformInputIterator<bool, IsNonzero<T>, const T*> in_iter(in, is_nonzero);
  return cub::DeviceReduce::Sum(tmp, tmp_bytes, in_iter, out, num_items, stream, false);
  return cudaSuccess;
}

template<typename T>
class NonzeroGpuKernel : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NonzeroGpuKernel);
  NonzeroGpuKernel() = default;
  ~NonzeroGpuKernel() = default;

 private:
  void ForwardDenseShape(const KernelCtx& ctx,
                         std::function<Blob*(const std::string&)> BnInOp2Blob) const override {}
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("in");

    Blob* num_nonzero_blob = BnInOp2Blob("num_nonzero");
    Blob* nnz_tmp_blob = BnInOp2Blob("nnz_tmp");
    size_t tmp_bytes = nnz_tmp_blob->shape().elem_cnt();
    CudaCheck(CubReduceCount<T>(nnz_tmp_blob->mut_dptr(), tmp_bytes, in_blob->dptr<T>(),
                                num_nonzero_blob->mut_dptr<int32_t>(), in_blob->shape().elem_cnt(),
                                ctx.device_ctx->cuda_stream()));

    Blob* out_blob = BnInOp2Blob("out");
  }
};

#define REGISTER_NONZERO_GPU_KERNEL(dtype)                                                        \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kLocalNonzeroConf, DeviceType::kGPU, dtype, \
                                        NonzeroGpuKernel<dtype>)

REGISTER_NONZERO_GPU_KERNEL(float);
REGISTER_NONZERO_GPU_KERNEL(double);
REGISTER_NONZERO_GPU_KERNEL(int32_t);
REGISTER_NONZERO_GPU_KERNEL(int64_t);

}  // namespace oneflow
