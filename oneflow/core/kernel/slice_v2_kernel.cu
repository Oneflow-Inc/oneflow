#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void SliceV2ForwardGpu(const int64_t n, const int64_t* offset, const T* entire,
                                  T* slice) {
  CUDA_1D_KERNEL_LOOP(i, n) { slice[i] = entire[offset[i]]; }
}

}  // namespace

template<typename T>
class SliceV2GpuKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SliceV2GpuKernel);
  SliceV2GpuKernel() = default;
  ~SliceV2GpuKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    const Blob* in_blob = BnInOp2Blob("in");
    Blob* offset_blob = BnInOp2Blob("out_to_in_offset");
    Blob* out_blob = BnInOp2Blob("out");

    const SliceV2OpConf& conf = this->op_conf().slice_v2_conf();
    WithHostBlobAndStreamSynchronizeEnv(ctx.device_ctx, offset_blob, [&](Blob* host_blob) {
      int64_t* host_blob_ptr = host_blob->mut_dptr<int64_t>();
      FOR_RANGE(int64_t, i, 0, host_blob->shape().elem_cnt()) {
        int64_t offset = 0;
        int64_t index = i;
        FOR_RANGE(int64_t, j, 0, host_blob->shape().NumAxes()) {
          const DimSliceConf& dim_slice_conf = conf.dim_slice_conf(j);
          const int64_t dim_len = in_blob->shape().At(j);
          const int64_t dim_elem_cnt = host_blob->shape().Count(j + 1);
          const int64_t dim_i = index / dim_elem_cnt;
          index = index % dim_elem_cnt;
          int64_t start = dim_slice_conf.has_start() ? dim_slice_conf.start() : 0;
          if (start < 0) { start += dim_len; }
          CHECK_GE(start, 0);
          CHECK_LT(start, dim_len);
          int64_t stride = dim_slice_conf.stride();
          CHECK_GT(stride, 0);
          offset += (start + dim_i * stride) * in_blob->shape().Count(j + 1);
        }
        host_blob_ptr[i] = offset;
      }
    });
    const int64_t num_output = out_blob->shape().elem_cnt();
    SliceV2ForwardGpu<T><<<BlocksNum4ThreadsNum(num_output), kCudaThreadsNumPerBlock, 0,
                           ctx.device_ctx->cuda_stream()>>>(
        num_output, offset_blob->dptr<int64_t>(), in_blob->dptr<T>(), out_blob->mut_dptr<T>());
  }
};

#define REGISTER_SLICE_V2_GPU_KERNEL(dtype)                                                  \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kSliceV2Conf, DeviceType::kGPU, dtype, \
                                        SliceV2GpuKernel<dtype>)

REGISTER_SLICE_V2_GPU_KERNEL(float);
REGISTER_SLICE_V2_GPU_KERNEL(double);
REGISTER_SLICE_V2_GPU_KERNEL(int8_t);
REGISTER_SLICE_V2_GPU_KERNEL(int32_t);
REGISTER_SLICE_V2_GPU_KERNEL(int64_t);

}  // namespace oneflow
