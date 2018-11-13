#include "oneflow/core/kernel/slice_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void SliceForwardGpu(const int64_t n, const int64_t* offset, const T* entire, T* slice) {
  CUDA_1D_KERNEL_LOOP(i, n) { slice[i] = entire[offset[i]]; }
}

template<typename T>
__global__ void SliceBackwardGpu(const int64_t n, const int64_t* offset, const T* slice,
                                 T* entire) {
  CUDA_1D_KERNEL_LOOP(i, n) { entire[offset[i]] /= slice[i]; }
}

}  // namespace

template<typename T>
void SliceKernel<DeviceType::kGPU, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* offset_blob = BnInOp2Blob("out_to_in_offset");
  Blob* out_blob = BnInOp2Blob("out");
  const int64_t num_output = out_blob->shape().elem_cnt();
  SliceForwardGpu<T><<<BlocksNum4ThreadsNum(num_output), kCudaThreadsNumPerBlock, 0,
                       ctx.device_ctx->cuda_stream()>>>(
      num_output, offset_blob->dptr<int64_t>(), in_blob->dptr<T>(), out_blob->mut_dptr<T>());
}

template<typename T>
void SliceKernel<DeviceType::kGPU, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_diff_blob = BnInOp2Blob(GenDiffBn("out"));
  const Blob* offset_blob = BnInOp2Blob("out_to_in_offset");
  Blob* in_diff_blob = BnInOp2Blob(GenDiffBn("in"));
  const int64_t num_output = out_diff_blob->shape().elem_cnt();
  SliceBackwardGpu<T><<<BlocksNum4ThreadsNum(num_output), kCudaThreadsNumPerBlock, 0,
                        ctx.device_ctx->cuda_stream()>>>(num_output, offset_blob->dptr<int64_t>(),
                                                         out_diff_blob->dptr<T>(),
                                                         in_diff_blob->mut_dptr<T>());
}

template<typename T>
void SliceKernel<DeviceType::kGPU, T>::InitConstBufBlobs(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  InitOut2InOffsetFromHost(ctx, BnInOp2Blob("in"), BnInOp2Blob("out_to_in_offset"));
}

template<typename T>
void SliceKernel<DeviceType::kGPU, T>::InitOut2InOffsetFromHost(DeviceCtx* ctx, const Blob* in_blob,
                                                                Blob* blob) const {
  const SliceOpConf& conf = op_conf().slice_conf();
  BEFORE_CPU_INITIALIZE();
  int64_t* host_blob_ptr = host_blob->mut_dptr<int64_t>();
  FOR_RANGE(size_t, i, 0, host_blob->shape().elem_cnt()) {
    size_t offset = 0;
    FOR_RANGE(size_t, j, 0, host_blob->shape().NumAxes()) {
      const size_t dim_idx = i / host_blob->shape().Count(j + 1);
      const size_t dim_len = host_blob->shape().At(j);
      int64_t start = 0;
      int64_t end = dim_len;
      int64_t stride = 1;
      if (j > 0) {
        const DimSliceConf& dim_slice_conf = conf.dim_slice_conf(j - 1);
        if (dim_slice_conf.has_start()) { start = dim_slice_conf.start(); }
        if (dim_slice_conf.has_end()) { end = dim_slice_conf.end(); }
        stride = dim_slice_conf.stride();
      }
      if (start < 0) { start += dim_len; }
      if (end < 0) { end += dim_len; }
      offset += (start + dim_idx * stride) * in_blob->shape().Count(j + 1);
    }
    host_blob_ptr[i] = offset;
  }
  AFTER_CPU_INITIALIZE();
}

#define INSTANTIATE_GPU_SLICE_KERNEL(type_cpp, type_proto) \
  template struct SliceKernel<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GPU_SLICE_KERNEL, ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
