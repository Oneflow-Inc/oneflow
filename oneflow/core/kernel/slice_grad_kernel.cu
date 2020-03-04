#include "oneflow/core/kernel/slice_kernel.h"
#include "oneflow/core/kernel/slice_grad_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void SliceBackwardGpu(const int64_t n, const int64_t* offset, const T* slice,
                                 T* entire) {
  CUDA_1D_KERNEL_LOOP(i, n) { entire[offset[i]] = slice[i]; }
}

}  // namespace

template<typename T>
void SliceGradKernel<DeviceType::kGPU, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* dy_blob = BnInOp2Blob("dy");
  const Blob* offset_blob = BnInOp2Blob("y_to_x_offset");
  Blob* dx_blob = BnInOp2Blob("dx");
  const int64_t num_output = dy_blob->shape().elem_cnt();
  Memset<DeviceType::kGPU>(ctx.device_ctx, dx_blob->mut_dptr<T>(), 0,
                           dx_blob->ByteSizeOfBlobBody());
  SliceBackwardGpu<T><<<BlocksNum4ThreadsNum(num_output), kCudaThreadsNumPerBlock, 0,
                        ctx.device_ctx->cuda_stream()>>>(
      num_output, offset_blob->dptr<int64_t>(), dy_blob->dptr<T>(), dx_blob->mut_dptr<T>());
}

template<typename T>
void SliceGradKernel<DeviceType::kGPU, T>::InitConstBufBlobs(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Shape in_shape(this->kernel_conf().slice_conf().in_shape());
  InitOut2InOffsetFromHost(ctx, in_shape, BnInOp2Blob("y_to_x_offset"));
}

template<typename T>
void SliceGradKernel<DeviceType::kGPU, T>::InitOut2InOffsetFromHost(DeviceCtx* ctx,
                                                                    const Shape& in_shape,
                                                                    Blob* blob) const {
  const SliceGradOpConf& conf = op_conf().slice_grad_conf();
  WithHostBlobAndStreamSynchronizeEnv(ctx, blob, [&](Blob* host_blob) {
    int64_t* host_blob_ptr = host_blob->mut_dptr<int64_t>();
    FOR_RANGE(int64_t, i, 0, host_blob->shape().elem_cnt()) {
      int64_t offset = 0;
      int64_t index = i;
      FOR_RANGE(int64_t, j, 0, host_blob->shape().NumAxes()) {
        const int64_t dim_elem_cnt = host_blob->shape().Count(j + 1);
        const int64_t dim_i = index / dim_elem_cnt;
        index = index % dim_elem_cnt;
        int64_t start = 0;
        int64_t stride = 1;
        const DimSliceConf& dim_slice_conf = conf.dim_slice_conf(j);
        if (dim_slice_conf.has_start()) { start = dim_slice_conf.start(); }
        if (start < 0) { start += host_blob->shape().At(j); }
        stride = dim_slice_conf.stride();
        offset += (start + dim_i * stride) * in_shape.Count(j + 1);
      }
      host_blob_ptr[i] = offset;
    }
  });
}

#define INSTANTIATE_GPU_SLICE_KERNEL(type_cpp, type_proto) \
  template struct SliceGradKernel<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_GPU_SLICE_KERNEL, ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
