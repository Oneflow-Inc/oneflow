#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/pooling_kernel.h"

namespace oneflow {

template<typename T>
void PoolingKernel<DeviceType::kGPU, T>::PoolingForward(const KernelCtx& kernel_ctx,
                                                        const PoolingCtx& pooling_ctx,
                                                        const Blob* in_blob, Blob* out_blob) const {
  if (this->GetPoolingKernelConf().need_infer_cudnn_desc_each_forward()) {
    const PoolingKernelConf& conf = this->GetPoolingKernelConf();
    const std::string& data_format = conf.data_format();
    CudnnTensorDesc in_desc(GetDataType<T>::value, in_blob->shape(), data_format);
    CudnnTensorDesc out_desc(GetDataType<T>::value, out_blob->shape(), data_format);

    const int dim = conf.dim();
    CHECK_GE(dim, 1);
    CHECK_LE(dim, 3);
    typedef fixed_vector<int, SHAPE_MAX_AXIS_SIZE> FixedVector;
    FixedVector pool_size(dim);
    FixedVector padding(dim);
    FixedVector strides(dim);
    FOR_RANGE(int, i, 0, dim) {
      int32_t index_in_3d = i + 3 - dim;
      pool_size[i] = conf.pool_size().Get(index_in_3d);
      padding[i] = std::max<int>(conf.padding_before().Get(index_in_3d),
                                 conf.padding_after().Get(index_in_3d));
      strides[i] = conf.strides().Get(index_in_3d);
    }
    CudnnPoolingDesc pooling_desc(this->GetCudnnPoolingMode(), dim, pool_size.data(),
                                  padding.data(), strides.data());

    CudaCheck(cudnnPoolingForward(kernel_ctx.device_ctx->cudnn_handle(), pooling_desc.Get(),
                                  CudnnSPOnePtr<T>(), in_desc.Get(), in_blob->dptr(),
                                  CudnnSPZeroPtr<T>(), out_desc.Get(), out_blob->mut_dptr()));
  } else {
    CudaCheck(cudnnPoolingForward(
        kernel_ctx.device_ctx->cudnn_handle(), pooling_ctx.cudnn_pooling_desc(), CudnnSPOnePtr<T>(),
        pooling_ctx.cudnn_in_tensor_desc(), in_blob->dptr(), CudnnSPZeroPtr<T>(),
        pooling_ctx.cudnn_out_tensor_desc(), out_blob->mut_dptr()));
  }
}

template<typename T>
void PoolingKernel<DeviceType::kGPU, T>::PoolingBackward(const KernelCtx& kernel_ctx,
                                                         const PoolingCtx& pooling_ctx,
                                                         const Blob* out_diff_blob,
                                                         const Blob* out_blob, const Blob* in_blob,
                                                         Blob* in_diff_blob) const {
  CudaCheck(cudnnPoolingBackward(
      kernel_ctx.device_ctx->cudnn_handle(), pooling_ctx.cudnn_pooling_desc(), CudnnSPOnePtr<T>(),
      pooling_ctx.cudnn_out_tensor_desc(), out_blob->dptr(), pooling_ctx.cudnn_out_tensor_desc(),
      out_diff_blob->dptr(), pooling_ctx.cudnn_in_tensor_desc(), in_blob->dptr(),
      CudnnSPZeroPtr<T>(), pooling_ctx.cudnn_in_tensor_desc(), in_diff_blob->mut_dptr()));
}

#define INSTANTIATE_POOLING_KERNEL(type_cpp, type_proto) \
  template class PoolingKernel<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_POOLING_KERNEL, FLOATING_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ)

}  // namespace oneflow
