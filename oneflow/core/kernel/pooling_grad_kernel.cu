#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/pooling_grad_kernel.h"

namespace oneflow {

template<typename T>
struct PoolingGradKernelUtil<DeviceType::kGPU, T> final {
  static void Compute(DeviceCtx* ctx, const PoolingConf& pooling_conf, const Blob* dy_blob,
                      const Blob* y_blob, const Blob* x_blob, Blob* dx_blob) {
    cudnnPoolingMode_t pooling_mode;
    CudnnTensorDesc x_desc(x_blob->data_type(), x_blob->shape(), pooling_conf.data_format());
    CudnnTensorDesc y_desc(y_blob->data_type(), y_blob->shape(), pooling_conf.data_format());
    std::unique_ptr<CudnnPoolingDesc> pooling_desc;

    if (pooling_conf.pool_mode() == "avg") {
      pooling_mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
    } else if (pooling_conf.pool_mode() == "max") {
      pooling_mode = CUDNN_POOLING_MAX;
    }

    CudaCheck(cudnnPoolingBackward(ctx->cudnn_handle(), pooling_desc->Get(), OnePtr<T>::value,
                                   y_desc.Get(), y_blob->dptr(), y_desc.Get(), dy_blob->dptr(),
                                   x_desc.Get(), x_blob->dptr(), ZeroPtr<T>::value, x_desc.Get(),
                                   dx_blob->mut_dptr()));
  }
};

#define INSTANTIATE_POOLING_GRAD_KERNEL_UTIL(type_cpp, type_proto) \
  template struct PoolingGradKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_POOLING_GRAD_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
