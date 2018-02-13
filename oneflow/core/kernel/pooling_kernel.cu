#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/pooling_kernel.h"

namespace oneflow {

template<typename T>
class Pooling3DKernelUtil<DeviceType::kGPU, T> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Pooling3DKernelUtil);
  Pooling3DKernelUtil() = delete;

  static void Forward(const KernelCtx& kernel_ctx, const Blob* in_blob,
                      Blob* out_blob, const Pooling3DCtx& pooling_ctx) {
    double alpha = 1.0;
    double beta = 0.0;
    CHECK_EQ(cudnnPoolingForward(
                 kernel_ctx.device_ctx->cudnn_handle(),
                 pooling_ctx.pooling_desc_ptr()->Get(), &alpha,
                 pooling_ctx.in_desc_ptr()->Get(), in_blob->dptr(), &beta,
                 pooling_ctx.out_desc_ptr()->Get(), out_blob->mut_dptr()),
             CUDNN_STATUS_SUCCESS);
  }

  static void Backward(const KernelCtx& kernel_ctx, const Blob* out_diff_blob,
                       const Blob* out_blob, const Blob* in_blob,
                       Blob* in_diff_blob, const Pooling3DCtx& pooling_ctx) {
    double alpha = 1.0;
    double beta = 0.0;
    CHECK_EQ(
        cudnnPoolingBackward(
            kernel_ctx.device_ctx->cudnn_handle(),
            pooling_ctx.pooling_desc_ptr()->Get(), &alpha,
            pooling_ctx.out_desc_ptr()->Get(), out_blob->dptr(),
            pooling_ctx.out_diff_desc_ptr()->Get(), out_diff_blob->dptr(),
            pooling_ctx.in_desc_ptr()->Get(), in_blob->dptr(), &beta,
            pooling_ctx.in_diff_desc_ptr()->Get(), in_diff_blob->mut_dptr()),
        CUDNN_STATUS_SUCCESS);
  }
};

#define INSTANTIATE_POOLING_3D_KERNEL_UTIL(type_cpp, type_proto) \
  template class Pooling3DKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_POOLING_3D_KERNEL_UTIL,
                     ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
