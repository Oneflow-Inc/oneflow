#include "oneflow/core/device/cudnn_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/average_pooling_3d_kernel.h"

namespace oneflow {

template<typename T>
class AveragePooling3DKernelUtil<DeviceType::kGPU, T> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AveragePooling3DKernelUtil);
  AveragePooling3DKernelUtil() = delete;

  static void Forward(const KernelCtx& kernel_ctx, const Blob* in_blob,
                      Blob* out_blob, const Pooling3DCtx& pooling_ctx) {
    std::vector<int> window{pooling_ctx.pool_size_d, pooling_ctx.pool_size_h,
                            pooling_ctx.pool_size_w};
    std::vector<int> padding{pooling_ctx.padding_d, pooling_ctx.padding_h,
                             pooling_ctx.padding_w};
    std::vector<int> stride{pooling_ctx.strides_d, pooling_ctx.strides_h,
                            pooling_ctx.strides_w};
    std::vector<int> in_dim{pooling_ctx.in_n, pooling_ctx.in_c,
                            pooling_ctx.in_d, pooling_ctx.in_h,
                            pooling_ctx.in_w};
    std::vector<int> out_dim{pooling_ctx.out_n, pooling_ctx.out_c,
                             pooling_ctx.out_d, pooling_ctx.out_h,
                             pooling_ctx.out_w};
    CudnnPoolingNdDesc pooling_desc(PoolingMode ::kAveragePooling, window,
                                    padding, stride);
    CudnnTensorNdDesc in_tensor_desc(GetDataType<T>::val, in_dim, stride);
    CudnnTensorNdDesc out_tensor_desc(GetDataType<T>::val, out_dim, stride);
    double alpha = 1.0;
    double beta = 0.0;
    CHECK_EQ(cudnnPoolingForward(kernel_ctx.device_ctx->cudnn_handle(),
                                 pooling_desc.Get(), &alpha,
                                 in_tensor_desc.Get(), in_blob->dptr(), &beta,
                                 out_tensor_desc.Get(), out_blob->mut_dptr()),
             CUDNN_STATUS_SUCCESS);
  }

  static void Backward(const KernelCtx& kernel_ctx, const Blob* out_diff_blob,
                       Blob* in_diff_blob, const Pooling3DCtx& pooling_ctx) {
    std::vector<int> window{pooling_ctx.pool_size_d, pooling_ctx.pool_size_h,
                            pooling_ctx.pool_size_w};
    std::vector<int> padding{pooling_ctx.padding_d, pooling_ctx.padding_h,
                             pooling_ctx.padding_w};
    std::vector<int> stride{pooling_ctx.strides_d, pooling_ctx.strides_h,
                            pooling_ctx.strides_w};
    std::vector<int> in_diff_dim{pooling_ctx.in_n, pooling_ctx.in_c,
                                 pooling_ctx.in_d, pooling_ctx.in_h,
                                 pooling_ctx.in_w};
    std::vector<int> out_diff_dim{pooling_ctx.out_n, pooling_ctx.out_c,
                                  pooling_ctx.out_d, pooling_ctx.out_h,
                                  pooling_ctx.out_w};
    CudnnPoolingNdDesc pooling_desc(PoolingMode ::kAveragePooling, window,
                                    padding, stride);
    CudnnTensorNdDesc in_diff_tensor_desc(GetDataType<T>::val, in_diff_dim,
                                          stride);
    CudnnTensorNdDesc out_diff_tensor_desc(GetDataType<T>::val, out_diff_dim,
                                           stride);
    cudnnTensorDescriptor_t desc_unused;
    double alpha = 1.0;
    double beta = 0.0;
    CHECK_EQ(cudnnPoolingBackward(
                 kernel_ctx.device_ctx->cudnn_handle(), pooling_desc.Get(),
                 &alpha, desc_unused, nullptr, out_diff_tensor_desc.Get(),
                 out_diff_blob->dptr(), desc_unused, nullptr, &beta,
                 in_diff_tensor_desc.Get(), in_diff_blob->mut_dptr()),
             CUDNN_STATUS_SUCCESS);
  }
};

#define INSTANTIATE_AVERAGE_POOLING_3D_KERNEL_UTIL(type_cpp, type_proto) \
  template class AveragePooling3DKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_AVERAGE_POOLING_3D_KERNEL_UTIL,
                     ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
