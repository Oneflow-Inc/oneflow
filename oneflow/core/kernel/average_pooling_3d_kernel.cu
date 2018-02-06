#include "oneflow/core/device/cudnn_util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/average_pooling_3d_kernel.h"

namespace oneflow {

namespace {

std::vector<int> Int64VecToIntVec(const std::vector<int64_t>& vec) {
  std::vector<int> ret;
  for (auto item : vec) { ret.push_back(static_cast<int>(item)); }
  return ret;
}

}  // namespace

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
    std::vector<int> in_dim = Int64VecToIntVec(in_blob->shape().dim_vec());
    std::vector<int> out_dim = Int64VecToIntVec(out_blob->shape().dim_vec());
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

  static void Backward(const KernelCtx& ctx, const Blob* out_diff_blob,
                       Blob* in_diff_blob, const Pooling3DCtx& pooling_ctx) {
    TODO();
  }
};

#define INSTANTIATE_AVERAGE_POOLING_3D_KERNEL_UTIL(type_cpp, type_proto) \
  template class AveragePooling3DKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_AVERAGE_POOLING_3D_KERNEL_UTIL,
                     ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
