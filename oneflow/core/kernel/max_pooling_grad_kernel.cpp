#include "oneflow/core/kernel/max_pooling_grad_kernel.h"

namespace oneflow {

template<typename T>
void MaxPoolingGradKernel<DeviceType::kCPU, T>::NCDHWProcessGrad(const T& in, const T& out,
                                                                 const T& out_diff,
                                                                 const int64_t size,
                                                                 T& in_diff) const {
  if (in == out) { in_diff += out_diff; }
}

template<typename T>
void MaxPoolingGradKernel<DeviceType::kCPU, T>::NDHWCProcessGrad(
    const int64_t out_col, const int64_t in_col, const int64_t size, ConstEigenArrayMap<T>& out_arr,
    ConstEigenArrayMap<T>& in_arr, ConstEigenArrayMap<T>& out_diff_arr,
    EigenArrayMap<T>& in_diff_arr) const {
  in_diff_arr.col(in_col) +=
      out_diff_arr.col(out_col)
      * (in_arr.col(in_col).cwiseEqual(out_arr.col(out_col)).template cast<T>());
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kMaxPooling1DGradConf, MaxPoolingGradKernel,
                           ARITHMETIC_DATA_TYPE_SEQ);

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kMaxPooling2DGradConf, MaxPoolingGradKernel,
                           ARITHMETIC_DATA_TYPE_SEQ);

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kMaxPooling3DGradConf, MaxPoolingGradKernel,
                           ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
