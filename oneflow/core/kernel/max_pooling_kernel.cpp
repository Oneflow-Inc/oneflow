#include "oneflow/core/kernel/max_pooling_kernel.h"

namespace oneflow {

template<typename T>
T MaxPoolingKernel<DeviceType::kCPU, T>::ForwardInitialize() const {
  return -std::numeric_limits<T>::max();
}

template<typename T>
void MaxPoolingKernel<DeviceType::kCPU, T>::NCDHWProcess(const T& lhs,
                                                         T& rhs) const {
  if (lhs > rhs) { rhs = lhs; }
}

template<typename T>
void MaxPoolingKernel<DeviceType::kCPU, T>::NDHWCProcess(
    const int64_t in_col, const int64_t out_col, ConstEigenMatrixMap<T>& in_mat,
    EigenMatrixMap<T>& out_mat) const {
  out_mat.col(out_col) = out_mat.col(out_col).cwiseMax(in_mat.col(in_col));
}

template<typename T>
void MaxPoolingKernel<DeviceType::kCPU, T>::NCDHWProcessGrad(const T& in,
                                                             const T& out,
                                                             const T& out_diff,
                                                             const int64_t size,
                                                             T& in_diff) const {
  if (in == out) { in_diff += out_diff; }
}

template<typename T>
void MaxPoolingKernel<DeviceType::kCPU, T>::NDHWCProcessGrad(
    const int64_t out_col, const int64_t in_col, const int64_t size,
    ConstEigenArrayMap<T>& out_arr, ConstEigenArrayMap<T>& in_arr,
    ConstEigenArrayMap<T>& out_diff_arr, EigenArrayMap<T>& in_diff_arr) const {
  in_diff_arr.col(in_col) += out_diff_arr.col(out_col)
                             * (in_arr.col(in_col)
                                    .cwiseEqual(out_arr.col(out_col))
                                    .template cast<T>());
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kMaxPooling1DConf, MaxPoolingKernel,
                           ARITHMETIC_DATA_TYPE_SEQ);

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kMaxPooling2DConf, MaxPoolingKernel,
                           ARITHMETIC_DATA_TYPE_SEQ);

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kMaxPooling3DConf, MaxPoolingKernel,
                           ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
