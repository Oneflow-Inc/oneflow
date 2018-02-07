#ifndef ONEFLOW_CORE_EIGEN_TENSOR_TYPE_H_
#define ONEFLOW_CORE_EIGEN_TENSOR_TYPE_H_

#include "unsupported/Eigen/CXX11/Tensor"

namespace oneflow {

template<typename T, int NDIMS = 1, typename IndexType = Eigen::DenseIndex>
using Tensor =
    Eigen::TensorMap<Eigen::Tensor<T, NDIMS, Eigen::RowMajor, IndexType>,
                     Eigen::Aligned>;

template<typename T, int NDIMS = 1, typename IndexType = Eigen::DenseIndex>
using ConstTensor =
    Eigen::TensorMap<Eigen::Tensor<const T, NDIMS, Eigen::RowMajor, IndexType>,
                     Eigen::Aligned>;

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EIGEN_TENSOR_TYPE_H_
