#ifndef ONEFLOW_CORE_EIGEN_EIGEN_TENSOR_INTERFACE_H_
#define ONEFLOW_CORE_EIGEN_EIGEN_TENSOR_INTERFACE_H_

#include "oneflow/core/eigen/tensor_type.h"

namespace oneflow {

class EigenTensorIf {
 public:
  EigenTensorIf() = default;
  virtual ~EigenTensorIf() = default;

  template<typename T>
  virtual EigenTensorIf& operator=(const T& rhs) = 0;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EIGEN_EIGEN_TENSOR_INTERFACE_H_
