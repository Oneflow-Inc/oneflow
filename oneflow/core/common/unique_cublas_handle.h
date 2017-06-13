#ifndef ONEFLOW_CORE_UNIQUE_CUBLAS_HANDLE_H_
#define ONEFLOW_CORE_UNIQUE_CUBLAS_HANDLE_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

class UniqueCublasHandle final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UniqueCublasHandle);
  UniqueCublasHandle() = delete;
  
  UniqueCublasHandle(const cudaStream_t* cuda_stream) {
    CHECK_EQ(cublasCreate(&handle_), CUBLAS_STATUS_SUCCESS);
    CHECK_EQ(cublasSetStream(handle_, *cuda_stream), CUBLAS_STATUS_SUCCESS);
  }

  ~UniqueCublasHandle() {
    CHECK_EQ(cublasDestroy(handle_), CUBLAS_STATUS_SUCCESS);
  }

  const cublasHandle_t* get() const { return &handle_; }

 private:
  cublasHandle_t handle_;

};

} // namespace oneflow

#endif // ONEFLOW_CORE_UNIQUE_CUBLAS_HANDLE_H_
