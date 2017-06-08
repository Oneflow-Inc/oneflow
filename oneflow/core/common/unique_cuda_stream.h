#ifndef ONEFLOW_CORE_UNIQUE_CUDA_STREAM_H_
#define ONEFLOW_CORE_UNIQUE_CUDA_STREAM_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

class UniqueCudaStream final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UniqueCudaStream);

  UniqueCudaStream() {
    CHECK_EQ(cudaStreamCreate(&val_), 0);
  }

  ~UniqueCudaStream() {
    CHECK_EQ(cudaStreamDestroy(val_), 0);
  }

  const cudaStream_t* get() const { return &val_; }

 private:
  cudaStream_t val_;

};

} // namespace oneflow

#endif // ONEFLOW_CORE_UNIQUE_CUDA_STREAM_H_
