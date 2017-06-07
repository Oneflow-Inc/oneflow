#ifndef ONEFLOW_CORE_THREAD_THREAD_CONTEXT_H_
#define ONEFLOW_CORE_THREAD_THREAD_CONTEXT_H_

namespace oneflow {

struct ThreadContext {
  ThreadContext() : copy_hd_cuda_stream(nullptr),
                    compute_cuda_stream(nullptr) {}
  cudaStream_t* copy_hd_cuda_stream;
  cudaStream_t* compute_cuda_stream;
};

} // namespace oneflow

#endif // ONEFLOW_CORE_THREAD_THREAD_CONTEXT_H_
