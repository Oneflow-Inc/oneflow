#ifdef WITH_CUDA

#include "oneflow/core/ep/include/primitive/primitive.h"
#include "oneflow/core/ep/include/primitive/gather.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include <cuda.h>

namespace oneflow {

namespace ep {

namespace primitive {

namespace gather {

namespace internal {

template<typename IndexType>
void DispatchMovementSize(cudaStream_t cuda_stream, )

template<typename T, typename IndexType>
class GatherImpl : public Gather {
 public:
  OF_DISTLLOW_CPOY_AND_MOVE(GatherImpl);
  GatherImpl() = default;
  ~GatherImpl() override = default;
  
  using Gather::Launch;
  void Launch(Stream *stream, 
    Stream *stream,
    const void *indices,
    int64_t num_indices,
    const void *in,
    const int64_t batch_dim_size,
    const int64_t outer_dim_size,
    const int64_t gather_dim_size,
    const int64_t inner_dim_size,
    void *out,
    const int64_t offset
  ){
    cudaStream_t cuda_stream = stream->As<CudaStream>()->cuda_stream();
    
  } 
};

  

} // namespace internel

} // namespace gather

} // namespace primitive

} // namespace ep

} // namespace oneflow

#endif // WITH_CUDA


