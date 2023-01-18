#ifdef WITH_CUDA

#include "oneflow/core/ep/include/primitive/primitive.h"
#include "oneflow/core/ep/include/primitive/gather.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/common/data_type.h" 
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/core/device/cuda_util.h"
#include <cuda.h>

namespace oneflow {

namespace ep {

namespace primitive {

namespace gather {

namespace internal {

template<typename T, >

bool IsSafeUseIndex32(int64_t outer_dim_size, int64_t gather_dim_size, int64_t inner_dim_size, int64_t num_indices){
  const int64_t in_elem_cnt = outer_dim_size * gather_dim_size * inner_dim_size;
  const int64_t out_elem_cnt = outer_dim_size * num_indices * inner_dim_size;
  return std::max(out_elem_cnt, in_elem_cnt) < GetMaxVal<int32_t>()/2;
}

template<typename MovementType, typename IndexType>
void DispatchIndexSize(cudaStream_t cuda_stream, int64_t outer_dim_size, int64_t gather_dim_size, int64_t inner_dim_byte_size,
int64_t num_indices, int64_t offset, const IndexType *indices, const void *in, void *out){
    const out_elem_cnt = outer_dim_size * num_indices + inner_dim_size;
    if(IsSafeUseIndex32(outer_dim_size, gather_dim_size, inner_dim_size, num_indices)){
      NdIndexOffsetHelper<int32_t, 3> in_helper(outer_dim_size, gather_dim_size, inner_dim_size);
      NdIndexOffsetHelper<int32_t, 3> out_helper(outer_dim_size, num_indices, inner_dim_size);
      
    } else {
      NdIndexOffsetHelper<int64_t, 3> in_helper(outer_dim_size, gather_dim_size, inner_dim_size);
      NdIndexOffsetHelper<int64_t, 3> out_helper(outer_dim_size, num_indices, inner_dim_size);
    }
}

template<typename IndexType, typename MovementType>
bool TryDispatchMovementType(cudaStream_t cuda_stream, int64_t outer_dim_size, int64_t gather_dim_size, int64_t inner_dim_byte_size, int64_t num_indices, int64_t offset, const IndexType *indices, const void *in, void *out){
    #define isAligned(src, alignment) reinterpret_cast<uintptr_t>(src) % sizeof(alignment) == 0
    if(isAligned(in, MovementType) && isAligned(out, MovementType) && inner_dim_byte_size % sizeof(MovementType) == 0){
      
      return true;
    } else {
      return false;    
    }
    #undef isAligned
}

template<typename IndexType>
void DispatchMovementSize(
    cudaStream_t cuda_stream, 
    int64_t outer_dim_size,
    int64_t gather_dim_size, 
    int64_t inner_dim_size,
    int64_t num_indices,
    int64_t offset,
    const IndexType *indices,
    const void *in,
    void *out){
  using Func = bool (*)(cudaStream_t cuda_stream, 
    int64_t outer_dim_size,
    int64_t gather_dim_size, 
    int64_t inner_dim_byte_size,
    int64_t num_indices,
    int64_t offset,
    const IndexType *indices,
    const void *in,
    void *out);
  
  Func func[] = {
    TryDispatchMovementType<IndexType, ulonglong2>, // 16-Bytes
    TryDispatchMovementType<IndexType, uint64_t>,   // 8-Bytes
    TryDispatchMovementType<IndexType, uint32_t>,   // 4-Bytes
    TryDispatchMovementType<IndexType, uint16_t>,   // 2-Bytes
    TryDispatchMovementType<IndexType, uint8_t>,    // 1-Bytes
  };

  for(size_t i=0; i < sizeof(func)/sizeof(func[0]); i++){
    if(func[i](cuda_stream, outer_dim_size, gather_dim_size, inner_dim_byte_size, num_indices, offset, indices, in, out)){
      break;
    }
  }
}

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
    const int64_t outer_dim_size,
    const int64_t gather_dim_size,
    const int64_t inner_dim_size,
    void *out,
    const int64_t offset
  ){
    cudaStream_t cuda_stream = stream->As<CudaStream>()->cuda_stream();
    DispatchMovementSize<IndexType>(cuda_stream, outer_dim_size, gather_dim_size, inner_dim_size*sizeof(T), num_indices, offset, indices, in, out); 
  } 
};

} // namespace internel

} // namespace gather

} // namespace primitive

} // namespace ep

} // namespace oneflow

#endif // WITH_CUDA


