#include "oneflow/core/ep/include/primitive/primitive.h"
#include "oneflow/core/ep/include/primitive/gather.h"
#include "oneflow/core/ep/cuda/primitive/type_seq.h"
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

template<typename PackedType, typename IndexType, typename IDX>
__global__ void GatherForwardGpu(
  const IDX elem_cnt, NdIndexOffsetHelper<IDX, 3> in_helper, NdIndexOffsetHelper<IDX, 3> out_helper, const IndexType* indices, const PackedType* in, const IDX gather_dim_size, PackedType* out, const IDX offset){
    IDX index[3];
    CUDA_1D_KERNEL_LOOP_T(IDX,i,elem_cnt){
        out_helper.OffsetToNdIndex(i, index);
        index[1] = indices[index[1]] - offset;
        PackedType v{};
        if(index[1] >= 0 && index[1] < gather_dim_size){
            v = in[in_helper.NdIndexToOffset(index)];
        }
        out[i] = v;
    }
}

bool IsSafeUseIndex32(int64_t outer_dim_size, int64_t gather_dim_size, int64_t inner_dim_size, int64_t num_indices){
  const int64_t in_elem_cnt = outer_dim_size * gather_dim_size * inner_dim_size;
  const int64_t out_elem_cnt = outer_dim_size * num_indices * inner_dim_size;
  return std::max(out_elem_cnt, in_elem_cnt) < GetMaxVal<int32_t>()/2;
}

template<typename PackedType, typename IndexType>
void DispatchIndexSize(cudaStream_t cuda_stream, int64_t outer_dim_size, int64_t gather_dim_size, int64_t inner_dim_size,
int64_t num_indices, int64_t offset, const IndexType *indices, const PackedType *in, PackedType *out){
    const int64_t out_elem_cnt = outer_dim_size * num_indices + inner_dim_size;
    if(IsSafeUseIndex32(outer_dim_size, gather_dim_size, inner_dim_size, num_indices)){
      NdIndexOffsetHelper<int32_t, 3> in_helper(outer_dim_size, gather_dim_size, inner_dim_size);
      NdIndexOffsetHelper<int32_t, 3> out_helper(outer_dim_size, num_indices, inner_dim_size);
      GatherForwardGpu<PackedType, IndexType, int32_t><<<BlocksNum4ThreadsNum(out_elem_cnt), kCudaThreadsNumPerBlock, 0, cuda_stream>>>(out_elem_cnt, in_helper, out_helper, indices, in, gather_dim_size, out, offset);
    } else {
      NdIndexOffsetHelper<int64_t, 3> in_helper(outer_dim_size, gather_dim_size, inner_dim_size);
      NdIndexOffsetHelper<int64_t, 3> out_helper(outer_dim_size, num_indices, inner_dim_size);
      GatherForwardGpu<PackedType, IndexType, int64_t><<<BlocksNum4ThreadsNum(out_elem_cnt), kCudaThreadsNumPerBlock, 0, cuda_stream>>>(out_elem_cnt, in_helper, out_helper, indices, in, gather_dim_size, out, offset);
    }
}

template<typename IndexType, typename PackedType>
bool TryDispatchPackedType(cudaStream_t cuda_stream, int64_t outer_dim_size, int64_t gather_dim_size, int64_t inner_dim_byte_size, int64_t num_indices, int64_t offset, const IndexType *indices, const void *in, void *out){
    #define isAligned(src, alignment) reinterpret_cast<uintptr_t>(src) % sizeof(alignment) == 0
    if(isAligned(in, PackedType) && 
       isAligned(out, PackedType) && 
       inner_dim_byte_size % sizeof(PackedType) == 0
    ){
      DispatchIndexSize<PackedType, IndexType>(cuda_stream, outer_dim_size, gather_dim_size, inner_dim_byte_size / sizeof(PackedType), num_indices, offset, indices, static_cast<const PackedType*>(in), static_cast<PackedType*>(out)); 
      return true;
    } else {
      return false;    
    }
    #undef isAligned
}

template<typename IndexType>
void DispatchPackedSize(cudaStream_t cuda_stream, int64_t outer_dim_size, int64_t gather_dim_size, 
    int64_t inner_dim_byte_size, int64_t num_indices, int64_t offset, const IndexType *indices,
    const void *in, void *out){
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
    TryDispatchPackedType<IndexType, ulonglong2>, // 16-Bytes
    TryDispatchPackedType<IndexType, uint64_t>,   // 8-Bytes
    TryDispatchPackedType<IndexType, uint32_t>,   // 4-Bytes
    TryDispatchPackedType<IndexType, uint16_t>,   // 2-Bytes
    TryDispatchPackedType<IndexType, uint8_t>,    // 1-Bytes
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
  OF_DISALLOW_COPY_AND_MOVE(GatherImpl);
  GatherImpl() = default;
  ~GatherImpl() override = default;
  
  using Gather::Launch;
  void Launch(Stream *stream,
    const void *indices,
    int64_t num_indices,
    const void* in,
    const int64_t outer_dim_size,
    const int64_t gather_dim_size,
    const int64_t inner_dim_size,
    void* out,
    const int64_t offset
  ) override {
    cudaStream_t cuda_stream = stream->As<CudaStream>()->cuda_stream();
    DispatchPackedSize<IndexType>(cuda_stream, outer_dim_size, gather_dim_size, inner_dim_size*sizeof(T), num_indices, offset, reinterpret_cast<const IndexType*>(indices), in, out); 
  } 
};

template<typename T, typename IndexType>
std::unique_ptr<Gather> NewGather() {
  return std::unique_ptr<Gather>(new GatherImpl<T, IndexType>());
}

class GatherFactoryImpl : public GatherFactory {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GatherFactoryImpl);
  GatherFactoryImpl() = default;
  ~GatherFactoryImpl() override = default;

  std::unique_ptr<Gather> New(DataType params_type, DataType indices_type) override {
#define MAKE_NEW_GATHER_ENTRY(params_type_pair, indices_type_pair) \
  {std::make_pair(OF_PP_PAIR_SECOND(params_type_pair), OF_PP_PAIR_SECOND(indices_type_pair)), \
   NewGather<OF_PP_PAIR_FIRST(params_type_pair), OF_PP_PAIR_FIRST(indices_type_pair)>},

    static const std::map<std::pair<DataType, DataType>, std::function<std::unique_ptr<Gather>()>>
      new_gather_handle{OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(
        MAKE_NEW_GATHER_ENTRY, CUDA_PRIMITIVE_ALL_TYPE_SEQ, CUDA_PRIMITIVE_INT_TYPE_SEQ)};

#undef MAKE_NEW_GATHER_ENTRY

    const auto it = new_gather_handle.find(std::make_pair(params_type, indices_type));
    if(it != new_gather_handle.end()) {
      return it->second();
    } else {
      return nullptr;
    }
  }
};

REGISTER_PRIMITIVE_FACTORY(DeviceType::kCUDA, GatherFactory, GatherFactoryImpl);

} // namespace internel

} // namespace gather

} // namespace primitive

} // namespace ep

} // namespace oneflow