#ifndef ONEFLOW_CORE_PRIMITIVE_COMMON_PAD_H_
#define ONEFLOW_CORE_PRIMITIVE_COMMON_PAD_H_

#include "oneflow/core/ep/include/primitive/primitive.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow{ 

namespace ep{

namespace primitive{ 

template<size_t num_dims, typename IndexType>
struct PadParams{
    NdIndexOffsetHelper<IndexType, num_dims> src_index_helper;
    NdIndexOffsetHelper<IndexType, num_dims> dst_index_helper;
    IndexType padding_before[num_dims];
    IndexType padding_after[num_dims];
    IndexType out_size[num_dims];
    IndexType elem_cnt{};
    const void* src{};
    void* dst{};
}; 

template<size_t num_dims, typename IndexType, typename T>
void LaunchKernel(Stream* stream, PadParams<num_dims, IndexType> params, T pad_val); 

template<size_t num_dims, typename IndexType, typename T>
void LaunchKernel(Stream* stream, void* dst,
                  const int64_t* dst_dims, const void* src,
                  const int64_t* src_dims, const int64_t* padding_before,
                  const int64_t* padding_after, T pad_val){
    PadParams<num_dims, IndexType> params; 
    params.dst_index_helper = NdIndexOffsetHelper<IndexType, num_dims>(dst_dims);
    params.src_index_helper = NdIndexOffsetHelper<IndexType, num_dims>(src_dims);
    params.dst = dst; 
    params.src = src; 
    size_t elem_cnt = 1; 
    for(int i = 0; i < num_dims; i++){
        params.padding_before[i] = padding_before[i];
        params.padding_after[i] = padding_after[i];
        params.out_size[i] = dst_dims[i];
        elem_cnt *= params.out_size[i]; 
    }
    params.elem_cnt = elem_cnt; 
    LaunchKernel<num_dims, IndexType, T>(stream, params, pad_val);               
}



template<size_t num_dims, typename T>
void DispatchIndexType(Stream* stream, void* dst,
                        const int64_t* dst_dims, const void* src,
                        const int64_t* src_dims, const int64_t* padding_before,
                        const int64_t* padding_after, T pad_val) {
  size_t elem_cnt = 1;
  for (size_t i = 0; i < num_dims; ++i) { elem_cnt *= dst_dims[i]; }
  if (elem_cnt < GetMaxVal<int32_t>()) {
    LaunchKernel<num_dims, int32_t, T>(stream, dst, dst_dims, src, src_dims, 
                                    padding_before, padding_after, pad_val);
  } else {
    LaunchKernel<num_dims, int64_t, T>(stream, dst, dst_dims, src, src_dims, 
                                    padding_before, padding_after, pad_val);
  }
}

template<typename T>
void LaunchWithSimplified(Stream* stream, size_t num_dims, void* dst,
                        const int64_t* dst_dims, const void* src,
                        const int64_t* src_dims, const int64_t* padding_before,
                        const int64_t* padding_after, T pad_val) {
  void (*func)(Stream* /*stream*/, void* /*dst*/,
               const int64_t* /*dst_dims*/, const void* /*src*/,
               const int64_t* /*src_dims*/, const int64_t* /*padding_before*/, const int64_t* /*padding_after*/, T) =
      nullptr;
  if (num_dims == 1) {
    func = DispatchIndexType<1, T>;
  } else if (num_dims == 2) {
    func = DispatchIndexType<2, T>;
  } else if (num_dims == 3) {
    func = DispatchIndexType<3, T>;
  } else if (num_dims == 4) {
    func = DispatchIndexType<4, T>;
  } else if (num_dims == 5) {
    func = DispatchIndexType<5, T>;
  } else if (num_dims == 6) {
    func = DispatchIndexType<6, T>;
  } else if (num_dims == 7) {
    func = DispatchIndexType<7, T>;
  } else if (num_dims == 8) {
    func = DispatchIndexType<8, T>;
  } else {
    UNIMPLEMENTED();
  }
  func(stream, dst, dst_dims, src,
       src_dims, padding_before, padding_after, pad_val);
}


} // namespace primitive

} // namespace ep 

} // namespace oneflow 


#endif // ONEFLOW_CORE_PRIMITIVE_COMMON_PAD_H_
