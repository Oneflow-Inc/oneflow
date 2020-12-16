#ifndef ONEFLOW_USER_KERNELS_OFFSET_TO_INDEX_UTIL_H_
#define ONEFLOW_USER_KERNELS_OFFSET_TO_INDEX_UTIL_H_
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ndarray/xpu_util.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow{

#define OFFSET_TO_NDINDEX_DATA_TYPE_SEQ \
  OF_PP_MAKE_TUPLE_SEQ(int32_t, DataType::kInt32) \
  OF_PP_MAKE_TUPLE_SEQ(int64_t, DataType::kInt64)

const int index_max_ndims = 6; 

template<typename IDX_T>
using IndexHelper = NdIndexOffsetHelper<IDX_T, index_max_ndims>;

namespace user_op{
template<DeviceType device_type, typename T>
struct OffsetToNdIndexFunctor final {
    void operator()(DeviceCtx* ctx, int32_t in_num, 
                    int32_t ndim, const T* index, T* dims_tensor, T* out);
};

template<typename T>
OF_DEVICE_FUNC void DoOffsetToIndex(int32_t in_num, 
                    int32_t ndim, const T* index, T* dims, T* out){
    IndexHelper<T> helper(dims, ndim);
    std::cout<<"Ndim is: "<<ndim<<std::endl; 
    std::cout<<"index(offset) is: "<<index<<std::endl;
    int offset = *index;
    helper.OffsetToNdIndex(offset, dims);
    for(int i = 0; i < ndim; i++){
        out[i] = dims[i];
    }
}

# define INSTANTIATE_OFFSET_TO_NDINDEX__FUNCTOR(device_type_v, dtype_pair) \ 
    template struct OffsetToNdIndexFunctor<device_type_v, OF_PP_PAIR_FIRST(dtype_pair)>;


} // namespace user_op

} // namespace oneflow

# endif // ONEFLOW_USER_KERNELS_OFFSET_TO_INDEX_UTIL_H_

