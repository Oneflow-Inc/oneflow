#ifndef ONEFLOW_USER_KERNELS_RAVEL_MULTI_INDEX_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_RAVEL_MULTI_INDEX_KERNEL_UTIL_H_
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/ndarray/xpu_util.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

#define RAVEL_MULTI_INDEX_DATA_TYPE_SEQ \
  INT_DATA_TYPE_SEQ

const int ravel_multi_index_max_ndims = 6;

template<typename IDX_T>
using RavelMultiIndexHelper = NdIndexOffsetHelper<IDX_T, ravel_multi_index_max_ndims>;

namespace user_op {
template<DeviceType device_type, typename T>
struct RavelMultiIndexFunctor final {
  void operator()(DeviceCtx* ctx, int32_t n, int32_t in_num,
                  RavelMultiIndexHelper<T> helper, std::vector<const T*> in_dptrs, 
                  T* out);
};

template<typename T>
OF_DEVICE_FUNC void DoIndexToOffset(int32_t n, int32_t in_num,
                  const RavelMultiIndexHelper<T> helper,
                  std::vector<const T*> in_dptrs, T* out) {
  XPU_1D_KERNEL_LOOP(elem_idx, n){
    std::vector<T> index_vec(in_num);
    XPU_1D_KERNEL_LOOP(idx, in_num){
        index_vec[idx] = (in_dptrs.at(idx)[elem_idx]);
    }
    XPU_1D_KERNEL_LOOP(idx, in_num){
        std::cout<<"Index vector element is: "<<index_vec[idx]<<std::endl;
    }
    T offset = helper.NdIndexToOffset(index_vec.data(), in_num);
    std::cout<<"Offset is: "<<offset<<std::endl;
    out[elem_idx] = offset;
  }
}

#define INSTANTIATE_RAVEL_MULTI_INDEX_FUNCTOR(device_type_v, dtype_pair) \
  template struct RavelMultiIndexFunctor<device_type_v, OF_PP_PAIR_FIRST(dtype_pair)>;

}  // namespace user_op
}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_RANGE_KERNEL_UTIL_H_
