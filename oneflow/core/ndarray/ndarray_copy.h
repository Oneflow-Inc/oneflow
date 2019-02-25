#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_COPY_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_COPY_H_

#include "oneflow/core/ndarray/cpu_ndarray.h"

namespace oneflow {

template<typename YT, typename XT, typename T = typename YT::dtype>
void NdarrayCopy(YT* y_ndarray, const XT& x_ndarray) {
  CHECK_EQ(y_ndarray->xpu_shape().ElemNum(), x_ndarray.xpu_shape().ElemNum());
  T* dst_ptr = nullptr;
  size_t dst_size = 0;
  T* src_ptr = nullptr;
  size_t src_size = 0;
  int64_t cur_index = 0;
  size_t total_elem_cnt = y_ndarray->xpu_shape().ElemNum();
  while (cur_index < total_elem_cnt) {
    if (dst_size == 0) { y_ndarray->GetMutPtrAndContiguousSize(cur_index, &dst_ptr, &dst_size); }
    if (src_size == 0) { x_ndarray.GetMutPtrAndContiguousSize(cur_index, &src_ptr, &src_size); }
    if (src_size == 0) { break; }
    size_t cp_size = std::min(dst_size, src_size);
    if (cp_size == 1) {
      *dst_ptr = *src_ptr;
    } else {
      memcpy(dst_ptr, src_ptr, sizeof(T) * cp_size);
    }
    dst_ptr += cp_size;
    src_ptr += cp_size;
    dst_size -= cp_size;
    src_size -= cp_size;
    cur_index += cp_size;
  }
  CHECK_EQ(dst_size, 0);
  CHECK_EQ(src_size, 0);
  CHECK_EQ(cur_index, total_elem_cnt);
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_COPY_H_
