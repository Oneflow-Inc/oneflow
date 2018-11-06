#include "oneflow/core/common/ndarray.h"

namespace oneflow {

template<typename T, int NDIMS>
void AssignVarNdArray(VarNdArray<T, NDIMS>* var_ndarray, const NdArray<T, NDIMS>& ndarray) {
  T* dst_ptr = nullptr;
  size_t dst_size = 0;
  T* src_ptr = nullptr;
  size_t src_size = 0;
  int64_t cur_index = 0;
  CHECK_EQ(var_ndarray->shape(), ndarray.shape());
  size_t total_elem_cnt = var_ndarray->shape().elem_cnt();
  while (cur_index < total_elem_cnt) {
    if (dst_size == 0) { var_ndarray->GetMutPtrAndContiguousSize(cur_index, &dst_ptr, &dst_size); }
    if (src_size == 0) { ndarray.GetMutPtrAndContiguousSize(cur_index, &src_ptr, &src_size); }
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
