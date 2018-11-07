#ifndef ONEFLOW_CORE_NDARRAY_NDARRAY_ASSIGN_H_
#define ONEFLOW_CORE_NDARRAY_NDARRAY_ASSIGN_H_

#include "oneflow/core/ndarray/ndarray.h"

namespace oneflow {

template<typename YT, typename XT, typename T = typename YT::dtype, int NDIMS = YT::ndims>
typename std::enable_if<!YT::immutable && !XT::immutable>::type NdArrayAssign(YT* y_ndarray,
                                                                              const XT& x_ndarray) {
  CHECK_EQ(y_ndarray->shape(), x_ndarray.shape());
  T* dst_ptr = nullptr;
  size_t dst_size = 0;
  T* src_ptr = nullptr;
  size_t src_size = 0;
  int64_t cur_index = 0;
  size_t total_elem_cnt = y_ndarray->shape().elem_cnt();
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

template<typename YT, typename XT, typename T = typename YT::dtype, int NDIMS = YT::ndims>
typename std::enable_if<!YT::immutable && XT::immutable && NDIMS == 1>::type NdArrayAssign(
    YT* y_ndarray, const XT& x_ndarray) {
  static_assert(YT::ndims == XT::ndims, "YT::ndims should equals XT::ndims");
  CHECK_EQ(y_ndarray->shape().NumAxes(), 1);
  CHECK_EQ(y_ndarray->shape(), x_ndarray.shape());
  int64_t dim0_size = y_ndarray->shape().At(0);
  FOR_RANGE(int64_t, i, 0, dim0_size) { *y_ndarray->Mut(i) = x_ndarray.Get(i); }
}

template<typename YT, typename XT, typename T = typename YT::dtype, int NDIMS = YT::ndims>
typename std::enable_if<!YT::immutable && XT::immutable && NDIMS == 2>::type NdArrayAssign(
    YT* y_ndarray, const XT& x_ndarray) {
  static_assert(YT::ndims == XT::ndims, "YT::ndims should equals XT::ndims");
  CHECK_EQ(y_ndarray->shape().NumAxes(), 2);
  CHECK_EQ(y_ndarray->shape(), x_ndarray.shape());
  int64_t dim0_size = y_ndarray->shape().At(0);
  int64_t dim1_size = y_ndarray->shape().At(1);
  FOR_RANGE(int64_t, i, 0, dim0_size) {
    FOR_RANGE(int64_t, j, 0, dim1_size) { *y_ndarray->Mut(i, j) = x_ndarray.Get(i, j); }
  }
}

template<typename YT, typename XT, typename T = typename YT::dtype, int NDIMS = YT::ndims>
typename std::enable_if<!YT::immutable && XT::immutable && NDIMS == 3>::type NdArrayAssign(
    YT* y_ndarray, const XT& x_ndarray) {
  static_assert(YT::ndims == XT::ndims, "YT::ndims should equals XT::ndims");
  CHECK_EQ(y_ndarray->shape().NumAxes(), 3);
  CHECK_EQ(y_ndarray->shape(), x_ndarray.shape());
  int64_t dim0_size = y_ndarray->shape().At(0);
  int64_t dim1_size = y_ndarray->shape().At(1);
  int64_t dim2_size = y_ndarray->shape().At(2);
  FOR_RANGE(int64_t, i, 0, dim0_size) {
    FOR_RANGE(int64_t, j, 0, dim1_size) {
      FOR_RANGE(int64_t, k, 0, dim2_size) { *y_ndarray->Mut(i, j, k) = x_ndarray.Get(i, j, k); }
    }
  }
}

template<typename YT, typename XT, typename T = typename YT::dtype, int NDIMS = YT::ndims>
typename std::enable_if<!YT::immutable && XT::immutable && NDIMS == 4>::type NdArrayAssign(
    YT* y_ndarray, const XT& x_ndarray) {
  static_assert(YT::ndims == XT::ndims, "YT::ndims should equals XT::ndims");
  CHECK_EQ(y_ndarray->shape().NumAxes(), 4);
  CHECK_EQ(y_ndarray->shape(), x_ndarray.shape());
  int64_t dim0_size = y_ndarray->shape().At(0);
  int64_t dim1_size = y_ndarray->shape().At(1);
  int64_t dim2_size = y_ndarray->shape().At(2);
  int64_t dim3_size = y_ndarray->shape().At(3);
  FOR_RANGE(int64_t, i, 0, dim0_size) {
    FOR_RANGE(int64_t, j, 0, dim1_size) {
      FOR_RANGE(int64_t, k, 0, dim2_size) {
        FOR_RANGE(int64_t, n, 0, dim3_size) {
          *y_ndarray->Mut(i, j, k, n) = x_ndarray.Get(i, j, k, n);
        }
      }
    }
  }
}

template<typename YT, typename XT, typename T = typename YT::dtype, int NDIMS = YT::ndims>
typename std::enable_if<!YT::immutable && XT::immutable && NDIMS == 5>::type NdArrayAssign(
    YT* y_ndarray, const XT& x_ndarray) {
  static_assert(YT::ndims == XT::ndims, "YT::ndims should equals XT::ndims");
  CHECK_EQ(y_ndarray->shape().NumAxes(), 5);
  CHECK_EQ(y_ndarray->shape(), x_ndarray.shape());
  int64_t dim0_size = y_ndarray->shape().At(0);
  int64_t dim1_size = y_ndarray->shape().At(1);
  int64_t dim2_size = y_ndarray->shape().At(2);
  int64_t dim3_size = y_ndarray->shape().At(3);
  int64_t dim4_size = y_ndarray->shape().At(4);
  FOR_RANGE(int64_t, i, 0, dim0_size) {
    FOR_RANGE(int64_t, j, 0, dim1_size) {
      FOR_RANGE(int64_t, k, 0, dim2_size) {
        FOR_RANGE(int64_t, n, 0, dim3_size) {
          FOR_RANGE(int64_t, m, 0, dim4_size) {
            *y_ndarray->Mut(i, j, k, n, m) = x_ndarray.Get(i, j, k, n, m);
          }
        }
      }
    }
  }
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_NDARRAY_ASSIGN_H_
