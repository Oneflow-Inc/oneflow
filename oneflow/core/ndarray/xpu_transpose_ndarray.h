#ifndef ONEFLOW_CORE_NDARRAY_XPU_TRANSPOSE_NDARRAY_H_
#define ONEFLOW_CORE_NDARRAY_XPU_TRANSPOSE_NDARRAY_H_

#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<typename T, int NDIMS, typename X = XpuVarNdarray<T>>
class XpuTransposeNdarray final {
 public:
  OF_DEVICE_FUNC XpuTransposeNdarray(const X& x, const int64_t perm[NDIMS])
      : x_(x), shape_(x.shape()) {
    for (int i = 0; i < NDIMS; ++i) {
      perm_[i] = perm[i];
      shape_.Set(i, x.shape().At(perm[i]));
    }
  }

  template<int ndims, typename = typename std::enable_if<ndims == NDIMS>::type>
  OF_DEVICE_FUNC T Get(int64_t offset) const {
    int64_t coord[NDIMS];
    Offset2Coord(offset, coord);
    return Get(coord);
  }

  template<int ndims, typename = typename std::enable_if<ndims == NDIMS>::type>
  OF_DEVICE_FUNC T* Mut(int64_t offset) const {
    int64_t coord[NDIMS];
    Offset2Coord(offset, coord);
    return Mut(coord);
  }

  template<int ndims, typename = typename std::enable_if<ndims == NDIMS>::type>
  OF_DEVICE_FUNC T Get(int64_t coord[ndims]) const {
    int64_t permuted_coord[NDIMS];
    PermuteCoord(coord, permuted_coord);
    return x_.template Get<ndims>(permuted_coord);
  }

  template<int ndims, typename = typename std::enable_if<ndims == NDIMS>::type>
  OF_DEVICE_FUNC T* Mut(int64_t coord[NDIMS]) const {
    int64_t permuted_coord[NDIMS];
    PermuteCoord(coord, permuted_coord);
    return x_.template Mut<NDIMS>(permuted_coord);
  }

 private:
  OF_DEVICE_FUNC void Offset2Coord(int64_t offset, int64_t coord[NDIMS]) const {
    shape_.template Offset2Coordinate<NDIMS>(offset, coord);
  }

  OF_DEVICE_FUNC void PermuteCoord(const int64_t coord[NDIMS],
                                   int64_t permuted_coord[NDIMS]) const {
    for (int i = 0; i < NDIMS; ++i) { permuted_coord[perm_[i]] = coord[i]; }
  }

  const X& x_;
  XpuShape shape_;
  int64_t perm_[NDIMS];
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_XPU_TRANSPOSE_NDARRAY_H_
