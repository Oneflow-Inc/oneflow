#ifndef ONEFLOW_CORE_NDARRAY_XPU_RESHAPE_NDARRAY_H_
#define ONEFLOW_CORE_NDARRAY_XPU_RESHAPE_NDARRAY_H_

namespace oneflow {

template<typename T, int NDIMS, typename X = XpuVarNdarray<T>>
class XpuReshapeNdarray final {
 public:
  OF_DEVICE_FUNC XpuReshapeNdarray(const X& x, const int64_t dim[NDIMS])
      : x_(x), shape_(dim, NDIMS) {}

  template<int ndims = NDIMS>
  OF_DEVICE_FUNC T Get(int64_t offset) const {
    return x_.template Get<ndims>(offset);
  }
  template<int ndims = NDIMS>
  OF_DEVICE_FUNC T* Mut(int64_t offset) const {
    return x_.template Mut<ndims>(offset);
  }
  template<int ndims = NDIMS>
  OF_DEVICE_FUNC T Get(int64_t coord[ndims]) const {
    return Get<ndims>(Coord2Offset(coord));
  }
  template<int ndims = NDIMS>
  OF_DEVICE_FUNC T* Mut(int64_t coord[NDIMS]) const {
    return Get<NDIMS>(Coord2Offset(coord));
  }

 private:
  OF_DEVICE_FUNC int64_t Coord2Offset(const int64_t coord[NDIMS]) const {
    return ExecShapeUtil<NDIMS>::Coord2Offset(shape_, coord);
  }
  const X& x_;
  ExecShape shape_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_XPU_RESHAPE_NDARRAY_H_
