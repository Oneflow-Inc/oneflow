/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
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
    return XpuShapeUtil<NDIMS>::Coord2Offset(shape_, coord);
  }
  const X& x_;
  XpuShape shape_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_XPU_RESHAPE_NDARRAY_H_
