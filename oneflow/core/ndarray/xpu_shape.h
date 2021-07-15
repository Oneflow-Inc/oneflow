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
#ifndef ONEFLOW_CORE_NDARRAY_XPU_SHAPE_H_
#define ONEFLOW_CORE_NDARRAY_XPU_SHAPE_H_

#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/shape_view.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/ndarray/xpu_util.h"

namespace oneflow {

template<int NDIMS>
struct XpuShapeUtil;

class XpuShape final {
 public:
  explicit XpuShape(const Shape& shape);
  explicit XpuShape(const ShapeView& shape);
  explicit XpuShape(const ShapeView& shape, int ndims_left_extend_to);
  OF_DEVICE_FUNC XpuShape(const int64_t dim[], int num_axes);
  OF_DEVICE_FUNC XpuShape(const XpuShape&) = default;

  OF_DEVICE_FUNC int64_t At(int64_t dim) const { return dim_[dim]; }
  OF_DEVICE_FUNC int64_t DimElemNum(int64_t dim) const { return dim_elem_num_[dim]; }
  OF_DEVICE_FUNC int64_t Count(int64_t dim) const { return At(dim) * DimElemNum(dim); }

  OF_DEVICE_FUNC size_t ElemNum() const { return elem_num_; }
  OF_DEVICE_FUNC size_t NumAxes() const { return num_axes_; }
  size_t HostElemNum() const { return elem_num_; }
  bool operator==(const XpuShape&) const;
  bool operator!=(const XpuShape& rhs) const { return !(*this == rhs); }

  OF_DEVICE_FUNC void Set(int64_t axis, int64_t value) {
    dim_[axis] = value;
    UpdateDimElemNumAndElemNum();
  }

  template<int NDIMS>
  OF_DEVICE_FUNC int64_t Coordinate2Offset(const int64_t coord[NDIMS]) const {
    return XpuShapeUtil<NDIMS>::Coordinate2Offset(*this, coord);
  }
  template<int NDIMS>
  OF_DEVICE_FUNC void Offset2Coordinate(int64_t offset, int64_t coord[NDIMS]) const {
    XpuShapeUtil<NDIMS>::Offset2Coordinate(*this, offset, coord);
  }

  OF_DEVICE_FUNC void UpdateDimElemNumAndElemNum() {
    elem_num_ = 1;
    for (int i = num_axes_ - 1; i >= 0; --i) {
      dim_elem_num_[i] = elem_num_;
      elem_num_ *= dim_[i];
    }
  }

  std::string ToString() const { return ShapeView(dim_, num_axes_).ToString(); }

 private:
  size_t num_axes_;
  size_t elem_num_;
  int64_t dim_[OF_PP_SEQ_SIZE(DIM_SEQ)];
  int64_t dim_elem_num_[OF_PP_SEQ_SIZE(DIM_SEQ)];
};

template<>
struct XpuShapeUtil<1> final {
  OF_DEVICE_FUNC static int64_t Coordinate2Offset(const XpuShape& shape, const int64_t coord[1]) {
    return coord[0];
  }
  OF_DEVICE_FUNC static void Offset2Coordinate(const XpuShape& shape, int64_t offset,
                                               int64_t coord[1]) {
    coord[0] = offset;
  }
};

#define COORD_MUL_STRIDE(i) coord[i] * shape.DimElemNum(i) +
#define EXTRACT_COORD(i)                   \
  coord[i] = offset / shape.DimElemNum(i); \
  offset %= shape.DimElemNum(i);

#define SPECIALIZE_XPU_SHAPE_UTIL(n)                                                    \
  template<>                                                                            \
  struct XpuShapeUtil<n + 2> final {                                                    \
    OF_DEVICE_FUNC static int64_t Coordinate2Offset(const XpuShape& shape,              \
                                                    const int64_t coord[n + 2]) {       \
      return OF_PP_FOR_EACH_TUPLE(COORD_MUL_STRIDE, GET_SEQ(n)) coord[n + 1];           \
    }                                                                                   \
    OF_DEVICE_FUNC static void Offset2Coordinate(const XpuShape& shape, int64_t offset, \
                                                 int64_t coord[n + 2]) {                \
      OF_PP_FOR_EACH_TUPLE(EXTRACT_COORD, GET_SEQ(n));                                  \
      coord[n + 1] = offset;                                                            \
    }                                                                                   \
  };

SPECIALIZE_XPU_SHAPE_UTIL(0);
SPECIALIZE_XPU_SHAPE_UTIL(1);
SPECIALIZE_XPU_SHAPE_UTIL(2);
SPECIALIZE_XPU_SHAPE_UTIL(3);
SPECIALIZE_XPU_SHAPE_UTIL(4);
#undef SPECIALIZE_XPU_SHAPE_UTIL
#undef EXTRACT_COORD
#undef COORD_MUL_STRIDE

void SimplifyBroadcastShapes(const XpuShape& y, const XpuShape& b, DimVector* simplified_y,
                             DimVector* simplified_b);

void SimplifyBroadcastShapes(const XpuShape& y, const XpuShape& a, const XpuShape& b,
                             DimVector* simplified_y, DimVector* simplified_a,
                             DimVector* simplified_b);
}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_XPU_SHAPE_H_
