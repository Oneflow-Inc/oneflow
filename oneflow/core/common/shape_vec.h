#ifndef ONEFLOW_CORE_COMMON_SHAPE_VEC_H_
#define ONEFLOW_CORE_COMMON_SHAPE_VEC_H_

#include "oneflow/core/common/fixed_vector.h"

namespace oneflow {

#define SHAPE_MAX_AXIS_SIZE 20
typedef fixed_vector<int64_t, SHAPE_MAX_AXIS_SIZE> DimVector;
typedef fixed_vector<int64_t, SHAPE_MAX_AXIS_SIZE> AxisVector;

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_SHAPE_VEC_H_
