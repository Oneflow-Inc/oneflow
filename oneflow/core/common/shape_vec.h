#ifndef ONEFLOW_CORE_COMMON_SHAPE_VEC_H_
#define ONEFLOW_CORE_COMMON_SHAPE_VEC_H_

#include "oneflow/core/common/fixed_vector.h"

namespace oneflow {

//#define DISABLE_FIXED_SHAPE_VEC

#if defined(DISABLE_FIXED_SHAPE_VEC)

typedef std::vector<int64_t> DimVector;
typedef std::vector<int64_t> AxisVector;

#else

#define SHAPE_MAX_AXIS_SIZE 20
typedef fixed_vector<int64_t, SHAPE_MAX_AXIS_SIZE> DimVector;
typedef fixed_vector<int64_t, SHAPE_MAX_AXIS_SIZE> AxisVector;

#endif
}

#endif  // ONEFLOW_CORE_COMMON_SHAPE_VEC_H_
