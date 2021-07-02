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
#ifndef ONEFLOW_CORE_COMMON_SHAPE_VEC_H_
#define ONEFLOW_CORE_COMMON_SHAPE_VEC_H_

#include "oneflow/core/common/fixed_vector.h"

namespace oneflow {

//#define DISABLE_FIXED_SHAPE_VEC
#define SHAPE_MAX_AXIS_SIZE 20

#if defined(DISABLE_FIXED_SHAPE_VEC)

typedef std::vector<int64_t> DimVector;
typedef std::vector<int64_t> AxisVector;

#else

typedef fixed_vector<int64_t, SHAPE_MAX_AXIS_SIZE> DimVector;
typedef fixed_vector<int64_t, SHAPE_MAX_AXIS_SIZE> AxisVector;

#endif
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_SHAPE_VEC_H_
