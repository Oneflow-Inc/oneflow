#ifndef ONEFLOW_CORE_COMMON_FLAT_SHAPE_H_
#define ONEFLOW_CORE_COMMON_FLAT_SHAPE_H_

#include <memory>
#include "oneflow/core/object_msg/flat_msg.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/shape_vec.h"

namespace oneflow {

class Shape;

// clang-format off

FLAT_MSG_BEGIN(FlatShape);
  // Methods
  OF_PUBLIC Maybe<void> Init(const std::shared_ptr<const Shape>& shape);
  OF_PUBLIC Maybe<void> Check(const std::shared_ptr<const Shape>& shape) const;

  // Fields
	FLAT_MSG_DEFINE_OPTIONAL(int64_t, num_axes);
	FLAT_MSG_DEFINE_REPEATED(int64_t, dim, SHAPE_MAX_AXIS_SIZE);
FLAT_MSG_END(FlatShape);

// clang-format on

}

#endif  // ONEFLOW_CORE_COMMON_FLAT_SHAPE_H_
