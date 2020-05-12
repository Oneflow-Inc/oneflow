#ifndef ONEFLOW_CUSTOMIZED_OPS_RESHAPE_USER_OP_UTIL
#define ONEFLOW_CUSTOMIZED_OPS_RESHAPE_USER_OP_UTIL

#include "oneflow/core/framework/sbp_context.h"
#include "oneflow/core/framework/user_op_conf.h"

namespace oneflow {
struct ReshapeUserOpUtil {
  static Maybe<Shape> GetLogicalOutBlobShape(const Shape& in_shape,
                                             const ShapeProto& reshape_proto);
  static Maybe<void> Squeeze(const Shape& origin, Shape* shape,
                             HashMap<int, int>* squeezed_axis2origin_axis);
  static Maybe<void> GetGroupStartInAxis2OutAxis(const Shape& in_shape, const Shape& out_shape,
                                                 const int64_t parallel_num,
                                                 HashMap<int, int>* group_start_in_axis2out_axis);
  static Maybe<void> GetReshapeUserOpSbpSignatures(const Shape& in_shape, const Shape& out_shape,
                                                   user_op::SbpContext* ctx);
};
}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_OPS_RESHAPE_USER_OP_UTIL
