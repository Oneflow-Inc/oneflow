#ifndef ONEFLOW_CORE_OPERATOR_RESHAPE_UTIL_H_
#define ONEFLOW_CORE_OPERATOR_RESHAPE_UTIL_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {
Maybe<Shape> GetLogicalOutBlobShape(const Shape& in_shape, const ShapeProto& reshape_proto);
Maybe<void> Squeeze(const Shape& origin, Shape* shape,
                    HashMap<int, int>* squeezed_axis2origin_axis);
Maybe<void> GetGroupStartInAxis2OutAxis(const Shape& in_shape, const Shape& out_shape,
                                        HashMap<int, int>* group_start_in_axis2out_axis);
Maybe<void> GetReshapeSbpSignatures(const Shape& in_shape, const Shape& out_shape,
                                    const PbRpf<std::string>& input_bns,
                                    const PbRpf<std::string>& output_bns,
                                    const int64_t parallel_num, SbpSignatureList* sbp_sig_list);
}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_RESHAPE_UTIL_H_
