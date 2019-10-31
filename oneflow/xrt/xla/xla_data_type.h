#ifndef ONEFLOW_XRT_XLA_XLA_DATA_TYPE_H_
#define ONEFLOW_XRT_XLA_XLA_DATA_TYPE_H_

#include "oneflow/core/common/data_type.pb.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace oneflow {
namespace xrt {
namespace mola {

// Convert oneflow `DataType` to xla `PrimitiveType`
xla::PrimitiveType DataTypeToPrimitiveType(DataType data_type);

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow

#endif
