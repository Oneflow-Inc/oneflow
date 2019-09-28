#ifndef ONEFLOW_ENGINE_XLA_OF2XLA_XLA_DATA_TYPE_H_
#define ONEFLOW_ENGINE_XLA_OF2XLA_XLA_DATA_TYPE_H_

#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "oneflow/core/common/data_type.pb.h"

namespace oneflow {
namespace mla {

// Convert oneflow `DataType` to xla `PrimitiveType`
xla::PrimitiveType DataTypeToPrimitiveType(DataType data_type);

}  // namespace mla
}  // namespace oneflow

#endif
